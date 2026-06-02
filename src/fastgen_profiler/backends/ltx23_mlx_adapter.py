"""Local LTX2.3 compatibility adapter for installed mlx_video APIs.

The installed mlx_video package exposes LTXModel/LTXModelConfig but not the
macgen-profile create_ltx23_pipeline(...) contract.  This module adapts those
helpers to the profiler backend boundary.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


_VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER = 6


def _flatten_parameter_names(parameters: Any, prefix: str = "") -> set[str]:
    if isinstance(parameters, dict):
        result: set[str] = set()
        for key, value in parameters.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_parameter_names(value, next_prefix))
        return result
    if prefix:
        return {prefix}
    return set()


class LTX23MLXPipeline:
    """Pipeline object implementing the macgen-profile LTX2.3 phase contract."""

    _MAX_CONFIG_DIMENSION = 65_536

    def __init__(
        self,
        *,
        model_path: Path,
        seed: int,
        width: int,
        height: int,
        frames: int,
        steps: int,
        fps: int = 24,
        guidance: float = 1.0,
        quant: str = "none",
        cache: str = "none",
        compile: str = "off",
        save_video: bool = False,
        dry_run: bool = False,
        text_encoder_dir: Path | str | None = None,
        tokenizer_dir: Path | str | None = None,
        auto_download: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.seed = seed
        self.width = width
        self.height = height
        self.frames = frames
        self.steps = steps
        self.fps = fps
        self.guidance = guidance
        self.quant = quant
        self.cache = cache
        self.compile = compile
        self.save_video = save_video
        self.dry_run = dry_run
        self.text_encoder_dir = Path(text_encoder_dir) if text_encoder_dir else None
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self.auto_download = auto_download

        self.config: Any | None = None
        self.model: Any | None = None
        self.scheduler: Any | None = None
        self.text_proj: Any | None = None
        self.latent_shape: tuple | None = None
        self.context_emb: Any | None = None
        self.cfg_disabled = guidance <= 1.0
        self._mlx_runtime_ready = False
        self._text_encode_started = False
        self._decode_started = False

    def _ensure_mlx_runtime_ready(self, phase: str) -> None:
        if self._mlx_runtime_ready:
            return
        from fastgen_profiler.mlx_guard import (
            check_memory_guard,
            check_run_allocation_budget,
            configure_mlx_resource_limits,
        )

        check_memory_guard(label=f"ltx2.3 {phase}")
        check_run_allocation_budget(
            width=self.width,
            height=self.height,
            frames=self.frames,
            guidance=self.guidance,
            label=f"ltx2.3 {phase}",
        )
        configure_mlx_resource_limits(label=f"ltx2.3 {phase}")
        self._mlx_runtime_ready = True

    def _check_memory(self, phase: str) -> None:
        try:
            from fastgen_profiler.mlx_guard import check_runtime_memory
        except ImportError as exc:
            raise RuntimeError(
                f"memory guard unavailable before ltx2.3 {phase}; refusing to continue without runtime memory checks"
            ) from exc
        check_runtime_memory(label=f"ltx2.3 {phase}")

    def _check_host_allocation(self, required_bytes: int, phase: str) -> None:
        try:
            from fastgen_profiler.mlx_guard import check_host_allocation_headroom
        except ImportError as exc:
            raise RuntimeError(
                f"memory guard unavailable before ltx2.3 {phase}; refusing to continue without host allocation checks"
            ) from exc
        check_host_allocation_headroom(required_bytes, label=f"ltx2.3 {phase}")

    def _check_file_load(self, path: Path, phase: str) -> None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
            raise RuntimeMemoryAbort(
                f"cannot stat {path} before loading {phase}; refusing to load without host allocation preflight"
            ) from exc
        # safetensors load can briefly hold file buffers plus filtered arrays.
        self._check_host_allocation(size * 2, phase)

    def _check_directory_load(self, path: Path, phase: str) -> None:
        if not path.is_dir():
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
            raise RuntimeMemoryAbort(
                f"cannot scan {path} before loading {phase}; refusing to load without host allocation preflight"
            )
        try:
            total = sum(
                file.stat().st_size
                for file in path.rglob("*.safetensors")
                if file.is_file()
            )
        except OSError as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
            raise RuntimeMemoryAbort(
                f"cannot scan {path} before loading {phase}; refusing to load without host allocation preflight"
            ) from exc
        if total > 0:
            self._check_host_allocation(total * 2, phase)

    def _check_tokenizer_load(self, path: Path, phase: str) -> None:
        if not path.is_dir():
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
            raise RuntimeMemoryAbort(
                f"cannot scan {path} before loading {phase}; refusing to load tokenizer without host allocation preflight"
            )
        try:
            total = sum(file.stat().st_size for file in path.iterdir() if file.is_file())
        except OSError as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
            raise RuntimeMemoryAbort(
                f"cannot scan {path} before loading {phase}; refusing to load tokenizer without host allocation preflight"
            ) from exc
        if total > 0:
            self._check_host_allocation(total * 4, phase)

    def _expected_latent_shape(self) -> tuple[int, int, int, int, int]:
        channels = int(self.config.in_channels) if self.config is not None else 128
        return (1, channels, self.frames, _latent_grid(self.height), _latent_grid(self.width))

    def _validate_latent_init_shape(self, *, width: int, height: int, frames: int) -> None:
        expected = (self.width, self.height, self.frames)
        actual = (width, height, frames)
        if actual != expected:
            raise RuntimeError(
                f"latent_init shape {actual} must match pipeline shape {expected}; "
                "create a fresh pipeline for a different run shape"
            )

    def _validate_latents_shape(self, latents: Any, phase: str) -> None:
        actual = tuple(getattr(latents, "shape", ()))
        expected = self._expected_latent_shape()
        if actual != expected:
            raise RuntimeError(
                f"latent shape {actual} for {phase} does not match expected {expected}; "
                "refusing to allocate derived MLX tensors for an unexpected run shape"
            )

    def _expected_frame_shape(self) -> tuple[int, int, int, int]:
        return (self.frames, self.height, self.width, 3)

    def _validate_frame_shape(self, frames: Any, phase: str) -> None:
        actual = tuple(getattr(frames, "shape", ()))
        expected = self._expected_frame_shape()
        if actual != expected:
            raise RuntimeError(
                f"decoded LTX2.3 frames must have shape [T,H,W,3] {expected} for {phase}, got {actual}"
            )

    def _preflight_config_shape(self, config_path: Path) -> dict[str, Any] | None:
        if not config_path.exists():
            raise FileNotFoundError(
                "LTX2.3 transformer config not found. Expected at:\n"
                f"  {config_path}\n"
                "Refusing to initialize MLX before local model config is present."
            )
        self._check_file_load(config_path, "preflight transformer config")
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in _iter_config_numbers(config_dict):
            if value <= 0 and _is_structural_config_key(key):
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"LTX2.3 config field {key}={value} must be a positive structural dimension; "
                    "refusing to construct MLX model"
                )
            if value > self._MAX_CONFIG_DIMENSION and _is_structural_config_key(key):
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"LTX2.3 config field {key}={value} exceeds safe structural dimension "
                    f"{self._MAX_CONFIG_DIMENSION}; refusing to construct MLX model"
                )

        channels = _positive_structural_int(config_dict.get("in_channels", 128), "in_channels")
        latent_h = _latent_grid(self.height)
        latent_w = _latent_grid(self.width)
        self._check_host_allocation(
            channels * self.frames * latent_h * latent_w * 4 * 4,
            "config latent tensor",
        )
        self._preflight_model_config(config_dict, "config model tensor")
        return config_dict

    def _preflight_model_config(self, config: dict[str, Any], phase: str) -> None:
        hidden_size = _positive_structural_int(config.get("hidden_size", config.get("caption_channels", 4096)), "hidden_size")
        intermediate_size = _positive_structural_int(
            config.get("intermediate_size", hidden_size * 4),
            "intermediate_size",
        )
        layers = _positive_structural_int(config.get("num_layers", config.get("num_hidden_layers", 1)), "num_layers")
        heads = _positive_structural_int(
            config.get("num_attention_heads", config.get("num_heads", 1)),
            "num_attention_heads",
        )
        for key, value in {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_layers": layers,
            "num_attention_heads": heads,
        }.items():
            if value <= 0:
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"LTX2.3 config field {key}={value} must be a positive structural dimension; "
                    "refusing to construct MLX model"
                )
            if value > self._MAX_CONFIG_DIMENSION:
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"LTX2.3 config field {key}={value} exceeds safe structural dimension "
                    f"{self._MAX_CONFIG_DIMENSION}; refusing to construct MLX model"
                )
        attention_floor = layers * hidden_size * hidden_size * 4
        mlp_floor = layers * hidden_size * intermediate_size * 3
        head_floor = layers * heads * hidden_size
        self._check_host_allocation((attention_floor + mlp_floor + head_floor) * 2, phase)

    def _preflight_text_model_config(self, text_config: dict[str, Any], phase: str) -> None:
        hidden_size = _positive_structural_int(text_config.get("hidden_size", 4096), "hidden_size")
        intermediate_size = _positive_structural_int(
            text_config.get("intermediate_size", hidden_size * 4),
            "intermediate_size",
        )
        layers = _positive_structural_int(
            text_config.get("num_hidden_layers", text_config.get("num_layers", 1)),
            "num_hidden_layers",
        )
        vocab_size = _positive_structural_int(text_config.get("vocab_size", 0), "vocab_size")
        heads = _positive_structural_int(text_config.get("num_attention_heads", 1), "num_attention_heads")
        for key, value in {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": layers,
            "vocab_size": vocab_size,
            "num_attention_heads": heads,
        }.items():
            if value <= 0:
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"LTX2.3 text encoder config field {key}={value} must be a positive structural dimension; "
                    "refusing to construct MLX text model"
                )
            if value > self._MAX_CONFIG_DIMENSION:
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"LTX2.3 text encoder config field {key}={value} exceeds safe structural dimension "
                    f"{self._MAX_CONFIG_DIMENSION}; refusing to construct MLX text model"
                )
        attention_floor = layers * hidden_size * hidden_size * 4
        mlp_floor = layers * hidden_size * intermediate_size * 3
        embedding_floor = vocab_size * hidden_size if vocab_size > 0 else 0
        self._check_host_allocation((attention_floor + mlp_floor + embedding_floor) * 2, phase)

    def _preflight_text_encoder_assets(self, text_encoder_dir: Path) -> list[str]:
        config_path = text_encoder_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "LTX2.3 text encoder config not found. Expected at:\n"
                f"  {config_path}\n"
                "Refusing to initialize MLX text encoder before local config is present."
            )
        shards = sorted([f.name for f in text_encoder_dir.iterdir() if f.is_file() and f.name.endswith(".safetensors")])
        if not shards:
            raise FileNotFoundError(
                "LTX2.3 text encoder weights not found. Expected at least one .safetensors shard in:\n"
                f"  {text_encoder_dir}\n"
                "Refusing to initialize MLX text encoder before local weights are present."
            )
        return shards

    def _preflight_text_prompt_tokens(self, prompt: str, text_encoder_dir: Path, tokenizer_dir: Path) -> None:
        self._check_tokenizer_load(tokenizer_dir, "read tokenizer")
        self._preflight_text_encoder_assets(text_encoder_dir)
        config_path = text_encoder_dir / "config.json"
        self._check_file_load(config_path, "preflight text_encoder config")
        with open(config_path) as f:
            full_config = json.load(f)
        text_config = full_config["text_config"]
        self._preflight_text_model_config(text_config, "text_encoder model config")
        from transformers import AutoTokenizer
        from fastgen_profiler.mlx_guard import check_token_sequence_budget

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"][0]
        check_token_sequence_budget(
            token_count=len(input_ids),
            max_tokens=int(text_config["max_position_embeddings"]),
            hidden_size=int(text_config["hidden_size"]),
            label="ltx2.3 text_encoder",
        )

    def synchronize(self, target: object | None = None) -> None:
        if target is None:
            return
        mx = sys.modules.get("mlx.core")
        if mx is None:
            return
        try:
            mx.eval(target)
        except Exception as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort, mlx_cleanup

            mlx_cleanup()
            raise RuntimeMemoryAbort(
                "Runtime memory abort [ltx2.3 synchronize]: MLX synchronization failed; "
                "aborting because Metal runtime state may be unsafe."
            ) from exc

    def _eval_mlx(self, mx: Any, *targets: Any, phase: str) -> None:
        try:
            mx.eval(*targets)
        except Exception as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

            _cleanup_loaded_runtime_after_error()
            raise RuntimeMemoryAbort(
                f"Runtime memory abort [ltx2.3 {phase}]: MLX eval failed; "
                "aborting because Metal runtime state may be unsafe."
            ) from exc

    def load_model(self) -> dict[str, object]:
        if self.model is not None:
            raise RuntimeError(
                "LTX2.3 MLX model is already loaded in this pipeline; create a fresh "
                "pipeline/process before loading again to avoid accumulating Metal state."
            )
        transformer_dir = self.model_path / "transformer"
        self._check_directory_load(transformer_dir, "preflight transformer")
        config_path = transformer_dir / "config.json"
        config_dict = self._preflight_config_shape(config_path)
        shards = sorted([f for f in os.listdir(transformer_dir) if f.endswith(".safetensors")])
        if not shards:
            raise FileNotFoundError(
                "LTX2.3 transformer weights not found. Expected at least one .safetensors shard in:\n"
                f"  {transformer_dir}\n"
                "Refusing to initialize MLX before local model weights are present."
            )
        if importlib.util.find_spec("mlx_video") is None:
            raise ModuleNotFoundError(
                "mlx_video is required for the LTX2.3 adapter; dependency check "
                "failed before initializing MLX"
            )
        self._ensure_mlx_runtime_ready("load_model")

        try:
            import mlx.core as mx
            from mlx_video.models.ltx_2.config import LTXModelConfig
            from mlx_video.models.ltx_2.ltx_2 import LTXModel
            from mlx_video.models.ltx_2.utils import load_safetensors

            self._check_memory("model_construct before")
            # Load config JSON directly to preserve original model structure
            if config_dict is None:
                with open(config_path) as f:
                    config_dict = json.load(f)
            self.config = LTXModelConfig(**config_dict)
            self.model = LTXModel(self.config)
            self._check_memory("model_construct after")

            model_param_names = _flatten_parameter_names(self.model.parameters())
            if not model_param_names:
                raise RuntimeError(
                    "LTX2.3 transformer exposed no model parameters; refusing to continue "
                    "because weight loading cannot be verified"
                )

            # Load all sharded weights from transformer directory
            loaded_keys: set[str] = set()
            for shard in shards:
                shard_path = transformer_dir / shard
                self._check_file_load(shard_path, f"read {shard}")
                self._check_memory(f"load_safetensors before {shard}")
                weights = load_safetensors(shard_path)
                self._check_memory(f"load_safetensors after {shard}")
                filtered_items = [
                    (key, value)
                    for key, value in weights.items()
                    if key in model_param_names
                    and not key.endswith(".input_scale")
                    and not key.endswith(".weight_scale")
                ]
                if filtered_items:
                    self._check_memory(f"model_load before {shard}")
                    self.model.load_weights(filtered_items, strict=False)
                    self._eval_mlx(mx, self.model.parameters(), phase=f"model_load {shard}")
                    loaded_keys.update(key for key, _ in filtered_items)
                    self._check_memory(f"model_load after {shard}")
                del weights, filtered_items
                gc.collect()
                mx.clear_cache()

            if not loaded_keys:
                raise RuntimeError(
                    "LTX2.3 transformer weights did not match any model parameters; "
                    "refusing to continue with an uninitialized transformer"
                )

            missing = model_param_names - loaded_keys
            if missing:
                extra_missing = {k for k in missing if "positional_embedding_max_pos" in k}
                if extra_missing != missing:
                    raise RuntimeError(f"Missing {len(missing)} model params not in weights: {sorted(missing)[:10]}")

            self.model.eval()
            self._eval_mlx(mx, self.model.parameters(), phase="model_eval")

            return {"model_type": self.config.model_type if self.config else "ltx2.3",
                    "width": self.width, "height": self.height}
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def prepare_prompt(self, *, prompt: str, negative_prompt: str | None) -> dict[str, str]:
        if self.config is None:
            raise RuntimeError("prepare_prompt called before load_model")
        resolved_negative = negative_prompt if negative_prompt else ""
        from fastgen_profiler.mlx_guard import check_text_prompt_budget
        check_text_prompt_budget(
            prompt=prompt,
            negative_prompt=resolved_negative,
            label="ltx2.3 prompt",
        )
        return {"prompt": prompt, "negative_prompt": resolved_negative}

    def encode_text(self, prepared_prompt: dict[str, str]) -> Any:
        if self.model is None or self.config is None:
            raise RuntimeError("encode_text called before load_model")
        if self._text_encode_started:
            raise RuntimeError(
                "LTX2.3 text encoding has already started in this pipeline; create a fresh "
                "pipeline/process before encoding again to avoid accumulating Metal state."
            )

        from fastgen_profiler.mlx_guard import check_text_prompt_budget

        check_text_prompt_budget(
            prompt=prepared_prompt["prompt"],
            negative_prompt=prepared_prompt["negative_prompt"],
            label="ltx2.3 direct text",
        )
        # Resolve and preflight all local text assets before opening MLX/Metal.
        from fastgen_profiler.backends.ltx23_text_encoder_download import ensure_text_encoder

        if self.text_encoder_dir and self.tokenizer_dir:
            text_encoder_dir = self.text_encoder_dir
            tokenizer_dir = self.tokenizer_dir
        else:
            dest = self.model_path.parent / "LTX-2-text-local"
            text_encoder_dir, tokenizer_dir = ensure_text_encoder(dest, auto_download=self.auto_download)
        self._preflight_text_prompt_tokens(prepared_prompt["prompt"], text_encoder_dir, tokenizer_dir)
        text_proj_path = self.model_path / "text_projections" / "model.safetensors"
        if not text_proj_path.exists():
            raise FileNotFoundError(
                "LTX2.3 text projection weights not found. Expected at:\n"
                f"  {text_proj_path}\n"
                "Refusing to initialize MLX text projection before local weights are present."
            )
        self._check_file_load(text_proj_path, "read text_projection")

        in_features = self.config.caption_channels if self.config else 4096
        hidden_size = self.config.cross_attention_dim if self.config else 4096
        self._check_host_allocation(in_features * hidden_size * 8, "text_projection construct")
        self._text_encode_started = True
        self._ensure_mlx_runtime_ready("encode_text")

        try:
            import mlx.core as mx
            from mlx_video.models.ltx_2.utils import load_safetensors
            from mlx_video.models.ltx_2.text_projection import PixArtAlphaTextProjection

            self._check_memory("text_projection construct before")
            self.text_proj = PixArtAlphaTextProjection(in_features, hidden_size)
            self._check_memory("text_projection construct after")

            self._check_memory("text_projection load_safetensors before")
            tp_weights = load_safetensors(text_proj_path)
            self._check_memory("text_projection load_safetensors after")
            tp_model_params = set()
            for k in self.text_proj.parameters():
                tp_model_params.add(k)
            filtered_tp = {k: v for k, v in tp_weights.items() if k in tp_model_params}
            if not filtered_tp:
                raise RuntimeError(
                    "LTX2.3 text projection weights did not match any text projection parameters; "
                    "refusing to continue with an uninitialized projection"
                )
            self.text_proj.load_weights(list(filtered_tp.items()))

            self.text_proj.eval()
            self._eval_mlx(mx, self.text_proj.parameters(), phase="text_projection parameters")

            context_emb = self._encode_with_gemma3(
                prepared_prompt["prompt"], text_encoder_dir, tokenizer_dir, in_features
            )

            self.context_emb = context_emb

            del self.text_proj
            self.text_proj = None
            gc.collect()
            mx.clear_cache()
            return self.context_emb
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def _encode_with_gemma3(
        self, prompt: str, text_encoder_dir: Path, tokenizer_dir: Path, in_features: int
    ) -> Any:
        """Encode text using real Gemma3 model from LTX-2-text-local."""
        self._check_directory_load(text_encoder_dir, "preflight text_encoder")
        self._check_tokenizer_load(tokenizer_dir, "read tokenizer")
        shards = self._preflight_text_encoder_assets(text_encoder_dir)
        config_path = text_encoder_dir / "config.json"
        self._check_file_load(config_path, "read text_encoder config")
        with open(config_path) as f:
            full_config = json.load(f)
        text_config = full_config["text_config"]
        self._preflight_text_model_config(text_config, "text_encoder model config")
        from transformers import AutoTokenizer
        from fastgen_profiler.mlx_guard import check_token_sequence_budget

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"][0]
        check_token_sequence_budget(
            token_count=len(input_ids),
            max_tokens=int(text_config["max_position_embeddings"]),
            hidden_size=int(text_config["hidden_size"]),
            label="ltx2.3 text_encoder",
        )

        self._ensure_mlx_runtime_ready("text_encoder")

        try:
            import mlx.core as mx
            from mlx_video.models.ltx_2.utils import load_safetensors
            from mlx_lm.models.gemma3_text import ModelArgs, Gemma3Model

            self._check_memory("text_encoder construct before")

            # Build ModelArgs
            model_args = ModelArgs(
                model_type=text_config["model_type"],
                hidden_size=text_config["hidden_size"],
                num_hidden_layers=text_config["num_hidden_layers"],
                intermediate_size=text_config["intermediate_size"],
                num_attention_heads=text_config["num_attention_heads"],
                head_dim=text_config["head_dim"],
                rms_norm_eps=text_config["rms_norm_eps"],
                vocab_size=text_config["vocab_size"],
                num_key_value_heads=text_config["num_key_value_heads"],
                rope_theta=text_config["rope_theta"],
                rope_local_base_freq=text_config.get("rope_local_base_freq", 10000),
                query_pre_attn_scalar=text_config["query_pre_attn_scalar"],
                sliding_window=text_config["sliding_window"],
                sliding_window_pattern=text_config["sliding_window_pattern"],
                max_position_embeddings=text_config["max_position_embeddings"],
                rope_scaling=text_config.get("rope_scaling"),
            )

            # Create and load Gemma3 text model
            text_model = Gemma3Model(model_args)
            self._check_memory("text_encoder construct after")
            text_model_params = _flatten_parameter_names(text_model.parameters())
            if not text_model_params:
                raise RuntimeError(
                    "LTX2.3 text encoder exposed no model parameters; refusing to continue "
                    "because weight loading cannot be verified"
                )

            # Load weights from shards, mapping keys
            loaded_keys: set[str] = set()
            for shard in shards:
                shard_path = text_encoder_dir / shard
                self._check_file_load(shard_path, f"read text_encoder {shard}")
                self._check_memory(f"text_encoder load_safetensors before {shard}")
                w = load_safetensors(shard_path)
                self._check_memory(f"text_encoder load_safetensors after {shard}")
                mapped_items = []
                for k, v in w.items():
                    if k.startswith("language_model.model."):
                        new_key = k[len("language_model.model."):]
                    elif k.startswith("model."):
                        new_key = k[len("model."):]
                    else:
                        continue
                    if new_key in text_model_params:
                        mapped_items.append((new_key, v))
                if mapped_items:
                    self._check_memory(f"text_encoder before {shard}")
                    text_model.load_weights(mapped_items, strict=False)
                    self._eval_mlx(mx, text_model.parameters(), phase=f"text_encoder {shard}")
                    loaded_keys.update(key for key, _ in mapped_items)
                    self._check_memory(f"text_encoder after {shard}")
                del w, mapped_items
                gc.collect()
                mx.clear_cache()

            if not loaded_keys:
                raise RuntimeError(
                    "LTX2.3 text encoder weights did not match any Gemma3 text model parameters; "
                    "refusing to continue with an uninitialized text encoder"
                )

            text_model.eval()
            self._eval_mlx(mx, text_model.parameters(), phase="text_encoder parameters")

            # Run text encoding
            input_ids_mx = mx.array(input_ids).reshape(1, -1)
            hidden_states = text_model(input_ids_mx)
            self._eval_mlx(mx, hidden_states, phase="text_encoder hidden states")

            # Pool embeddings (mean pooling) -> (1, hidden_size)
            pooled = hidden_states.mean(axis=1)

            # Project through PixArtAlphaTextProjection
            context_emb = self.text_proj(pooled)
            self._eval_mlx(mx, context_emb, phase="text_encoder context projection")

            # Clean up text encoder
            del text_model, hidden_states, pooled, input_ids_mx
            del tokenizer
            gc.collect()
            mx.clear_cache()

            return context_emb
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def init_latents(self, *, seed: int, width: int, height: int, frames: int) -> Any:
        if self.model is None or self.config is None:
            raise RuntimeError("init_latents called before load_model")
        self._validate_latent_init_shape(width=width, height=height, frames=frames)
        self._ensure_mlx_runtime_ready("latent_init")

        try:
            import mlx.core as mx

            self._check_memory("latent_init before")
            mx.random.seed(seed)
            # Official shape: (B, C=128, T, H, W) — channel first
            latent_h = _latent_grid(height)
            latent_w = _latent_grid(width)
            c = self.config.in_channels
            self._latent_hw = (latent_h, latent_w)
            self._check_host_allocation(
                c * frames * latent_h * latent_w * 4 * 4,
                "latent_init tensor",
            )
            latents = mx.random.normal((1, c, frames, latent_h, latent_w))
            self._eval_mlx(mx, latents, phase="latent_init")
            self._check_memory("latent_init after")
            return latents
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def denoise_step(self, latents: Any, *, step_index: int, steps: int, guidance: float, cache: str) -> Any:
        """Single denoise step matching official denoise_distilled logic."""
        if self.model is None or self.config is None:
            raise RuntimeError("denoise_step called before load_model")
        self._validate_latents_shape(latents, f"denoise {step_index + 1}/{steps}")

        self._check_memory(f"denoise {step_index + 1}/{steps} before")
        self._ensure_mlx_runtime_ready("denoise")

        try:
            import mlx.core as mx
            from mlx_video.models.ltx_2.transformer import Modality

            dtype = latents.dtype
            b, c, f, h, w = latents.shape
            num_tokens = f * h * w
            latent_elements = b * c * f * h * w
            position_elements = b * 3 * num_tokens * 2
            timestep_elements = b * num_tokens
            denoise_floor_bytes = (
                latent_elements * 4 * 8
                + position_elements * 8 * 2
                + timestep_elements * 4 * 4
            )
            self._check_host_allocation(
                denoise_floor_bytes,
                f"denoise {step_index + 1}/{steps} tensors",
            )

            # Compute sigma schedule (linear 1.0 → 0.0)
            np = _numpy()
            sigmas = mx.array(np.linspace(1.0, 0.0, steps + 1), dtype=mx.float32)
            sigma = float(sigmas[step_index])
            sigma_next = float(sigmas[step_index + 1])

            # Flatten: (B, C, T*H*W) → (B, T*H*W, C)
            latents_flat = mx.transpose(
                mx.reshape(latents, (b, c, -1)), (0, 2, 1)
            ).astype(dtype)

            # Timesteps per token
            timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)

            # Positions: (batch, 3, num_tokens, 2)
            frame_idx = mx.broadcast_to(mx.arange(f).reshape(-1, 1, 1), (f, h, w)).reshape(-1)
            y_idx = mx.broadcast_to(mx.arange(h).reshape(1, -1, 1), (f, h, w)).reshape(-1)
            x_idx = mx.broadcast_to(mx.arange(w).reshape(1, 1, -1), (f, h, w)).reshape(-1)
            positions = mx.expand_dims(
                mx.stack([
                    mx.stack([frame_idx, frame_idx], axis=-1),
                    mx.stack([y_idx, y_idx], axis=-1),
                    mx.stack([x_idx, x_idx], axis=-1),
                ], axis=0), axis=0
            )

            modality = Modality(
                latent=latents_flat,
                timesteps=timesteps,
                positions=positions,
                context=self.context_emb,
                context_mask=None,
                enabled=True,
                sigma=mx.full((b,), sigma, dtype=dtype),
            )

            # Forward pass → velocity prediction
            velocity, _ = self.model(video=modality, audio=None)
            self._eval_mlx(mx, velocity, phase=f"denoise {step_index + 1}/{steps} velocity")

            # Velocity → denoised (x0): x0 = latent - timestep * velocity
            sigma_f32 = mx.array(sigma, dtype=mx.float32)
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
            denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

            self._eval_mlx(mx, denoised, phase=f"denoise {step_index + 1}/{steps} denoised")

            # Euler step
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                next_latents = denoised + sigma_next_f32 * (latents.astype(mx.float32) - denoised) / sigma_f32
            else:
                next_latents = denoised

            self._eval_mlx(mx, next_latents, phase=f"denoise {step_index + 1}/{steps} next_latents")
            next_latents = next_latents.astype(dtype)
            self._check_memory(f"denoise {step_index + 1}/{steps} after")
            return next_latents
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def decode(self, latents: Any) -> Any:
        if self.model is None or self.config is None:
            raise RuntimeError("decode called before load_model")
        if self._decode_started:
            raise RuntimeError(
                "LTX2.3 decode has already started in this pipeline; create a fresh "
                "pipeline/process before decoding again to avoid accumulating Metal state."
            )
        self._validate_latents_shape(latents, "decode")

        vae_decoder_dir = self.model_path / "vae" / "decoder"
        if not vae_decoder_dir.exists():
            raise FileNotFoundError(
                "No VAE decoder found. Expected at:\n"
                f"  {vae_decoder_dir}/\n"
                "Ensure the LTX-2.3-distilled-mlx model includes vae/decoder/."
            )
        self._check_directory_load(vae_decoder_dir, "preflight vae_decoder")
        vae_shards = sorted(
            path.name
            for path in vae_decoder_dir.iterdir()
            if path.is_file() and path.name.endswith(".safetensors")
        )
        if not vae_shards:
            raise FileNotFoundError(
                "LTX2.3 VAE decoder weights not found. Expected at least one .safetensors file in:\n"
                f"  {vae_decoder_dir}\n"
                "Refusing to initialize MLX VAE before local weights are present."
            )
        upscaler_path = self.model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        if upscaler_path.exists():
            self._check_file_load(upscaler_path, "preflight upsampler")

        self._decode_started = True
        self._ensure_mlx_runtime_ready("decode")

        try:
            import mlx.core as mx
            from mlx_video.models.ltx_2.video_vae.decoder import VideoDecoder
            from mlx_video.models.ltx_2.upsampler import load_upsampler, upsample_latents

            self._check_memory("vae_decode before")
            # 1. Latents are already (B, C, T, H, W) from denoise

            # 2. Load spatial upscaler (2x spatial upscale in latent space)
            if upscaler_path.exists():
                self._check_memory("upsampler_load before")
                upsampler, _ = load_upsampler(str(upscaler_path))
                self._eval_mlx(mx, upsampler.parameters(), phase="upsampler parameters")
                self._check_memory("upsampler_load after")

                self._check_directory_load(vae_decoder_dir, "read upsampler_vae_stats")
                vae_temp = VideoDecoder.from_pretrained(str(vae_decoder_dir))
                self._eval_mlx(mx, vae_temp.parameters(), phase="upsampler vae stats")
                self._check_memory("upsampler_vae_stats after")

                latents = upsample_latents(
                    latents,
                    upsampler,
                    vae_temp.per_channel_statistics.mean,
                    vae_temp.per_channel_statistics.std,
                )
                self._eval_mlx(mx, latents, phase="upsample_latents")
                self._check_memory("upsample_latents after")
                del upsampler, vae_temp
                mx.clear_cache()

            # 3. VAE decode
            self._check_directory_load(vae_decoder_dir, "read vae_decoder")
            self._check_memory("vae_load before")
            vae = VideoDecoder.from_pretrained(str(vae_decoder_dir))
            self._eval_mlx(mx, vae.parameters(), phase="vae parameters")
            self._check_memory("vae_load after")

            self._check_host_allocation(self.frames * self.height * self.width * 3 * 4 * 4, "vae output tensor")
            video = vae(latents)
            self._eval_mlx(mx, video, phase="vae video")
            self._check_memory("vae_forward after")

            # 4. Post-process: (B, 3, T, H, W) → (T, H, W, 3) uint8
            video = mx.squeeze(video, axis=0)
            transposed = mx.transpose(video, (1, 2, 3, 0))
            self._validate_frame_shape(transposed, "decode")
            self._check_host_allocation(math.prod(transposed.shape) * 13, "numpy_frames")
            np = _numpy()
            frames = np.array(transposed)

            del vae, video, transposed
            gc.collect()
            mx.clear_cache()
            self._check_memory("numpy_frames after")

            frames = _normalize_video_frames(np, frames)

            gc.collect()
            self._check_memory("vae_decode after")
            return frames
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def encode_video(self, frames: Any, *, fps: int) -> Any | Path:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise RuntimeError(f"decoded LTX2.3 frames must have shape [T,H,W,3], got {frames.shape}")
        self._validate_frame_shape(frames, "video_encode")
        if self.dry_run or not self.save_video:
            return frames

        self._check_host_allocation(
            frames.nbytes * _VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER,
            "video_encode frames",
        )
        self._check_memory("video_encode before")
        if importlib.util.find_spec("mlx_video") is None:
            raise ModuleNotFoundError(
                "mlx_video is required for LTX2.3 video postprocess; dependency check "
                "failed before initializing MLX"
            )
        self._ensure_mlx_runtime_ready("video_encode")
        temp_path: Path | None = None
        try:
            from mlx_video.models.ltx_2.postprocess import save_video

            handle = tempfile.NamedTemporaryFile(prefix="macgen-ltx23-", suffix=".mp4", delete=False)
            temp_path = Path(handle.name)
            handle.close()
            save_video(frames, str(temp_path), fps=fps)
            self._check_memory("video_encode after")
            return temp_path
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            _cleanup_loaded_runtime_after_error()
            raise

    def write_output(self, video: Any | Path, output_dir: Path, *, run_id: str) -> Path:
        self._check_memory("file_write before")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{run_id}.mp4"
        if isinstance(video, Path):
            video.replace(output_path)
            return output_path

        self._validate_frame_shape(video, "file_write")
        self._check_host_allocation(
            video.nbytes * _VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER,
            "file_write frames",
        )
        if importlib.util.find_spec("mlx_video") is None:
            raise ModuleNotFoundError(
                "mlx_video is required for LTX2.3 video postprocess; dependency check "
                "failed before initializing MLX"
            )
        self._ensure_mlx_runtime_ready("file_write")
        try:
            from mlx_video.models.ltx_2.postprocess import save_video

            save_video(video, str(output_path), fps=self.fps)
            self._check_memory("file_write after")
            return output_path
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise


def create_ltx23_pipeline(**kwargs: Any) -> LTX23MLXPipeline:
    return LTX23MLXPipeline(**kwargs)


def _latent_grid(size: int) -> int:
    return max(1, (size + 7) // 8)


def _positive_structural_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"LTX2.3 config field {key}={value!r} must be a positive structural dimension; "
            "refusing to construct MLX model"
        )
    result = value
    if result <= 0:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"LTX2.3 config field {key}={result} must be a positive structural dimension; "
            "refusing to construct MLX model"
        )
    return result


def _numpy() -> Any:
    import numpy as np

    return np


def _normalize_video_frames(np: Any, frames: Any) -> Any:
    """Normalize VAE float frames to uint8 with minimal host-side copies."""
    np.add(frames, 1.0, out=frames)
    np.multiply(frames, 127.5, out=frames)
    np.clip(frames, 0, 255, out=frames)
    return frames.astype(np.uint8, copy=False)


def _cleanup_loaded_runtime_after_error(exc: BaseException | None = None) -> None:
    if exc is not None:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        if isinstance(exc, RuntimeMemoryAbort):
            return
    try:
        from fastgen_profiler.mlx_guard import mlx_cleanup

        mlx_cleanup()
    except Exception:
        pass


def _iter_config_numbers(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_config_numbers(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from _iter_config_numbers(child, child_prefix)
    elif isinstance(value, int) and not isinstance(value, bool):
        yield prefix, value


def _is_structural_config_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        marker in lowered
        for marker in (
            "channel",
            "dim",
            "head",
            "hidden",
            "layer",
            "length",
            "patch",
            "rank",
            "size",
        )
    )
