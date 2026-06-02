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

from fastgen_profiler.metrics import MAX_RUN_DIMENSION, MAX_RUN_FPS, MAX_RUN_FRAMES, MAX_RUN_STEPS


_VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER = 6
_MAX_CONFIG_JSON_BYTES = 1 * 1024 * 1024
_MAX_PRELOAD_SCAN_DIRS = 10_000
_MAX_PRELOAD_SCAN_FILES = 10_000
_MAX_DENOISE_STEPS = 512
_MAX_FILTERED_WEIGHT_ITEMS = 100_000
_MAX_PARAMETER_NAMES = 100_000
_MAX_PARAMETER_NAME_CHARS = 1024
_MAX_PARAMETER_NAME_DEPTH = 256
_MAX_CONFIG_JSON_ITEMS = 10_000
_MAX_CONFIG_JSON_DEPTH = 32


def _dependency_available(module_name: str) -> bool:
    module_prefix = f"{module_name}."
    return (
        module_name in sys.modules
        or any(name.startswith(module_prefix) for name in sys.modules)
        or importlib.util.find_spec(module_name) is not None
    )


def _flatten_parameter_names(parameters: Any, prefix: str = "", *, label: str = "parameters") -> set[str]:
    result: set[str] = set()
    stack: list[tuple[Any, str, int]] = [(parameters, prefix, 0)]
    while stack:
        current, current_prefix, depth = stack.pop()
        if depth > _MAX_PARAMETER_NAME_DEPTH:
            _raise_runtime_abort(
                f"{label} nesting exceeds {_MAX_PARAMETER_NAME_DEPTH}; "
                "refusing unbounded parameter-name traversal"
            )
        if isinstance(current, dict):
            for key, value in current.items():
                key_text = _parameter_key_text(key, label=label)
                next_prefix = _join_parameter_name(current_prefix, key_text, label=label)
                stack.append((value, next_prefix, depth + 1))
            continue
        if current_prefix:
            _add_parameter_name(result, current_prefix, label=label)
    return result


def _parameter_key_text(key: Any, *, label: str) -> str:
    if not isinstance(key, str):
        key_type = type(key)
        _raise_runtime_abort(
            f"{label} key <{key_type.__module__}.{key_type.__qualname__}> is not a string; "
            "refusing unbounded parameter-name materialization"
        )
    if len(key) > _MAX_PARAMETER_NAME_CHARS:
        _raise_runtime_abort(
            f"{label} key exceeds {_MAX_PARAMETER_NAME_CHARS} characters; "
            "refusing unbounded parameter-name materialization"
        )
    return key


def _join_parameter_name(prefix: str, key: str, *, label: str) -> str:
    name = f"{prefix}.{key}" if prefix else key
    if len(name) > _MAX_PARAMETER_NAME_CHARS:
        _raise_runtime_abort(
            f"{label} name exceeds {_MAX_PARAMETER_NAME_CHARS} characters; "
            "refusing unbounded parameter-name materialization"
        )
    return name


def _add_parameter_name(result: set[str], name: str, *, label: str) -> None:
    if len(name) > _MAX_PARAMETER_NAME_CHARS:
        _raise_runtime_abort(
            f"{label} name exceeds {_MAX_PARAMETER_NAME_CHARS} characters; "
            "refusing unbounded parameter-name materialization"
        )
    if name in result:
        return
    if len(result) >= _MAX_PARAMETER_NAMES:
        _raise_runtime_abort(
            f"{label} exposed more than {_MAX_PARAMETER_NAMES} parameter names; "
            "refusing unbounded parameter-name materialization"
        )
    result.add(name)


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
        _validate_pipeline_run_bounds(width=width, height=height, frames=frames, fps=fps, steps=steps)
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
            _raise_runtime_abort(
                f"cannot scan {path} before loading {phase}; refusing to load without host allocation preflight"
            )
        total = _recursive_safetensor_size(path, phase)
        if total > 0:
            self._check_host_allocation(total * 2, phase)

    def _check_tokenizer_load(self, path: Path, phase: str) -> None:
        if not path.is_dir():
            _raise_runtime_abort(
                f"cannot scan {path} before loading {phase}; refusing to load tokenizer without host allocation preflight"
            )
        total = _flat_file_size_total(path, phase)
        if total > 0:
            self._check_host_allocation(total * 4, phase)

    def _expected_latent_shape(self) -> tuple[int, int, int, int, int]:
        channels = (
            _positive_structural_int(self.config.in_channels, "in_channels")
            if self.config is not None
            else 128
        )
        return (1, channels, self.frames, _latent_grid(self.height), _latent_grid(self.width))

    def _validate_latent_init_shape(self, *, width: int, height: int, frames: int) -> None:
        expected = (self.width, self.height, self.frames)
        actual = (width, height, frames)
        if actual != expected:
            raise RuntimeError(
                f"latent_init shape {actual} must match pipeline shape {expected}; "
                "create a fresh pipeline for a different run shape"
            )

    def _validate_latents_shape(self, latents: Any, phase: str) -> tuple[int, int, int, int, int]:
        expected = self._expected_latent_shape()
        actual = _bounded_shape_tuple(latents, expected_rank=len(expected), label=f"LTX2.3 latent {phase}")
        if actual != expected:
            raise RuntimeError(
                f"latent shape {actual} for {phase} does not match expected {expected}; "
                "refusing to allocate derived MLX tensors for an unexpected run shape"
            )
        return expected

    def _expected_context_shape(self) -> tuple[int, int]:
        hidden_size = 4096
        if self.config is not None:
            hidden_size = _positive_structural_int(
                getattr(self.config, "cross_attention_dim", getattr(self.config, "caption_channels", 4096)),
                "cross_attention_dim",
            )
        return (1, hidden_size)

    def _validate_context_shape(self, context: Any, phase: str) -> tuple[int, int]:
        expected = self._expected_context_shape()
        actual = _bounded_shape_tuple(context, expected_rank=len(expected), label=f"LTX2.3 context {phase}")
        if actual != expected:
            raise RuntimeError(
                f"LTX2.3 context shape {actual} for {phase} does not match expected {expected}; "
                "refusing to run transformer with unexpected text conditioning memory"
            )
        return expected

    def _validate_denoise_step_args(self, *, step_index: int, steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            _raise_runtime_abort(
                "LTX2.3 denoise step arguments are invalid: steps must be an integer, "
                f"got {_shape_dim_text(steps)}"
            )
        if steps <= 0:
            _raise_runtime_abort(
                f"LTX2.3 denoise step arguments are invalid: steps must be positive, got {steps}"
            )
        if steps > _MAX_DENOISE_STEPS:
            _raise_runtime_abort(
                f"LTX2.3 denoise step arguments are invalid: steps={steps} exceeds safe maximum "
                f"{_MAX_DENOISE_STEPS}"
            )
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            _raise_runtime_abort(
                "LTX2.3 denoise step arguments are invalid: step_index must be an integer, "
                f"got {_shape_dim_text(step_index)}"
            )
        if step_index < 0 or step_index >= steps:
            _raise_runtime_abort(
                f"LTX2.3 denoise step arguments are invalid: step_index={step_index} must be in [0, {steps})"
            )

    def _expected_frame_shape(self) -> tuple[int, int, int, int]:
        return (self.frames, self.height, self.width, 3)

    def _validate_frame_shape(self, frames: Any, phase: str) -> tuple[int, int, int, int]:
        expected = self._expected_frame_shape()
        actual = _bounded_shape_tuple(frames, expected_rank=len(expected), label=f"LTX2.3 frames {phase}")
        if actual != expected:
            raise RuntimeError(
                f"decoded LTX2.3 frames must have shape [T,H,W,3] {expected} for {phase}, got {actual}"
            )
        return expected

    def _preflight_config_shape(self, config_path: Path) -> dict[str, Any] | None:
        if not config_path.exists():
            raise FileNotFoundError(
                "LTX2.3 transformer config not found. Expected at:\n"
                f"  {config_path}\n"
                "Refusing to initialize MLX before local model config is present."
            )
        self._check_file_load(config_path, "preflight transformer config")
        config_dict = _read_bounded_json_config(config_path, "transformer config")
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
        max_positions = _positive_structural_int(
            text_config.get("max_position_embeddings", 1),
            "max_position_embeddings",
        )
        head_dim = _positive_structural_int(text_config.get("head_dim", hidden_size), "head_dim")
        kv_heads = _positive_structural_int(text_config.get("num_key_value_heads", heads), "num_key_value_heads")
        sliding_window = _positive_structural_int(text_config.get("sliding_window", max_positions), "sliding_window")
        sliding_window_pattern = _positive_structural_int(
            text_config.get("sliding_window_pattern", 1),
            "sliding_window_pattern",
        )
        for key, value in {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": layers,
            "vocab_size": vocab_size,
            "num_attention_heads": heads,
            "head_dim": head_dim,
            "num_key_value_heads": kv_heads,
            "sliding_window": sliding_window,
            "sliding_window_pattern": sliding_window_pattern,
            "max_position_embeddings": max_positions,
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
        shards = _flat_safetensor_names(text_encoder_dir, "preflight text_encoder weights")
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
        full_config = _read_bounded_json_config(config_path, "text_encoder config")
        text_config = full_config["text_config"]
        self._preflight_text_model_config(text_config, "text_encoder model config")
        from transformers import AutoTokenizer
        from fastgen_profiler.mlx_guard import check_token_sequence_budget

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"][0]
        max_tokens = _positive_structural_int(
            text_config["max_position_embeddings"],
            "max_position_embeddings",
        )
        hidden_size = _positive_structural_int(text_config["hidden_size"], "hidden_size")
        check_token_sequence_budget(
            token_count=len(input_ids),
            max_tokens=max_tokens,
            hidden_size=hidden_size,
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
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

            target = None
            gc.collect()
            _cleanup_loaded_runtime_after_error(exc)
            raise RuntimeMemoryAbort(
                "Runtime memory abort [ltx2.3 synchronize]: MLX synchronization failed; "
                "aborting because Metal runtime state may be unsafe."
            ) from exc

    def _eval_mlx(self, mx: Any, *targets: Any, phase: str) -> None:
        try:
            mx.eval(*targets)
        except Exception as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

            targets = ()
            gc.collect()
            _cleanup_loaded_runtime_after_error(exc)
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
        shards = _flat_safetensor_names(transformer_dir, "preflight transformer weights")
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
                config_dict = _read_bounded_json_config(config_path, "transformer config")
            self.config = LTXModelConfig(**config_dict)
            self.model = LTXModel(self.config)
            self._check_memory("model_construct after")

            model_param_names = _flatten_parameter_names(
                self.model.parameters(),
                label="LTX2.3 transformer parameters",
            )
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
                filtered_items = _filtered_weight_items(
                    weights.items(),
                    allowed_names=model_param_names,
                    excluded_suffixes=(".input_scale", ".weight_scale"),
                    label=f"LTX2.3 transformer weights {shard}",
                )
                if filtered_items.match_count > 0:
                    self._check_memory(f"model_load before {shard}")
                    self.model.load_weights(filtered_items, strict=False)
                    self._eval_mlx(mx, self.model.parameters(), phase=f"model_load {shard}")
                    loaded_keys.update(filtered_items.matched_keys)
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
            tp_model_params: set[str] = set()
            for k in self.text_proj.parameters():
                _add_parameter_name(
                    tp_model_params,
                    _parameter_key_text(k, label="LTX2.3 text projection parameters"),
                    label="LTX2.3 text projection parameters",
                )
            filtered_tp = _filtered_weight_items(
                tp_weights.items(),
                allowed_names=tp_model_params,
                label="LTX2.3 text projection weights",
            )
            if filtered_tp.match_count == 0:
                raise RuntimeError(
                    "LTX2.3 text projection weights did not match any text projection parameters; "
                    "refusing to continue with an uninitialized projection"
                )
            self.text_proj.load_weights(filtered_tp)

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
        full_config = _read_bounded_json_config(config_path, "text_encoder config")
        text_config = full_config["text_config"]
        self._preflight_text_model_config(text_config, "text_encoder model config")
        from transformers import AutoTokenizer
        from fastgen_profiler.mlx_guard import check_token_sequence_budget

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"][0]
        max_tokens = _positive_structural_int(
            text_config["max_position_embeddings"],
            "max_position_embeddings",
        )
        hidden_size = _positive_structural_int(text_config["hidden_size"], "hidden_size")
        check_token_sequence_budget(
            token_count=len(input_ids),
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            label="ltx2.3 text_encoder",
        )
        if not _dependency_available("mlx_lm"):
            raise ModuleNotFoundError(
                "mlx_lm is required for the LTX2.3 text encoder; dependency check "
                "failed before initializing MLX"
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
            text_model_params = _flatten_parameter_names(
                text_model.parameters(),
                label="LTX2.3 text encoder parameters",
            )
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
                mapped_items = _mapped_ltx_text_encoder_weight_items(
                    w.items(),
                    allowed_names=text_model_params,
                    label=f"LTX2.3 text encoder weights {shard}",
                )
                if mapped_items.match_count > 0:
                    self._check_memory(f"text_encoder before {shard}")
                    text_model.load_weights(mapped_items, strict=False)
                    self._eval_mlx(mx, text_model.parameters(), phase=f"text_encoder {shard}")
                    loaded_keys.update(mapped_items.matched_keys)
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
        self._validate_denoise_step_args(step_index=step_index, steps=steps)
        if self.context_emb is None:
            raise RuntimeError("denoise_step called before encode_text")
        phase = f"denoise {step_index + 1}/{steps}"
        latent_shape = self._validate_latents_shape(latents, phase)
        context_shape = self._validate_context_shape(self.context_emb, phase)
        if not _dependency_available("mlx_video"):
            raise ModuleNotFoundError(
                "mlx_video is required for LTX2.3 denoise; dependency check "
                "failed before initializing MLX"
            )

        self._check_memory(f"{phase} before")
        self._ensure_mlx_runtime_ready("denoise")

        try:
            import mlx.core as mx
            from mlx_video.models.ltx_2.transformer import Modality

            dtype = latents.dtype
            b, c, f, h, w = latent_shape
            num_tokens = f * h * w
            latent_elements = b * c * f * h * w
            position_elements = b * 3 * num_tokens * 2
            timestep_elements = b * num_tokens
            context_elements = math.prod(context_shape)
            denoise_floor_bytes = (
                latent_elements * 4 * 8
                + position_elements * 8 * 2
                + timestep_elements * 4 * 4
                + context_elements * 4 * 8
            )
            self._check_host_allocation(
                denoise_floor_bytes,
                f"{phase} tensors",
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
            self._eval_mlx(mx, velocity, phase=f"{phase} velocity")

            # Velocity → denoised (x0): x0 = latent - timestep * velocity
            sigma_f32 = mx.array(sigma, dtype=mx.float32)
            latents_flat_f32 = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
            timesteps_f32 = mx.expand_dims(timesteps.astype(mx.float32), axis=-1)
            x0_f32 = latents_flat_f32 - timesteps_f32 * velocity.astype(mx.float32)
            denoised = mx.reshape(mx.transpose(x0_f32, (0, 2, 1)), (b, c, f, h, w))

            self._eval_mlx(mx, denoised, phase=f"{phase} denoised")

            # Euler step
            if sigma_next > 0:
                sigma_next_f32 = mx.array(sigma_next, dtype=mx.float32)
                next_latents = denoised + sigma_next_f32 * (latents.astype(mx.float32) - denoised) / sigma_f32
            else:
                next_latents = denoised

            self._eval_mlx(mx, next_latents, phase=f"{phase} next_latents")
            next_latents = next_latents.astype(dtype)
            self._check_memory(f"{phase} after")
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
        vae_shards = _flat_safetensor_names(vae_decoder_dir, "preflight vae_decoder weights")
        if not vae_shards:
            raise FileNotFoundError(
                "LTX2.3 VAE decoder weights not found. Expected at least one .safetensors file in:\n"
                f"  {vae_decoder_dir}\n"
                "Refusing to initialize MLX VAE before local weights are present."
            )
        upscaler_path = self.model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        if upscaler_path.exists():
            self._check_file_load(upscaler_path, "preflight upsampler")
        if not _dependency_available("mlx_video"):
            raise ModuleNotFoundError(
                "mlx_video is required for LTX2.3 decode; dependency check "
                "failed before initializing MLX"
            )

        self._decode_started = True
        self._ensure_mlx_runtime_ready("decode")

        upsampler = None
        vae_temp = None
        vae = None
        video = None
        transposed = None
        frames = None
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
                self._check_tensor_shape_allocation(latents, "upsampled latent tensor", multiplier=8)
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
            self._check_host_allocation(math.prod(self._expected_frame_shape()) * 13, "numpy_frames")
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
            upsampler = vae_temp = vae = video = transposed = frames = latents = None
            gc.collect()
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def encode_video(self, frames: Any, *, fps: int) -> Any | Path:
        frame_shape = self._validate_frame_shape(frames, "video_encode")
        if self.dry_run or not self.save_video:
            return frames

        self._check_host_allocation(
            _frame_postprocess_budget_bytes(frames, frame_shape=frame_shape),
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
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            frames = None
            gc.collect()
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def write_output(self, video: Any | Path, output_dir: Path, *, run_id: str) -> Path:
        self._check_memory("file_write before")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{run_id}.mp4"
        if isinstance(video, Path):
            video.replace(output_path)
            return output_path

        frame_shape = self._validate_frame_shape(video, "file_write")
        self._check_host_allocation(
            _frame_postprocess_budget_bytes(video, frame_shape=frame_shape),
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

    def _check_tensor_shape_allocation(self, tensor: Any, phase: str, *, multiplier: int = 4) -> None:
        dimensions = _bounded_shape_tuple(
            tensor,
            expected_rank=len(self._expected_latent_shape()),
            label=f"LTX2.3 {phase}",
        )
        for dim in dimensions:
            if not isinstance(dim, int) or isinstance(dim, bool):
                _raise_runtime_abort(
                    f"LTX2.3 {phase} shape {dimensions!r} is not a finite integer shape; "
                    "refusing to continue without allocation preflight"
                )
        elements = math.prod(dimensions)
        if elements <= 0:
            _raise_runtime_abort(
                f"LTX2.3 {phase} shape {dimensions!r} has no positive elements; refusing to continue"
            )
        self._check_host_allocation(elements * 4 * multiplier, phase)


def create_ltx23_pipeline(**kwargs: Any) -> LTX23MLXPipeline:
    return LTX23MLXPipeline(**kwargs)


def _validate_pipeline_run_bounds(*, width: Any, height: Any, frames: Any, fps: Any, steps: Any) -> None:
    _validate_positive_capped_int(width, "width", MAX_RUN_DIMENSION)
    _validate_positive_capped_int(height, "height", MAX_RUN_DIMENSION)
    _validate_positive_capped_int(frames, "frames", MAX_RUN_FRAMES)
    _validate_positive_capped_int(fps, "fps", MAX_RUN_FPS)
    _validate_positive_capped_int(steps, "steps", MAX_RUN_STEPS)


def _validate_positive_capped_int(value: Any, name: str, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > max_value:
        raise ValueError(f"{name} must be no greater than {max_value}")


def _latent_grid(size: int) -> int:
    return max(1, (size + 7) // 8)


def _positive_structural_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"LTX2.3 config field {key}={_shape_dim_text(value)} must be a positive structural dimension; "
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


def _read_bounded_json_config(path: Path, label: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"cannot stat {path} before reading LTX2.3 {label}; refusing unbounded config load"
        ) from exc
    if size > _MAX_CONFIG_JSON_BYTES:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"LTX2.3 {label} at {path} is {size} bytes, above safe config limit "
            f"{_MAX_CONFIG_JSON_BYTES}; refusing unbounded config load"
        )
    config = json.loads(_read_bounded_text_file(path, label=label))
    if not isinstance(config, dict):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"LTX2.3 {label} at {path} must be a JSON object; refusing to construct MLX model"
        )
    _assert_bounded_json_structure(config, label=label)
    return config


def _read_bounded_text_file(path: Path, *, label: str) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_CONFIG_JSON_BYTES + 1)
    except OSError as exc:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"cannot read {path} before reading LTX2.3 {label}; refusing unbounded config load"
        ) from exc
    if len(data) > _MAX_CONFIG_JSON_BYTES:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"LTX2.3 {label} at {path} exceeded safe config limit "
            f"{_MAX_CONFIG_JSON_BYTES} during read; refusing unbounded config load"
        )
    return data.decode("utf-8")


def _assert_bounded_json_structure(value: Any, *, label: str) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited_items = 0
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_CONFIG_JSON_DEPTH:
            _raise_runtime_abort(
                f"LTX2.3 {label} JSON nesting exceeds safe depth {_MAX_CONFIG_JSON_DEPTH}; "
                "refusing unbounded config traversal"
            )
        if isinstance(current, dict):
            visited_items += len(current)
            if visited_items > _MAX_CONFIG_JSON_ITEMS:
                _raise_runtime_abort(
                    f"LTX2.3 {label} JSON structure exceeds safe item limit "
                    f"{_MAX_CONFIG_JSON_ITEMS}; refusing unbounded config traversal"
                )
            stack.extend((item, depth + 1) for item in current.values())
            continue
        if isinstance(current, list):
            visited_items += len(current)
            if visited_items > _MAX_CONFIG_JSON_ITEMS:
                _raise_runtime_abort(
                    f"LTX2.3 {label} JSON structure exceeds safe item limit "
                    f"{_MAX_CONFIG_JSON_ITEMS}; refusing unbounded config traversal"
                )
            stack.extend((item, depth + 1) for item in current)


def _recursive_safetensor_size(path: Path, phase: str) -> int:
    total = 0
    visited_dirs = 0
    scanned_files = 0
    stack = [path]
    while stack:
        current = stack.pop()
        visited_dirs += 1
        if visited_dirs > _MAX_PRELOAD_SCAN_DIRS:
            _raise_runtime_abort(
                f"cannot scan {path} before loading {phase}; directory scan exceeded "
                f"{_MAX_PRELOAD_SCAN_DIRS} directories"
            )
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        _raise_runtime_abort(
                            f"cannot scan {entry.path} before loading {phase}; "
                            "refusing to load without host allocation preflight"
                        )
                    scanned_files += 1
                    if scanned_files > _MAX_PRELOAD_SCAN_FILES:
                        _raise_runtime_abort(
                            f"cannot scan {path} before loading {phase}; file scan exceeded "
                            f"{_MAX_PRELOAD_SCAN_FILES} files"
                        )
                    if entry.name.endswith(".safetensors"):
                        try:
                            total += entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            _raise_runtime_abort(
                                f"cannot stat {entry.path} before loading {phase}; "
                                "refusing to load without host allocation preflight"
                            )
        except OSError:
            _raise_runtime_abort(
                f"cannot scan {current} before loading {phase}; refusing to load without host allocation preflight"
            )
    return total


def _flat_file_size_total(path: Path, phase: str) -> int:
    total = 0
    scanned = 0
    try:
        for entry in path.iterdir():
            scanned += 1
            if scanned > _MAX_PRELOAD_SCAN_FILES:
                _raise_runtime_abort(
                    f"cannot scan {path} before loading {phase}; file scan exceeded "
                    f"{_MAX_PRELOAD_SCAN_FILES} files"
                )
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        _raise_runtime_abort(
            f"cannot scan {path} before loading {phase}; refusing to load without host allocation preflight"
        )
    return total


def _flat_safetensor_names(path: Path, phase: str) -> list[str]:
    names: list[str] = []
    scanned = 0
    try:
        for entry in path.iterdir():
            scanned += 1
            if scanned > _MAX_PRELOAD_SCAN_FILES:
                _raise_runtime_abort(
                    f"cannot scan {path} before loading {phase}; file scan exceeded "
                    f"{_MAX_PRELOAD_SCAN_FILES} files"
                )
            if entry.is_file() and entry.name.endswith(".safetensors"):
                names.append(entry.name)
    except OSError:
        _raise_runtime_abort(
            f"cannot scan {path} before loading {phase}; refusing to load without host allocation preflight"
        )
    return sorted(names)


def _raise_runtime_abort(message: str) -> None:
    from fastgen_profiler.mlx_guard import RuntimeMemoryAbort, mlx_cleanup

    try:
        mlx_cleanup()
    except Exception:
        pass

    raise RuntimeMemoryAbort(message)


def _numpy() -> Any:
    import numpy as np

    return np


def _bounded_shape_tuple(value: Any, *, expected_rank: int, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        _raise_runtime_abort(f"{label} has no shape; refusing unbounded shape inspection")
    dims: list[int] = []
    try:
        iterator = iter(shape)
    except TypeError:
        _raise_runtime_abort(f"{label} shape is not iterable; refusing unbounded shape inspection")
    for dim in iterator:
        if len(dims) >= expected_rank:
            _raise_runtime_abort(
                f"{label} shape rank exceeds {expected_rank}; refusing unbounded shape inspection"
            )
        if not isinstance(dim, int) or isinstance(dim, bool):
            _raise_runtime_abort(
                f"{label} shape contains non-integer dimension {_shape_dim_text(dim)}; "
                "refusing unbounded shape inspection"
            )
        dims.append(dim)
    if len(dims) != expected_rank:
        _raise_runtime_abort(
            f"{label} shape rank is {len(dims)}, expected {expected_rank}; "
            "refusing under-bounded shape inspection"
        )
    return tuple(dims)


def _shape_dim_text(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return str(value)
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


class _FilteredWeightItems:
    def __init__(
        self,
        iterator: Any,
        *,
        allowed_names: set[str],
        excluded_suffixes: tuple[str, ...],
        label: str,
        first_match: tuple[Any, Any] | None,
        scanned: int,
    ) -> None:
        self._iterator = iterator
        self._allowed_names = allowed_names
        self._excluded_suffixes = excluded_suffixes
        self._label = label
        self._first_match = first_match
        self._scanned = scanned
        self.match_count = 1 if first_match is not None else 0
        self.matched_keys: set[str] = set()

    def __iter__(self):
        if self._first_match is not None:
            key, value = self._first_match
            self.matched_keys.add(key)
            self._first_match = None
            yield key, value
        for scanned, item in enumerate(self._iterator, start=self._scanned + 1):
            if scanned > _MAX_FILTERED_WEIGHT_ITEMS:
                _raise_runtime_abort(
                    f"{self._label} scan exceeded {_MAX_FILTERED_WEIGHT_ITEMS} items; "
                    "refusing unbounded weight filtering"
                )
            try:
                key, value = item
            except (TypeError, ValueError):
                _raise_runtime_abort(f"{self._label} contained malformed weight item")
            key = _weight_key_text(key, label=self._label)
            if key in self._allowed_names and not key.endswith(self._excluded_suffixes):
                self.match_count += 1
                self.matched_keys.add(key)
                yield key, value


def _filtered_weight_items(
    items: Any,
    *,
    allowed_names: set[str],
    label: str,
    excluded_suffixes: tuple[str, ...] = (),
) -> _FilteredWeightItems:
    iterator = iter(items)
    for scanned, item in enumerate(iterator, start=1):
        if scanned > _MAX_FILTERED_WEIGHT_ITEMS:
            _raise_runtime_abort(
                f"{label} scan exceeded {_MAX_FILTERED_WEIGHT_ITEMS} items; refusing unbounded weight filtering"
            )
        try:
            key, value = item
        except (TypeError, ValueError):
            _raise_runtime_abort(f"{label} contained malformed weight item")
        key = _weight_key_text(key, label=label)
        if key in allowed_names and not key.endswith(excluded_suffixes):
            return _FilteredWeightItems(
                iterator,
                allowed_names=allowed_names,
                excluded_suffixes=excluded_suffixes,
                label=label,
                first_match=(key, value),
                scanned=scanned,
            )
    return _FilteredWeightItems(
        iter(()),
        allowed_names=allowed_names,
        excluded_suffixes=excluded_suffixes,
        label=label,
        first_match=None,
        scanned=0,
    )


class _MappedLTXTextEncoderWeightItems:
    def __init__(
        self,
        iterator: Any,
        *,
        allowed_names: set[str],
        label: str,
        first_match: tuple[str, Any] | None,
        scanned: int,
    ) -> None:
        self._iterator = iterator
        self._allowed_names = allowed_names
        self._label = label
        self._first_match = first_match
        self._scanned = scanned
        self.match_count = 1 if first_match is not None else 0
        self.matched_keys: set[str] = set()

    def __iter__(self):
        if self._first_match is not None:
            key, value = self._first_match
            self.matched_keys.add(key)
            self._first_match = None
            yield key, value
        for scanned, item in enumerate(self._iterator, start=self._scanned + 1):
            if scanned > _MAX_FILTERED_WEIGHT_ITEMS:
                _raise_runtime_abort(
                    f"{self._label} scan exceeded {_MAX_FILTERED_WEIGHT_ITEMS} items; "
                    "refusing unbounded weight filtering"
                )
            try:
                raw_key, value = item
            except (TypeError, ValueError):
                _raise_runtime_abort(f"{self._label} contained malformed weight item")
            key = _map_ltx_text_encoder_weight_key(raw_key)
            if key is not None and key in self._allowed_names:
                self.match_count += 1
                self.matched_keys.add(key)
                yield key, value


def _mapped_ltx_text_encoder_weight_items(
    items: Any,
    *,
    allowed_names: set[str],
    label: str,
) -> _MappedLTXTextEncoderWeightItems:
    iterator = iter(items)
    for scanned, item in enumerate(iterator, start=1):
        if scanned > _MAX_FILTERED_WEIGHT_ITEMS:
            _raise_runtime_abort(
                f"{label} scan exceeded {_MAX_FILTERED_WEIGHT_ITEMS} items; refusing unbounded weight filtering"
            )
        try:
            raw_key, value = item
        except (TypeError, ValueError):
            _raise_runtime_abort(f"{label} contained malformed weight item")
        key = _map_ltx_text_encoder_weight_key(raw_key)
        if key is not None and key in allowed_names:
            return _MappedLTXTextEncoderWeightItems(
                iterator,
                allowed_names=allowed_names,
                label=label,
                first_match=(key, value),
                scanned=scanned,
            )
    return _MappedLTXTextEncoderWeightItems(
        iter(()),
        allowed_names=allowed_names,
        label=label,
        first_match=None,
        scanned=0,
    )


def _map_ltx_text_encoder_weight_key(key: Any) -> str | None:
    key = _weight_key_text(key, label="LTX2.3 text encoder weights")
    if key.startswith("language_model.model."):
        return key[len("language_model.model."):]
    if key.startswith("model."):
        return key[len("model."):]
    return None


def _weight_key_text(key: Any, *, label: str) -> str:
    if not isinstance(key, str):
        key_type = type(key)
        _raise_runtime_abort(
            f"{label} key <{key_type.__module__}.{key_type.__qualname__}> is not a string; "
            "refusing unbounded weight-key materialization"
        )
    if len(key) > _MAX_PARAMETER_NAME_CHARS:
        _raise_runtime_abort(
            f"{label} key exceeds {_MAX_PARAMETER_NAME_CHARS} characters; "
            "refusing unbounded weight-key materialization"
        )
    return key


def _frame_postprocess_budget_bytes(frames: Any, *, frame_shape: tuple[int, int, int, int] | None = None) -> int:
    if frame_shape is None:
        frame_shape = _bounded_shape_tuple(frames, expected_rank=4, label="LTX2.3 postprocess frames")
    shape_floor = math.prod(frame_shape) * 4
    reported_nbytes = getattr(frames, "nbytes", 0)
    if not isinstance(reported_nbytes, int) or isinstance(reported_nbytes, bool) or reported_nbytes < 0:
        reported_nbytes = 0
    return max(reported_nbytes, shape_floor) * _VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER


def _normalize_video_frames(np: Any, frames: Any) -> Any:
    """Normalize VAE float frames to uint8 with minimal host-side copies."""
    np.add(frames, 1.0, out=frames)
    np.multiply(frames, 127.5, out=frames)
    np.clip(frames, 0, 255, out=frames)
    return frames.astype(np.uint8, copy=False)


def _cleanup_loaded_runtime_after_error(exc: BaseException | None = None) -> None:
    if exc is not None:
        _clear_traceback_frames(exc)
    try:
        from fastgen_profiler.mlx_guard import mlx_cleanup

        mlx_cleanup()
    except Exception:
        pass


def _clear_traceback_frames(exc: BaseException) -> None:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        tb = current.__traceback__
        while tb is not None:
            try:
                tb.tb_frame.clear()
            except RuntimeError:
                pass
            tb = tb.tb_next
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)


def _iter_config_numbers(value: Any, prefix: str = ""):
    stack: list[tuple[Any, str, int]] = [(value, prefix, 0)]
    visited_items = 0
    while stack:
        current, current_prefix, depth = stack.pop()
        if depth > _MAX_CONFIG_JSON_DEPTH:
            _raise_runtime_abort(
                f"LTX2.3 config numeric scan exceeds safe depth {_MAX_CONFIG_JSON_DEPTH}; "
                "refusing unbounded config traversal"
            )
        if isinstance(current, dict):
            visited_items += len(current)
            if visited_items > _MAX_CONFIG_JSON_ITEMS:
                _raise_runtime_abort(
                    f"LTX2.3 config numeric scan exceeds safe item limit {_MAX_CONFIG_JSON_ITEMS}; "
                    "refusing unbounded config traversal"
                )
            for key, child in current.items():
                if not isinstance(key, str):
                    key_type = type(key)
                    _raise_runtime_abort(
                        f"LTX2.3 config key <{key_type.__module__}.{key_type.__qualname__}> "
                        "is not a string; refusing unbounded config traversal"
                    )
                child_prefix = f"{current_prefix}.{key}" if current_prefix else key
                stack.append((child, child_prefix, depth + 1))
            continue
        if isinstance(current, (list, tuple)):
            visited_items += len(current)
            if visited_items > _MAX_CONFIG_JSON_ITEMS:
                _raise_runtime_abort(
                    f"LTX2.3 config numeric scan exceeds safe item limit {_MAX_CONFIG_JSON_ITEMS}; "
                    "refusing unbounded config traversal"
                )
            for index, child in enumerate(current):
                child_prefix = f"{current_prefix}[{index}]"
                stack.append((child, child_prefix, depth + 1))
            continue
        if isinstance(current, int) and not isinstance(current, bool):
            yield current_prefix, current


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
