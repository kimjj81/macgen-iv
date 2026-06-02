"""Local Wan2.2 compatibility adapter for installed mlx_video APIs.

The installed mlx_video package used by this repo exposes lower-level Wan2.2
helpers, but not the macgen-profile create_wan22_pipeline(...) contract.  This
module adapts those helpers to the profiler backend boundary without making the
profiler core depend on mlx_video internals.
"""

from __future__ import annotations

from dataclasses import fields
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any


_VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER = 6
_MAX_CONFIG_JSON_BYTES = 1 * 1024 * 1024
_MAX_PRELOAD_SCAN_FILES = 10_000
_MAX_CONFIG_DIMENSION = 65_536
_MAX_CONFIG_AREA = 4096 * 4096


def _require_non_empty_parameters(component: Any, label: str) -> Any:
    try:
        parameters = component.parameters()
    except Exception as exc:
        raise RuntimeError(
            f"Wan2.2 {label} parameters could not be inspected; refusing to continue "
            "because weight loading cannot be verified"
        ) from exc
    if isinstance(parameters, dict) and not parameters:
        raise RuntimeError(
            f"Wan2.2 {label} exposed no parameters; refusing to continue with an "
            "uninitialized model component"
        )
    if parameters is None:
        raise RuntimeError(
            f"Wan2.2 {label} exposed no parameters; refusing to continue with an "
            "uninitialized model component"
        )
    return parameters


class Wan22MLXPipeline:
    """Pipeline object implementing the macgen-profile Wan2.2 phase contract."""

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
        self._requested_shape = (width, height, frames)

        self.mx: Any | None = None
        self.config: Any | None = None
        self.quantization: dict[str, Any] | None = None
        self.tokenizer: Any | None = None
        self.t5_encoder: Any | None = None
        self.model: Any | None = None
        self.scheduler: Any | None = None
        self.latent_shape: tuple[int, int, int, int] | None = None
        self.seq_len: int | None = None
        self.context_cond: Any | None = None
        self.context_cfg: Any | None = None
        self.cross_kv: Any | None = None
        self.rope_cos_sin: Any | None = None
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

        check_memory_guard(label=f"wan2.2 {phase}")
        check_run_allocation_budget(
            width=self.width,
            height=self.height,
            frames=self.frames,
            guidance=self.guidance,
            label=f"wan2.2 {phase}",
        )
        configure_mlx_resource_limits(label=f"wan2.2 {phase}")
        self._mlx_runtime_ready = True

    def _check_run_budget(self, phase: str) -> None:
        from fastgen_profiler.mlx_guard import check_run_allocation_budget

        check_run_allocation_budget(
            width=self.width,
            height=self.height,
            frames=self.frames,
            guidance=self.guidance,
            label=f"wan2.2 {phase}",
        )

    def _check_memory(self, phase: str) -> None:
        try:
            from fastgen_profiler.mlx_guard import check_runtime_memory
        except ImportError as exc:
            raise RuntimeError(
                f"memory guard unavailable before wan2.2 {phase}; refusing to continue without runtime memory checks"
            ) from exc
        check_runtime_memory(label=f"wan2.2 {phase}")

    def _check_host_allocation(self, required_bytes: int, phase: str) -> None:
        try:
            from fastgen_profiler.mlx_guard import check_host_allocation_headroom
        except ImportError as exc:
            raise RuntimeError(
                f"memory guard unavailable before wan2.2 {phase}; refusing to continue without host allocation checks"
            ) from exc
        check_host_allocation_headroom(required_bytes, label=f"wan2.2 {phase}")

    def _check_file_load(self, path: Path, phase: str) -> None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
            raise RuntimeMemoryAbort(
                f"cannot stat {path} before loading {phase}; refusing to load without host allocation preflight"
            ) from exc
        self._check_host_allocation(size * 2, phase)

    def _check_tokenizer_load(self, path: Path, phase: str) -> None:
        if not path.is_dir():
            _raise_runtime_abort(
                f"cannot scan {path} before loading {phase}; refusing to load tokenizer without host allocation preflight"
            )
        total = _flat_file_size_total(path, phase)
        if total > 0:
            self._check_host_allocation(total * 4, phase)

    def _check_mlx_tensor_floor(self, elements: int, phase: str, *, multiplier: int = 4) -> None:
        from fastgen_profiler.mlx_guard import MemoryGuardError

        for name, value in (("elements", elements), ("multiplier", multiplier)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MemoryGuardError(
                    f"Memory guard [wan2.2 {phase}]: {name} must be a positive integer, got {value!r}"
                )
        self._check_host_allocation(elements * 4 * multiplier, phase)

    def _expected_frame_shape(self) -> tuple[int, int, int, int]:
        return (self.frames, self.height, self.width, 3)

    def _validate_frame_shape(self, frames: Any, phase: str) -> None:
        actual = tuple(getattr(frames, "shape", ()))
        expected = self._expected_frame_shape()
        if actual != expected:
            raise RuntimeError(
                f"decoded Wan2.2 frames must have shape [T,H,W,3] {expected} for {phase}, got {actual}"
            )

    def _validate_latent_init_shape(self, *, width: int, height: int, frames: int) -> None:
        actual = (width, height, frames)
        expected_shapes = {(self.width, self.height, self.frames), self._requested_shape}
        if actual not in expected_shapes:
            expected = " or ".join(str(shape) for shape in sorted(expected_shapes))
            raise RuntimeError(
                f"latent_init shape {actual} does not match Wan2.2 pipeline shape {expected}"
            )

    def _validate_latents_shape(self, latents: Any, phase: str) -> None:
        if self.latent_shape is None:
            raise RuntimeError(f"{phase} called before latent shape was initialized")
        actual = tuple(getattr(latents, "shape", ()))
        expected = self.latent_shape
        if actual != expected:
            raise RuntimeError(
                f"Wan2.2 latent shape {actual} does not match expected {expected} for {phase}"
            )

    def _validate_loaded_config(self) -> None:
        if self.config is None:
            raise RuntimeError("Wan2.2 config was not initialized")
        _validate_wan_shape_config(self.config)
        self._preflight_model_config(self.config, "loaded model config tensor")
        if self.config.dual_model:
            raise RuntimeError(
                "local Wan2.2 compatibility adapter currently supports single-model converted directories only; "
                "dual-model directories require high_noise_model.safetensors and low_noise_model.safetensors adapter support"
            )
        if self.quant not in {"none", "fp16", "bf16"}:
            raise RuntimeError(f"unsupported Wan2.2 MLX quant setting for local adapter: {self.quant}")
        if self.cache not in {"none", "kv"}:
            raise RuntimeError(f"unsupported Wan2.2 MLX cache setting for local adapter: {self.cache}")
        if self.compile != "off":
            raise RuntimeError("mx.compile is disabled for baseline MLX benchmarking; rerun with --compile off")

    def _preflight_model_config(self, config: Any, phase: str) -> None:
        hidden_size = _positive_int(getattr(config, "dim", getattr(config, "hidden_size", 4096)), "dim")
        ffn_size = _positive_int(getattr(config, "ffn_dim", hidden_size * 4), "ffn_dim")
        layers = _positive_int(getattr(config, "num_layers", getattr(config, "num_hidden_layers", 1)), "num_layers")
        heads = _positive_int(getattr(config, "num_heads", getattr(config, "num_attention_heads", 1)), "num_heads")
        text_dim = _positive_int(getattr(config, "text_dim", hidden_size), "text_dim")
        for key, value in {
            "dim": hidden_size,
            "ffn_dim": ffn_size,
            "num_layers": layers,
            "num_heads": heads,
            "text_dim": text_dim,
        }.items():
            if value <= 0:
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"Wan2.2 config field {key}={value} must be a positive structural dimension; "
                    "refusing to construct MLX model"
                )
            if value > _MAX_CONFIG_DIMENSION:
                from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

                raise RuntimeMemoryAbort(
                    f"Wan2.2 config field {key}={value} exceeds safe structural dimension "
                    f"{_MAX_CONFIG_DIMENSION}; refusing to construct MLX model"
                )
        attention_floor = layers * hidden_size * hidden_size * 4
        mlp_floor = layers * hidden_size * ffn_size * 3
        text_floor = hidden_size * text_dim * 2
        head_floor = layers * heads * hidden_size
        self._check_host_allocation((attention_floor + mlp_floor + text_floor + head_floor) * 2, phase)

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
                "Runtime memory abort [wan2.2 synchronize]: MLX synchronization failed; "
                "aborting because Metal runtime state may be unsafe."
            ) from exc

    def _eval_mlx(self, mx: Any, *targets: Any, phase: str) -> None:
        try:
            mx.eval(*targets)
        except Exception as exc:
            from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

            _cleanup_loaded_runtime_after_error()
            raise RuntimeMemoryAbort(
                f"Runtime memory abort [wan2.2 {phase}]: MLX eval failed; "
                "aborting because Metal runtime state may be unsafe."
            ) from exc

    def load_model(self) -> dict[str, object]:
        if self.model is not None or self.t5_encoder is not None:
            raise RuntimeError(
                "Wan2.2 MLX model is already loaded in this pipeline; create a fresh "
                "pipeline/process before loading again to avoid accumulating Metal state."
            )
        t5_path = self.model_path / "t5_encoder.safetensors"
        model_path = self.model_path / "model.safetensors"
        tokenizer_path = self.model_path / "tokenizer"
        self._check_file_load(t5_path, "preflight t5_encoder")
        self._check_file_load(model_path, "preflight model")
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                "Wan2.2 tokenizer not found. Expected a local tokenizer at:\n"
                f"  {tokenizer_path}/\n"
                "Automatic downloads are disabled for memory-safe local profiling."
            )
        self._check_tokenizer_load(tokenizer_path, "preflight tokenizer")
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "Wan2.2 config not found. Expected a local config at:\n"
                f"  {config_path}\n"
                "Refusing to initialize MLX before model structure can be preflighted."
            )
        self._check_file_load(config_path, "preflight config")
        if importlib.util.find_spec("mlx_video") is None:
            raise ModuleNotFoundError(
                "mlx_video is required for the Wan2.2 adapter; dependency check "
                "failed before initializing MLX"
            )

        raw_config = _load_raw_config_for_preflight(config_path)
        raw_width, raw_height = _aligned_size(self.width, self.height, raw_config)
        original_width, original_height = self.width, self.height
        self.width, self.height = raw_width, raw_height
        self._check_run_budget("aligned config preflight")
        raw_latent_shape, _ = _latent_shape_and_seq_len(
            raw_width,
            raw_height,
            self.frames,
            raw_config,
        )
        self._check_host_allocation(
            math.prod(raw_latent_shape) * 4 * 4,
            "config latent tensor",
        )
        self._preflight_model_config(raw_config, "config model tensor")
        self._ensure_mlx_runtime_ready("load_model")

        try:
            self.config, self.quantization = _load_config(self.model_path)
            self.width, self.height = _aligned_size(self.width, self.height, self.config)
            self.latent_shape, self.seq_len = _latent_shape_and_seq_len(
                self.width,
                self.height,
                self.frames,
                self.config,
            )

            self._validate_loaded_config()
            self._check_run_budget("aligned load_model")
            self._requested_shape = (original_width, original_height, self.frames)

            import mlx.core as mx  # type: ignore[import-not-found]
            from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler
            from mlx_video.models.wan_2.utils import load_t5_encoder, load_wan_model
            from transformers import AutoTokenizer

            self.mx = mx
            if self.config is None or self.latent_shape is None or self.seq_len is None:
                self.config, self.quantization = _load_config(self.model_path)
                self.width, self.height = _aligned_size(self.width, self.height, self.config)
                self.latent_shape, self.seq_len = _latent_shape_and_seq_len(
                    self.width,
                    self.height,
                    self.frames,
                    self.config,
                )
                self._validate_loaded_config()
                self._check_run_budget("aligned load_model")
            self._check_file_load(t5_path, "read t5_encoder")
            self._check_memory("t5_load before")
            self.t5_encoder = load_t5_encoder(t5_path, self.config)
            self._eval_mlx(mx, _require_non_empty_parameters(self.t5_encoder, "t5_encoder"), phase="t5_load")
            self._check_memory("t5_load after")
            self._check_tokenizer_load(tokenizer_path, "read tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
            self._check_file_load(model_path, "read model")
            self._check_memory("model_load before")
            self.model = load_wan_model(
                model_path,
                self.config,
                self.quantization,
            )
            self._eval_mlx(mx, _require_non_empty_parameters(self.model, "transformer"), phase="model_load")
            self._check_memory("model_load after")
            self.scheduler = FlowUniPCScheduler(num_train_timesteps=self.config.num_train_timesteps)
            self.scheduler.set_timesteps(self.steps, shift=self.config.sample_shift)
            mx.random.seed(self.seed)
            np = _numpy()
            np.random.seed(self.seed)
            return {"model_type": self.config.model_type, "width": self.width, "height": self.height}
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def prepare_prompt(self, *, prompt: str, negative_prompt: str | None) -> dict[str, str]:
        if self.config is None:
            raise RuntimeError("prepare_prompt called before load_model")
        resolved_negative = negative_prompt
        if resolved_negative is None:
            resolved_negative = self.config.sample_neg_prompt
        from fastgen_profiler.mlx_guard import check_text_prompt_budget
        check_text_prompt_budget(
            prompt=prompt,
            negative_prompt=resolved_negative,
            label="wan2.2 prompt",
        )
        return {"prompt": prompt, "negative_prompt": resolved_negative}

    def encode_text(self, prepared_prompt: dict[str, str]) -> Any:
        if self.mx is None or self.config is None or self.model is None or self.t5_encoder is None or self.tokenizer is None:
            raise RuntimeError("encode_text called before load_model")
        if self.seq_len is None:
            raise RuntimeError("latent sequence length was not initialized")
        if self._text_encode_started:
            raise RuntimeError(
                "Wan2.2 text encoding has already started in this pipeline; create a fresh "
                "pipeline/process before encoding again to avoid accumulating Metal state."
            )

        from fastgen_profiler.mlx_guard import check_text_prompt_budget

        check_text_prompt_budget(
            prompt=prepared_prompt["prompt"],
            negative_prompt=prepared_prompt["negative_prompt"],
            label="wan2.2 direct text",
        )
        self._check_prompt_token_budget(prepared_prompt["prompt"], "text_encoder prompt")
        if not self.cfg_disabled:
            self._check_prompt_token_budget(prepared_prompt["negative_prompt"], "text_encoder negative_prompt")

        text_dim = _positive_int(getattr(self.config, "dim", 4096), "dim")
        text_len = _positive_int(self.config.text_len, "text_len")
        cfg_factor = 1 if self.cfg_disabled else 2
        self._check_mlx_tensor_floor(
            text_len * text_dim * cfg_factor,
            "text_encoder context tensors",
            multiplier=6,
        )
        f_grid = self.latent_shape[1] // self.config.patch_size[0]  # type: ignore[index]
        h_grid = self.latent_shape[2] // self.config.patch_size[1]  # type: ignore[index]
        w_grid = self.latent_shape[3] // self.config.patch_size[2]  # type: ignore[index]
        self._check_mlx_tensor_floor(
            f_grid * h_grid * w_grid * text_dim * cfg_factor,
            "text_encoder rope tensors",
            multiplier=2,
        )
        self._text_encode_started = True
        try:
            from mlx_video.models.wan_2.utils import encode_text

            self._check_memory("text_encoder before")
            context = encode_text(
                self.t5_encoder,
                self.tokenizer,
                prepared_prompt["prompt"],
                self.config.text_len,
            )
            if self.cfg_disabled:
                context_null = None
                self._eval_mlx(self.mx, context, phase="text_encoder context")
                context_emb = self.model.embed_text([context])
                self._eval_mlx(self.mx, context_emb, phase="text_encoder embedding")
                self.context_cond = context_emb[0:1]
                self.cross_kv = self.model.prepare_cross_kv(self.context_cond)
            else:
                context_null = encode_text(
                    self.t5_encoder,
                    self.tokenizer,
                    prepared_prompt["negative_prompt"],
                    self.config.text_len,
                )
                self._eval_mlx(self.mx, context, context_null, phase="text_encoder cfg context")
                context_emb = self.model.embed_text([context, context_null])
                self._eval_mlx(self.mx, context_emb, phase="text_encoder cfg embedding")
                self.context_cfg = self.mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)
                self.cross_kv = self.model.prepare_cross_kv(self.context_cfg)
            self._eval_mlx(self.mx, self.cross_kv, phase="text_encoder cross_kv")

            rope_grid_sizes = [(f_grid, h_grid, w_grid)] if self.cfg_disabled else [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
            self.rope_cos_sin = self.model.prepare_rope(rope_grid_sizes)
            self._eval_mlx(self.mx, self.rope_cos_sin, phase="text_encoder rope")
            self._check_memory("text_encoder after")

            del context
            if context_null is not None:
                del context_null
            del self.t5_encoder
            self.t5_encoder = None
            gc.collect()
            self.mx.clear_cache()
            return self.context_cond if self.cfg_disabled else self.context_cfg
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def _check_prompt_token_budget(self, prompt: str, phase: str) -> None:
        if self.config is None or self.tokenizer is None:
            raise RuntimeError("token budget check called before load_model")
        from fastgen_profiler.mlx_guard import check_token_sequence_budget

        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"][0]
        hidden_size = _positive_int(
            getattr(
                self.config,
                "text_dim",
                getattr(self.config, "dim", getattr(self.config, "hidden_size", 4096)),
            ),
            "text_dim",
        )
        max_tokens = _positive_int(self.config.text_len, "text_len")
        check_token_sequence_budget(
            token_count=len(input_ids),
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            label=f"wan2.2 {phase}",
        )

    def init_latents(self, *, seed: int, width: int, height: int, frames: int) -> Any:
        if self.mx is None or self.latent_shape is None:
            raise RuntimeError("init_latents called before load_model")
        self._validate_latent_init_shape(width=width, height=height, frames=frames)
        try:
            self._check_memory("latent_init before")
            self.mx.random.seed(seed)
            self._check_mlx_tensor_floor(math.prod(self.latent_shape), "latent_init tensor")
            latents = self.mx.random.normal(self.latent_shape)
            self._eval_mlx(self.mx, latents, phase="latent_init")
            self._check_memory("latent_init after")
            return latents
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def denoise_step(self, latents: Any, *, step_index: int, steps: int, guidance: float, cache: str) -> Any:
        if self.mx is None or self.model is None or self.scheduler is None:
            raise RuntimeError("denoise_step called before load_model")
        if self.seq_len is None or self.cross_kv is None or self.rope_cos_sin is None:
            raise RuntimeError("denoise_step called before encode_text")

        try:
            self._validate_latents_shape(latents, f"denoise {step_index + 1}/{steps}")
            self._check_memory(f"denoise {step_index + 1}/{steps} before")
            timestep_val = self.scheduler.timesteps.tolist()[step_index]
            latent_shape = getattr(latents, "shape", None)
            if latent_shape is not None:
                cfg_factor = 1 if self.cfg_disabled else 2
                self._check_mlx_tensor_floor(
                    math.prod(latent_shape) * cfg_factor,
                    f"denoise {step_index + 1}/{steps} tensors",
                    multiplier=8,
                )
            if self.cfg_disabled:
                t_batch = self.mx.array([timestep_val])
                preds = self.model(
                    [latents],
                    t=t_batch,
                    context=self.context_cond,
                    seq_len=self.seq_len,
                    cross_kv_caches=self.cross_kv,
                    rope_cos_sin=self.rope_cos_sin,
                )
                noise_pred = preds[0]
                del preds
            else:
                t_batch = self.mx.array([timestep_val, timestep_val])
                preds = self.model(
                    [latents, latents],
                    t=t_batch,
                    context=self.context_cfg,
                    seq_len=self.seq_len,
                    cross_kv_caches=self.cross_kv,
                    rope_cos_sin=self.rope_cos_sin,
                )
                noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
                noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
                del noise_pred_cond, noise_pred_uncond, preds

            next_latents = self.scheduler.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
            del noise_pred
            self._eval_mlx(self.mx, next_latents, phase=f"denoise {step_index + 1}/{steps} next_latents")
            self._check_memory(f"denoise {step_index + 1}/{steps} after")
            return next_latents
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise

    def decode(self, latents: Any) -> Any:
        if self.mx is None or self.config is None:
            raise RuntimeError("decode called before load_model")
        if self._decode_started:
            raise RuntimeError(
                "Wan2.2 decode has already started in this pipeline; create a fresh "
                "pipeline/process before decoding again to avoid accumulating Metal state."
            )
        self._validate_latents_shape(latents, "decode")
        vae_path = self.model_path / "vae.safetensors"
        self._check_file_load(vae_path, "read vae")
        self._decode_started = True
        self._ensure_mlx_runtime_ready("decode")

        try:
            from mlx_video.models.wan_2.utils import load_vae_decoder
            from mlx_video.models.wan_2.vae22 import denormalize_latents

            self._check_memory("vae_decode before")
            self._check_memory("vae_load before")
            vae = load_vae_decoder(vae_path, self.config)
            self._eval_mlx(self.mx, vae.parameters(), phase="vae parameters")
            self._check_memory("vae_load after")
            if self.config.vae_z_dim == 48:
                z = latents.transpose(1, 2, 3, 0)[None]
                z = denormalize_latents(z)
                self._check_host_allocation(self.frames * self.height * self.width * 3 * 4 * 4, "vae output tensor")
                video = vae(z)
                self._eval_mlx(self.mx, video, phase="vae video")
                self._check_memory("vae_forward after")
                self._validate_frame_shape(video[0], "decode")
                self._check_host_allocation(math.prod(video[0].shape) * 13, "numpy_frames")
                np = _numpy()
                frames = np.array(video[0])
                del z
            else:
                self._check_host_allocation(self.frames * self.height * self.width * 3 * 4 * 4, "vae output tensor")
                video = vae.decode(latents[None])
                self._eval_mlx(self.mx, video, phase="vae video")
                self._check_memory("vae_forward after")
                video_slice = video[0]
                self._validate_frame_shape(video_slice, "decode")
                self._check_host_allocation(math.prod(video_slice.shape) * 13, "numpy_frames")
                np = _numpy()
                frames = np.array(video_slice).transpose(1, 2, 3, 0)
                del video_slice
            del vae, video
            gc.collect()
            self.mx.clear_cache()
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
            raise RuntimeError(f"decoded Wan2.2 frames must have shape [T,H,W,3], got {frames.shape}")
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
                "mlx_video is required for Wan2.2 video postprocess; dependency check "
                "failed before initializing MLX"
            )
        self._ensure_mlx_runtime_ready("video_encode")
        temp_path: Path | None = None
        try:
            from mlx_video.models.wan_2.postprocess import save_video

            handle = tempfile.NamedTemporaryFile(prefix="macgen-wan22-", suffix=".mp4", delete=False)
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

        fps = self.config.sample_fps if self.config is not None else self.fps
        self._validate_frame_shape(video, "file_write")
        self._check_host_allocation(
            video.nbytes * _VIDEO_POSTPROCESS_ALLOCATION_MULTIPLIER,
            "file_write frames",
        )
        if importlib.util.find_spec("mlx_video") is None:
            raise ModuleNotFoundError(
                "mlx_video is required for Wan2.2 video postprocess; dependency check "
                "failed before initializing MLX"
            )
        self._ensure_mlx_runtime_ready("file_write")
        try:
            from mlx_video.models.wan_2.postprocess import save_video

            save_video(video, str(output_path), fps=fps)
            self._check_memory("file_write after")
            return output_path
        except Exception as exc:
            _cleanup_loaded_runtime_after_error(exc)
            raise


def create_wan22_pipeline(**kwargs: Any) -> Wan22MLXPipeline:
    return Wan22MLXPipeline(**kwargs)


def _load_config(model_path: Path) -> tuple[Any, dict[str, Any] | None]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Wan2.2 config not found. Expected a local config at:\n"
            f"  {config_path}\n"
            "Refusing to load mlx_video config defaults before local model structure is known."
        )

    config_dict = _read_bounded_json_config(config_path, "model config")
    quantization = config_dict.pop("quantization", None)
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in config_dict and isinstance(config_dict[key], list):
            config_dict[key] = tuple(config_dict[key])
    from mlx_video.models.wan_2.config import WanModelConfig

    valid_names = {field.name for field in fields(WanModelConfig)}
    config = WanModelConfig(**{key: value for key, value in config_dict.items() if key in valid_names})
    if config.in_dim == 48 and config.vae_z_dim != 48:
        config = WanModelConfig(
            **{
                **{field.name: getattr(config, field.name) for field in fields(WanModelConfig)},
                "vae_z_dim": 48,
                "vae_stride": (4, 16, 16),
                "sample_fps": 24,
            }
        )
    return config, quantization


def _load_raw_config_for_preflight(config_path: Path) -> Any:
    config_dict = _read_bounded_json_config(config_path, "preflight config")
    config = SimpleNamespace(
        patch_size=_positive_int_tuple(
            config_dict.get("patch_size", (1, 2, 2)),
            "patch_size",
            length=3,
            max_value=_MAX_CONFIG_DIMENSION,
        ),
        vae_stride=_positive_int_tuple(
            config_dict.get("vae_stride", (4, 16, 16)),
            "vae_stride",
            length=3,
            max_value=_MAX_CONFIG_DIMENSION,
        ),
        max_area=_non_negative_int(
            config_dict.get("max_area", 0),
            "max_area",
            max_value=_MAX_CONFIG_AREA,
        ),
        vae_z_dim=_positive_int(
            config_dict.get("vae_z_dim", config_dict.get("in_dim", 48)),
            "vae_z_dim",
            max_value=_MAX_CONFIG_DIMENSION,
        ),
        dim=_positive_int(config_dict.get("dim", config_dict.get("hidden_size", 4096)), "dim"),
        ffn_dim=_positive_int(config_dict.get("ffn_dim", 16_384), "ffn_dim"),
        num_layers=_positive_int(config_dict.get("num_layers", config_dict.get("num_hidden_layers", 1)), "num_layers"),
        num_heads=_positive_int(config_dict.get("num_heads", config_dict.get("num_attention_heads", 1)), "num_heads"),
        text_dim=_positive_int(config_dict.get("text_dim", config_dict.get("dim", 4096)), "text_dim"),
    )
    return config


def _aligned_size(width: int, height: int, config: Any) -> tuple[int, int]:
    _validate_wan_shape_config(config)
    align_h = config.patch_size[1] * config.vae_stride[1]
    align_w = config.patch_size[2] * config.vae_stride[2]
    if height % align_h != 0:
        height = max(align_h, (height // align_h) * align_h)
    if width % align_w != 0:
        width = max(align_w, (width // align_w) * align_w)
    if config.max_area > 0 and height * width > config.max_area:
        ratio = width / height
        width = int((config.max_area * ratio) ** 0.5) // align_w * align_w
        height = int(config.max_area / max(width, align_w)) // align_h * align_h
        width = max(width, align_w)
        height = max(height, align_h)
    return width, height


def _latent_shape_and_seq_len(width: int, height: int, frames: int, config: Any) -> tuple[tuple[int, int, int, int], int]:
    _validate_wan_shape_config(config)
    z_dim = config.vae_z_dim
    t_latent = (frames - 1) // config.vae_stride[0] + 1
    h_latent = height // config.vae_stride[1]
    w_latent = width // config.vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)
    seq_len = math.ceil((h_latent * w_latent) / (config.patch_size[1] * config.patch_size[2]) * t_latent)
    return target_shape, seq_len


def _positive_int(value: Any, key: str, *, max_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key}={value!r} must be a positive integer; "
            "refusing to construct MLX model"
        )
    result = value
    if result <= 0:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key}={result} must be a positive integer; "
            "refusing to construct MLX model"
        )
    if max_value is not None and result > max_value:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key}={result} exceeds safe structural dimension "
            f"{max_value}; refusing to construct MLX model"
        )
    return result


def _read_bounded_json_config(path: Path, label: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"cannot stat {path} before reading Wan2.2 {label}; refusing unbounded config load"
        ) from exc
    if size > _MAX_CONFIG_JSON_BYTES:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 {label} at {path} is {size} bytes, above safe config limit "
            f"{_MAX_CONFIG_JSON_BYTES}; refusing unbounded config load"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 {label} at {path} must be a JSON object; refusing to construct MLX model"
        )
    return config


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


def _raise_runtime_abort(message: str) -> None:
    from fastgen_profiler.mlx_guard import RuntimeMemoryAbort, mlx_cleanup

    try:
        mlx_cleanup()
    except Exception:
        pass

    raise RuntimeMemoryAbort(message)


def _non_negative_int(value: Any, key: str, *, max_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key}={value!r} must be zero or a positive integer; "
            "refusing to construct MLX model"
        )
    result = value
    if result < 0:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key}={result} must be zero or a positive integer; "
            "refusing to construct MLX model"
        )
    if max_value is not None and result > max_value:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key}={result} exceeds safe structural dimension "
            f"{max_value}; refusing to construct MLX model"
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


def _positive_int_tuple(
    value: Any,
    key: str,
    *,
    length: int,
    max_value: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

        raise RuntimeMemoryAbort(
            f"Wan2.2 config field {key} must be a {length}-item positive integer tuple; "
            "refusing to construct MLX model"
        )
    return tuple(
        _positive_int(item, f"{key}[{index}]", max_value=max_value)
        for index, item in enumerate(value)
    )


def _validate_wan_shape_config(config: Any) -> None:
    for key, value in {
        "patch_size[0]": config.patch_size[0],
        "patch_size[1]": config.patch_size[1],
        "patch_size[2]": config.patch_size[2],
        "vae_stride[0]": config.vae_stride[0],
        "vae_stride[1]": config.vae_stride[1],
        "vae_stride[2]": config.vae_stride[2],
        "vae_z_dim": config.vae_z_dim,
    }.items():
        _positive_int(value, key, max_value=_MAX_CONFIG_DIMENSION)
    _non_negative_int(config.max_area, "max_area", max_value=_MAX_CONFIG_AREA)
