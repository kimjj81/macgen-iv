"""MLX backend — delegates to model-specific adapters (e.g. wan22_mlx_adapter).

Falls back to a scaffold that reports "not implemented" when no adapter matches.
When an adapter is loaded, the denoise loop checks memory each step via
check_runtime_memory() to prevent watchdog timeout panics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from fastgen_profiler.metrics import MeasurementRecord, REQUIRED_PHASES, RunConfig

from .base import BackendAdapter, timed_section

logger = logging.getLogger("fastgen_profiler.mlx_backend")


class MLXBackend(BackendAdapter):
    name = "mlx"
    scaffold_only = True  # set False by _resolve_adapter when a real adapter matches

    # -- adapter resolution ------------------------------------------------

    def _resolve_adapter(self, config: RunConfig):
        """Return (module_path, class_name, kwargs) or None."""
        model = (config.model or "").lower()
        if "wan" in model and ("2.2" in model or "22" in model):
            return (
                "fastgen_profiler.backends.wan22_mlx_adapter",
                "Wan22MLXPipeline",
                dict(
                    model_path=Path(config.model_path) if config.model_path else Path("."),
                    seed=config.seed,
                    width=config.width,
                    height=config.height,
                    frames=config.frames,
                    steps=config.steps,
                    fps=config.fps,
                    guidance=config.guidance,
                    quant=config.quant or "none",
                    cache=config.cache or "none",
                    compile=config.compile or "off",
                    save_video=config.save_video,
                    dry_run=config.dry_run,
                ),
            )
        return None

    # -- main run ----------------------------------------------------------

    def run(
        self,
        config: RunConfig,
        *,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
    ) -> list[MeasurementRecord]:
        adapter_info = self._resolve_adapter(config)
        if adapter_info is not None:
            self.scaffold_only = False  # real adapter available
        if adapter_info is None:
            return self._run_scaffold(config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine)
        return self._run_with_adapter(config, adapter_info, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine)

    # -- scaffold fallback -------------------------------------------------

    def _run_scaffold(
        self,
        config: RunConfig,
        *,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
    ) -> list[MeasurementRecord]:
        records: list[MeasurementRecord] = []
        total_started = perf_counter()
        model_location = f" at {config.model_path}" if config.model_path else ""
        error = (
            f"{config.model} MLX pipeline integration{model_location} is not implemented yet; "
            "use --backend stub for profiler verification without model weights."
        )

        for phase in REQUIRED_PHASES:
            if phase == "denoise_step":
                for step_index in range(config.steps):
                    try:
                        from fastgen_profiler.mlx_guard import check_runtime_memory
                        check_runtime_memory(
                            label=f"{config.model} step {step_index}/{config.steps}"
                        )
                    except ImportError as exc:
                        raise RuntimeError(
                            "mlx_guard unavailable before MLX runtime watchdog; "
                            "refusing to continue without memory checks"
                        ) from exc

                    records.append(
                        self.record(
                            config,
                            run_id=run_id,
                            timestamp_utc=timestamp_utc,
                            machine=machine,
                            phase=phase,
                            step_index=step_index,
                            seconds=0.0,
                            error=error,
                        )
                    )
                continue

            with timed_section() as timing:
                pass
            seconds = perf_counter() - total_started if phase == "total" else timing["seconds"]
            records.append(
                self.record(
                    config,
                    run_id=run_id,
                    timestamp_utc=timestamp_utc,
                    machine=machine,
                    phase=phase,
                    seconds=seconds,
                    error=error,
                )
            )
        return records

    # -- real adapter execution --------------------------------------------

    def _run_with_adapter(
        self,
        config: RunConfig,
        adapter_info: tuple,
        *,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
    ) -> list[MeasurementRecord]:
        # Fail fast if memory guard is unavailable — safety critical.
        try:
            from fastgen_profiler.mlx_guard import check_runtime_memory  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "mlx_guard unavailable before MLX runtime watchdog; "
                "refusing to continue without memory checks"
            ) from exc

        import importlib

        module_path, class_name, kwargs = adapter_info
        logger.info("Loading adapter %s.%s", module_path, class_name)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        pipeline = cls(**kwargs)

        records: list[MeasurementRecord] = []
        total_started = perf_counter()

        # Phase: model_load
        with timed_section() as timing:
            pipeline.load_model()
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="model_load", seconds=timing["seconds"],
            )
        )

        # Phase: prompt_prepare
        with timed_section() as timing:
            prepared = pipeline.prepare_prompt(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
            )
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="prompt_prepare", seconds=timing["seconds"],
            )
        )

        # Phase: text_encoder
        with timed_section() as timing:
            context = pipeline.encode_text(prepared)
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="text_encoder", seconds=timing["seconds"],
            )
        )

        # Phase: latent_init
        with timed_section() as timing:
            latents = pipeline.init_latents(
                seed=config.seed,
                width=config.width,
                height=config.height,
                frames=config.frames,
            )
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="latent_init", seconds=timing["seconds"],
            )
        )

        # Phase: denoise_total + denoise_step (N steps)
        denoise_started = perf_counter()
        for step_index in range(config.steps):
            with timed_section() as timing:
                latents = pipeline.denoise_step(
                    latents,
                    step_index=step_index,
                    steps=config.steps,
                    guidance=config.guidance,
                    cache=config.cache or "none",
                )
            records.append(
                self.record(
                    config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                    phase="denoise_step", step_index=step_index, seconds=timing["seconds"],
                )
            )
        denoise_total_seconds = perf_counter() - denoise_started
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="denoise_total", seconds=denoise_total_seconds,
            )
        )

        # Phase: vae_decode
        with timed_section() as timing:
            frames = pipeline.decode(latents)
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="vae_decode", seconds=timing["seconds"],
            )
        )

        # Phase: video_encode
        with timed_section() as timing:
            video = pipeline.encode_video(frames, fps=config.fps)
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="video_encode", seconds=timing["seconds"],
            )
        )

        # Phase: file_write
        with timed_section() as timing:
            output_path = pipeline.write_output(
                video, Path(config.output_dir), run_id=run_id,
            )
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="file_write", seconds=timing["seconds"],
                output_path=str(output_path),
            )
        )

        # Phase: total
        total_seconds = perf_counter() - total_started
        records.append(
            self.record(
                config, run_id=run_id, timestamp_utc=timestamp_utc, machine=machine,
                phase="total", seconds=total_seconds,
            )
        )

        return records
