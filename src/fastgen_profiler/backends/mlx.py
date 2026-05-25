"""Scaffolded MLX backend for future model integrations."""

from __future__ import annotations

from time import perf_counter

from fastgen_profiler.metrics import MeasurementRecord, REQUIRED_PHASES, RunConfig

from .base import BackendAdapter, timed_section


class MLXBackend(BackendAdapter):
    name = "mlx"

    def run(
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
