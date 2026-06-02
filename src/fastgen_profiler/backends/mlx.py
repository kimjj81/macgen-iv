"""MLX backend — scaffold with runtime memory watchdog.

When a real MLX pipeline adapter is loaded (e.g. wan22_mlx_adapter),
the denoise loop checks memory each step via check_runtime_memory().
The stub scaffold also calls it so the guard path is exercised in tests.
"""

from __future__ import annotations

import logging
from time import perf_counter

from fastgen_profiler.metrics import MeasurementRecord, REQUIRED_PHASES, RunConfig

from .base import BackendAdapter, timed_section

logger = logging.getLogger("fastgen_profiler.mlx_backend")


class MLXBackend(BackendAdapter):
    name = "mlx"
    scaffold_only = True

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
                    # Runtime memory watchdog — check every step.
                    # If memory is critically low, raise RuntimeMemoryAbort
                    # instead of letting the system hit a watchdog timeout.
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
