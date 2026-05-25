"""Profiler orchestration."""

from __future__ import annotations

from time import perf_counter

from .backends.base import BackendAdapter
from .metrics import BenchmarkConfig, BenchmarkResult, PhaseTiming


class Profiler:
    """Coordinates benchmark execution through a backend adapter."""

    def __init__(self, backend: BackendAdapter) -> None:
        self.backend = backend

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        started = perf_counter()
        result = self.backend.run(config)
        total_duration = perf_counter() - started

        if not any(phase.name == "total" for phase in result.phase_timings):
            result.phase_timings.append(
                PhaseTiming(
                    name="total",
                    duration_seconds=total_duration,
                    synchronization="profiler_wall_time",
                )
            )

        return result
