"""Common backend adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from fastgen_profiler.metrics import BenchmarkConfig, BenchmarkResult, PhaseTiming


class BackendAdapter(ABC):
    name: str

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    @abstractmethod
    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run one benchmark and return structured metrics."""

    def placeholder_result(self, config: BenchmarkConfig) -> BenchmarkResult:
        started = perf_counter()
        duration = perf_counter() - started
        return BenchmarkResult(
            config=config,
            phase_timings=[
                PhaseTiming(
                    name="total",
                    duration_seconds=duration,
                    synchronization="dry_run_no_mlx_work",
                    notes="Dry run placeholder; no model was loaded.",
                )
            ],
            notes=[
                "Dry run only.",
                "Replace this adapter path with real MLX model loading and phase timers.",
            ],
        )
