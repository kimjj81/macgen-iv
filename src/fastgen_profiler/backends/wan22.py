"""Wan2.2 backend adapter placeholder."""

from __future__ import annotations

from fastgen_profiler.metrics import BenchmarkConfig, BenchmarkResult

from .base import BackendAdapter


class Wan22Backend(BackendAdapter):
    name = "wan2.2"

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        if self.dry_run:
            return self.placeholder_result(config)
        raise NotImplementedError(
            "Wan2.2 MLX pipeline integration is not implemented yet. "
            "Use --dry-run to validate profiler output shape."
        )
