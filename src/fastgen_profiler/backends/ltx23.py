"""LTX2.3 backend adapter placeholder."""

from __future__ import annotations

from fastgen_profiler.metrics import BenchmarkConfig, BenchmarkResult

from .base import BackendAdapter


class LTX23Backend(BackendAdapter):
    name = "ltx2.3"

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        if self.dry_run:
            return self.placeholder_result(config)
        raise NotImplementedError(
            "LTX2.3 MLX pipeline integration is not implemented yet. "
            "Use --dry-run to validate profiler output shape."
        )
