from __future__ import annotations

from fastgen_profiler.backends import create_backend
from fastgen_profiler.metrics import BenchmarkConfig
from fastgen_profiler.profiler import Profiler


def test_profiler_result_schema_contains_required_config_and_total_phase():
    config = BenchmarkConfig(
        model="ltx2.3",
        backend="ltx2.3",
        prompt="schema test",
        seed=1,
        width=512,
        height=288,
        frames=8,
        steps=2,
        precision="fp16",
        guidance=3.5,
        cache_enabled=False,
        compile_enabled=False,
    )

    result = Profiler(create_backend("ltx2.3", dry_run=True)).run(config)
    record = result.to_dict()

    assert record["config"]["model"] == "ltx2.3"
    assert record["config"]["cache_enabled"] is False
    assert any(phase["name"] == "total" for phase in record["phase_timings"])
    assert record["notes"]
