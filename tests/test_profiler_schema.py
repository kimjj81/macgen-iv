from __future__ import annotations

from pathlib import Path

from fastgen_profiler.backends import create_backend
from fastgen_profiler.metrics import REQUIRED_PHASES, RunConfig
from fastgen_profiler.profiler import Profiler


REQUIRED_FIELDS = {
    "run_id",
    "timestamp_utc",
    "model",
    "backend",
    "model_path",
    "model_id",
    "model_source_root",
    "prompt_hash",
    "negative_prompt_hash",
    "seed",
    "width",
    "height",
    "frames",
    "fps",
    "steps",
    "guidance",
    "quant",
    "cache",
    "compile",
    "phase",
    "step_index",
    "seconds",
    "peak_memory",
    "active_memory",
    "cache_memory",
    "output_path",
    "error",
    "machine",
}


def test_profiler_records_required_fields_and_phases():
    config = RunConfig(
        model="ltx2.3",
        backend="stub",
        model_path=None,
        model_id=None,
        model_source_root=None,
        prompt="schema test",
        negative_prompt="",
        seed=1,
        width=512,
        height=288,
        frames=8,
        fps=8,
        steps=2,
        guidance=3.5,
        quant="none",
        cache="none",
        compile="off",
        output_dir=Path("unused"),
        result_jsonl=Path("unused.jsonl"),
        save_video=False,
        dry_run=True,
    )

    records = Profiler(create_backend("stub")).run(config)
    serialized = [record.to_dict() for record in records]

    assert serialized
    assert all(REQUIRED_FIELDS == set(record) for record in serialized)
    assert {record["phase"] for record in serialized} == set(REQUIRED_PHASES)
    assert len([record for record in serialized if record["phase"] == "denoise_step"]) == 2
    assert all(record["model"] == "ltx2.3" for record in serialized)
    assert all(record["prompt_hash"] != "schema test" for record in serialized)
    assert all("python_version" in record["machine"] for record in serialized)
