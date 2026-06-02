from __future__ import annotations

import builtins
from pathlib import Path
import sys

from fastgen_profiler.backends import create_backend
from fastgen_profiler.metrics import (
    MAX_METRIC_COLLECTION_ITEMS,
    MAX_METRIC_TEXT_FIELD_CHARS,
    REQUIRED_PHASES,
    RunConfig,
    machine_metadata,
    make_record,
    read_jsonl,
)
from fastgen_profiler.profiler import Profiler
from fastgen_profiler.reports.markdown import render_markdown_report


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
    "profile_id",
    "profile_name",
    "preset",
    "variant_label",
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


def test_machine_metadata_does_not_import_mlx(monkeypatch):
    sys.modules.pop("mlx", None)
    sys.modules.pop("mlx.core", None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"mlx", "mlx.core"}:
            raise AssertionError("machine metadata must not initialize MLX")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    metadata = machine_metadata()

    assert "python_version" in metadata
    assert "mlx" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_read_jsonl_rejects_file_larger_than_limit(tmp_path):
    path = tmp_path / "oversized.jsonl"
    path.write_text('{"phase":"total"}\n{"phase":"denoise_step"}\n', encoding="utf-8")

    try:
        read_jsonl(path, max_bytes=8)
    except ValueError as exc:
        assert "exceeds JSONL read limit" in str(exc)
    else:
        raise AssertionError("read_jsonl should reject oversized input files")


def test_read_jsonl_rejects_more_records_than_limit(tmp_path):
    path = tmp_path / "too-many-records.jsonl"
    path.write_text('{"phase":"total"}\n{"phase":"denoise_step"}\n', encoding="utf-8")

    try:
        read_jsonl(path, max_records=1)
    except ValueError as exc:
        assert "exceeds JSONL record limit" in str(exc)
    else:
        raise AssertionError("read_jsonl should reject excessive record counts")


def test_measurement_record_bounds_serialized_text_fields():
    config = RunConfig(
        model="wan2.2",
        backend="stub",
        model_path="x" * (MAX_METRIC_TEXT_FIELD_CHARS * 2),
        model_id=None,
        model_source_root=None,
        prompt="prompt",
        negative_prompt="",
        seed=1,
        width=256,
        height=256,
        frames=4,
        fps=4,
        steps=1,
        guidance=1.0,
        quant="none",
        cache="none",
        compile="off",
        output_dir=Path("unused"),
        result_jsonl=Path("unused.jsonl"),
        save_video=False,
        dry_run=True,
    )
    long_text = "error:" + ("z" * (MAX_METRIC_TEXT_FIELD_CHARS * 2))
    long_key = "key:" + ("k" * (MAX_METRIC_TEXT_FIELD_CHARS * 2))
    long_list = list(range(MAX_METRIC_COLLECTION_ITEMS + 20))
    long_dict = {f"item-{index}": index for index in range(MAX_METRIC_COLLECTION_ITEMS + 20)}

    record = make_record(
        config,
        run_id="run",
        timestamp_utc="2026-01-01T00:00:00Z",
        machine={
            "nested": {
                "detail": long_text,
                long_key: "long key",
                "long_list": long_list,
                "long_dict": long_dict,
            }
        },
        phase="total",
        seconds=0.0,
        error=long_text,
    ).to_dict()

    assert "<truncated>" in record["error"]
    assert "<truncated>" in record["model_path"]
    assert "<truncated>" in record["machine"]["nested"]["detail"]
    assert len(record["error"]) <= MAX_METRIC_TEXT_FIELD_CHARS
    assert any("<truncated>" in key for key in record["machine"]["nested"])
    assert len(record["machine"]["nested"]["long_list"]) == MAX_METRIC_COLLECTION_ITEMS + 1
    assert record["machine"]["nested"]["long_list"][-1] == {"__truncated_items__": True}
    assert len(record["machine"]["nested"]["long_dict"]) == MAX_METRIC_COLLECTION_ITEMS + 1
    assert record["machine"]["nested"]["long_dict"]["__truncated_items__"] is True


def test_measurement_record_bounds_before_deepcopying_metadata():
    class DeepcopyBlocked:
        def __deepcopy__(self, memo):
            raise AssertionError("metric serialization must not deep-copy metadata before bounding")

        def __repr__(self):
            return "DeepcopyBlocked(" + ("x" * (MAX_METRIC_TEXT_FIELD_CHARS * 2)) + ")"

    config = RunConfig(
        model="wan2.2",
        backend="stub",
        model_path=None,
        model_id=None,
        model_source_root=None,
        prompt="prompt",
        negative_prompt="",
        seed=1,
        width=256,
        height=256,
        frames=4,
        fps=4,
        steps=1,
        guidance=1.0,
        quant="none",
        cache="none",
        compile="off",
        output_dir=Path("unused"),
        result_jsonl=Path("unused.jsonl"),
        save_video=False,
        dry_run=True,
    )

    record = make_record(
        config,
        run_id="run",
        timestamp_utc="2026-01-01T00:00:00Z",
        machine={"object": DeepcopyBlocked()},
        phase="total",
        seconds=0.0,
    ).to_dict()

    assert "<truncated>" in record["machine"]["object"]
    assert len(record["machine"]["object"]) <= MAX_METRIC_TEXT_FIELD_CHARS


def test_markdown_report_bounds_text_and_ignores_non_finite_metrics():
    long_error = "failed: " + ("x" * 1_000)
    records = [
        {
            "run_id": "run|bad\nid",
            "preset": "manual",
            "variant_label": "variant|" + ("y" * 1_000),
            "model": "wan2.2",
            "backend": "stub",
            "phase": "total",
            "seconds": float("inf"),
            "peak_memory": -1,
            "error": long_error,
        },
        {
            "run_id": "run|bad\nid",
            "preset": "manual",
            "variant_label": "variant|" + ("y" * 1_000),
            "model": "wan2.2",
            "backend": "stub",
            "phase": "denoise_total",
            "seconds": float("nan"),
            "peak_memory": "not-an-int",
            "error": long_error,
        },
        {
            "run_id": "run|bad\nid",
            "preset": "manual",
            "variant_label": "variant|" + ("y" * 1_000),
            "model": "wan2.2",
            "backend": "stub",
            "phase": "decode",
            "seconds": -5,
            "peak_memory": 1024,
            "error": long_error,
        },
    ]

    report = render_markdown_report(records)

    assert "inf" not in report.lower()
    assert "nan" not in report.lower()
    assert "run/bad id" in report
    assert "variant/" in report
    assert "<truncated>" in report
    assert "1024 bytes" in report
    assert "x" * 300 not in report
