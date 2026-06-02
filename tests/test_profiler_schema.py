from __future__ import annotations

import builtins
from pathlib import Path
import sys

import pytest

from fastgen_profiler.backends import create_backend
from fastgen_profiler.metrics import (
    MAX_METRIC_COLLECTION_ITEMS,
    MAX_METRIC_RECORD_JSON_BYTES,
    MAX_METRIC_TEXT_FIELD_CHARS,
    REQUIRED_PHASES,
    RunConfig,
    append_jsonl,
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("width", 4097, "width must be no greater than 4096"),
        ("height", 4097, "height must be no greater than 4096"),
        ("frames", 258, "frames must be no greater than 257"),
        ("fps", 241, "fps must be no greater than 240"),
        ("steps", 513, "steps must be no greater than 512"),
    ],
)
def test_profiler_rejects_direct_run_config_outside_safe_bounds(field, value, message):
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
    setattr(config, field, value)

    with pytest.raises(ValueError, match=message):
        Profiler(create_backend("stub")).run(config)


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


def test_machine_metadata_sysctl_does_not_capture_unbounded_output(monkeypatch):
    import subprocess

    import fastgen_profiler.metrics as metrics

    calls: list[dict[str, object]] = []

    def fake_run(args, **kwargs):
        kwargs["stdout"].write(b"Apple M4 Pro\n" if args[-1] == "machdep.cpu.brand_string" else b"123456\n")
        calls.append(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(metrics.sys, "platform", "darwin")
    monkeypatch.setattr(metrics.subprocess, "run", fake_run)

    metadata = metrics.machine_metadata()

    assert metadata["chip"] == "Apple M4 Pro"
    assert metadata["total_memory"] == 123456
    assert calls
    assert all(call["stderr"] is subprocess.DEVNULL for call in calls)
    assert all("capture_output" not in call for call in calls)


def test_machine_metadata_sysctl_treats_oversized_output_as_unknown(monkeypatch):
    import subprocess

    import fastgen_profiler.metrics as metrics

    def fake_run(args, **kwargs):
        kwargs["stdout"].write(b"x" * (metrics.MAX_MACHINE_METADATA_OUTPUT_BYTES + 1))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(metrics.sys, "platform", "darwin")
    monkeypatch.setattr(metrics.subprocess, "run", fake_run)

    assert metrics._sysctl_value("machdep.cpu.brand_string") is None


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


def test_append_jsonl_rejects_more_records_than_limit(tmp_path):
    config = RunConfig(
        model="ltx2.3",
        backend="stub",
        model_path=None,
        model_id=None,
        model_source_root=None,
        prompt="write limit",
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
        output_dir=tmp_path,
        result_jsonl=tmp_path / "records.jsonl",
        save_video=False,
        dry_run=True,
    )

    def records():
        while True:
            yield make_record(
                config,
                run_id="run",
                timestamp_utc="2026-01-01T00:00:00Z",
                machine={},
                phase="total",
                seconds=0.0,
            )

    with pytest.raises(ValueError, match="exceeds JSONL write record limit"):
        append_jsonl(config.result_jsonl, records(), max_records=1)

    assert len(config.result_jsonl.read_text(encoding="utf-8").splitlines()) == 1


def test_append_jsonl_rejects_record_larger_than_byte_limit(tmp_path):
    config = RunConfig(
        model="ltx2.3",
        backend="stub",
        model_path=None,
        model_id=None,
        model_source_root=None,
        prompt="write byte limit",
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
        output_dir=tmp_path,
        result_jsonl=tmp_path / "oversized-record.jsonl",
        save_video=False,
        dry_run=True,
    )
    machine = {
        f"field_{index}": "x" * MAX_METRIC_TEXT_FIELD_CHARS
        for index in range(MAX_METRIC_COLLECTION_ITEMS)
    }
    record = make_record(
        config,
        run_id="run",
        timestamp_utc="2026-01-01T00:00:00Z",
        machine=machine,
        phase="total",
        seconds=0.0,
    )

    with pytest.raises(ValueError, match=f"JSONL record exceeds write byte limit: .* > {MAX_METRIC_RECORD_JSON_BYTES}"):
        append_jsonl(config.result_jsonl, [record])

    assert not config.result_jsonl.exists() or config.result_jsonl.read_text(encoding="utf-8") == ""


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

    assert "DeepcopyBlocked" in record["machine"]["object"]
    assert len(record["machine"]["object"]) <= MAX_METRIC_TEXT_FIELD_CHARS


def test_measurement_record_summarizes_unknown_metric_values_without_repr_or_str():
    class UnsafeValue:
        def __repr__(self):
            raise AssertionError("metric serialization must not call repr on unknown values")

        def __str__(self):
            raise AssertionError("metric serialization must not call str on unknown values")

    class UnsafeKey:
        def __repr__(self):
            raise AssertionError("metric serialization must not call repr on unknown keys")

        def __str__(self):
            raise AssertionError("metric serialization must not call str on unknown keys")

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
        machine={
            "value": UnsafeValue(),
            UnsafeKey(): "keyed",
        },
        phase="total",
        seconds=0.0,
    ).to_dict()

    assert "UnsafeValue" in record["machine"]["value"]
    assert any("UnsafeKey" in key for key in record["machine"])


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


def test_markdown_report_summarizes_unknown_values_without_repr_or_str():
    class UnsafeValue:
        def __repr__(self):
            raise AssertionError("markdown report must not call repr on unknown values")

        def __str__(self):
            raise AssertionError("markdown report must not call str on unknown values")

    records = [
        {
            "run_id": UnsafeValue(),
            "preset": "manual",
            "variant_label": "manual",
            "model": "wan2.2",
            "backend": "stub",
            "phase": UnsafeValue(),
            "seconds": 0.0,
            "peak_memory": None,
            "error": UnsafeValue(),
        }
    ]

    report = render_markdown_report(records)

    assert "UnsafeValue" in report


def test_markdown_report_does_not_coerce_unknown_numeric_values():
    class UnsafeNumeric:
        def __float__(self):
            raise AssertionError("markdown report must not call float on unknown values")

        def __int__(self):
            raise AssertionError("markdown report must not call int on unknown values")

        def __repr__(self):
            raise AssertionError("markdown report must not call repr on unknown values")

        def __str__(self):
            raise AssertionError("markdown report must not call str on unknown values")

    records = [
        {
            "run_id": "unsafe-numeric",
            "preset": "manual",
            "variant_label": "manual",
            "model": "wan2.2",
            "backend": "stub",
            "phase": "total",
            "seconds": UnsafeNumeric(),
            "peak_memory": UnsafeNumeric(),
            "error": None,
        }
    ]

    report = render_markdown_report(records)

    assert "0.000000" in report
    assert "unavailable" in report
    assert "UnsafeNumeric" not in report


def test_markdown_report_ignores_oversized_numeric_strings():
    huge_numeric = "1" * 10_000
    records = [
        {
            "run_id": "oversized-numeric",
            "preset": "manual",
            "variant_label": "manual",
            "model": "wan2.2",
            "backend": "stub",
            "phase": "total",
            "seconds": huge_numeric,
            "peak_memory": huge_numeric,
            "error": None,
        }
    ]

    report = render_markdown_report(records)

    assert "0.000000" in report
    assert "unavailable" in report
    assert huge_numeric not in report


def test_markdown_report_rejects_too_many_runs_before_rendering_sections():
    from fastgen_profiler.reports.markdown import MAX_REPORT_RUNS

    records = [
        {
            "run_id": f"run-{index}",
            "preset": "manual",
            "variant_label": "manual",
            "model": "wan2.2",
            "backend": "stub",
            "phase": "total",
            "seconds": 0.0,
            "peak_memory": None,
            "error": None,
        }
        for index in range(MAX_REPORT_RUNS + 1)
    ]

    with pytest.raises(ValueError, match="report run limit"):
        render_markdown_report(records)
