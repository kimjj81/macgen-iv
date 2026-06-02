from __future__ import annotations

import builtins
import json
import sys
from unittest.mock import patch

import pytest

import fastgen_profiler.cli as cli_module
from fastgen_profiler.cli import main
from fastgen_profiler.metrics import RunConfig


@pytest.fixture(autouse=True)
def _reset_mlx_run_counter():
    from fastgen_profiler.mlx_guard import reset_run_counter

    reset_run_counter()
    yield
    reset_run_counter()


def test_cli_run_parses_required_arguments_and_stub_writes_jsonl(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "a test prompt",
            "--negative-prompt",
            "blur",
            "--seed",
            "7",
            "--width",
            "512",
            "--height",
            "288",
            "--frames",
            "16",
            "--fps",
            "12",
            "--steps",
            "4",
            "--guidance",
            "3.5",
            "--quant",
            "q8",
            "--cache",
            "prompt",
            "--compile",
            "off",
            "--output-dir",
            str(output_dir),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert records
    assert {record["phase"] for record in records} >= {"model_load", "denoise_step", "total"}
    assert records[0]["model"] == "wan2.2"
    assert records[0]["backend"] == "stub"
    assert records[0]["negative_prompt_hash"]
    assert all(record["output_path"] is None for record in records)


def test_cli_save_video_writes_placeholder_and_records_output_path(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "run",
            "--model",
            "ltx2.3",
            "--backend",
            "stub",
            "--prompt",
            "video",
            "--negative-prompt",
            "",
            "--seed",
            "1",
            "--width",
            "320",
            "--height",
            "180",
            "--frames",
            "8",
            "--fps",
            "8",
            "--steps",
            "2",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "on",
            "--output-dir",
            str(output_dir),
            "--result-jsonl",
            str(jsonl_path),
            "--save-video",
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    output_paths = [record["output_path"] for record in records if record["output_path"]]
    assert output_paths
    assert output_paths[0].endswith(".stub.mp4")


def test_cli_run_appends_jsonl_records(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    args = [
        "run",
        "--model",
        "wan2.2",
        "--backend",
        "stub",
        "--prompt",
        "append",
        "--negative-prompt",
        "",
        "--seed",
        "2",
        "--width",
        "128",
        "--height",
        "128",
        "--frames",
        "2",
        "--fps",
        "2",
        "--steps",
        "1",
        "--guidance",
        "1.0",
        "--quant",
        "none",
        "--cache",
        "none",
        "--compile",
        "off",
        "--output-dir",
        str(tmp_path / "outputs"),
        "--result-jsonl",
        str(jsonl_path),
        "--no-save-video",
        "--dry-run",
    ]

    assert main(args) == 0
    first_count = len(_read_jsonl(jsonl_path))
    assert main(args) == 0

    assert len(_read_jsonl(jsonl_path)) == first_count * 2


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--width", "4097", "width must be no greater than 4096"),
        ("--height", "4097", "height must be no greater than 4096"),
        ("--frames", "258", "frames must be no greater than 257"),
        ("--steps", "513", "steps must be no greater than 512"),
        ("--fps", "241", "fps must be no greater than 240"),
    ],
)
def test_cli_rejects_manual_run_dimensions_above_memory_safe_caps(
    tmp_path,
    option,
    value,
    message,
):
    args = [
        "run",
        "--model",
        "wan2.2",
        "--backend",
        "stub",
        "--prompt",
        "oversized",
        "--negative-prompt",
        "",
        "--seed",
        "2",
        "--width",
        "128",
        "--height",
        "128",
        "--frames",
        "2",
        "--fps",
        "2",
        "--steps",
        "1",
        "--guidance",
        "1.0",
        "--quant",
        "none",
        "--cache",
        "none",
        "--compile",
        "off",
        "--output-dir",
        str(tmp_path / "outputs"),
        "--result-jsonl",
        str(tmp_path / "benchmarks.jsonl"),
        "--no-save-video",
        option,
        value,
    ]

    with pytest.raises(SystemExit, match=message):
        main(args)


def test_cli_rejects_preset_override_above_memory_safe_caps(tmp_path):
    with pytest.raises(SystemExit, match="steps must be no greater than 512"):
        main(
            [
                "run",
                "--preset",
                "cache-experiment",
                "--model",
                "wan2.2",
                "--backend",
                "stub",
                "--prompt",
                "oversized preset",
                "--negative-prompt",
                "",
                "--seed",
                "2",
                "--steps",
                "513",
                "--output-dir",
                str(tmp_path / "outputs"),
                "--result-jsonl",
                str(tmp_path / "benchmarks.jsonl"),
                "--no-save-video",
            ]
        )


def test_report_command_produces_markdown_from_jsonl(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    report_path = tmp_path / "report.md"

    main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "report",
            "--negative-prompt",
            "",
            "--seed",
            "2",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "2",
            "--guidance",
            "2.0",
            "--quant",
            "q4",
            "--cache",
            "all",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    exit_code = main(["report", "--input", str(jsonl_path), "--output", str(report_path)])

    assert exit_code == 0
    report = report_path.read_text(encoding="utf-8")
    assert "Total Time By Run" in report
    assert "average denoise step time" in report
    assert "Recommended Next Bottleneck" in report


def test_report_command_fails_closed_before_rendering_oversized_jsonl(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(cli_module, "MAX_REPORT_RECORDS", 1)
    jsonl_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "run_id": f"run-{index}",
                    "phase": "total",
                    "model": "wan2.2",
                    "backend": "stub",
                    "seconds": 0.0,
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSONL record limit"):
        main(["report", "--input", str(jsonl_path), "--output", str(report_path)])

    assert not report_path.exists()


def test_run_command_fails_closed_on_oversized_record_stream(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    monkeypatch.setattr(cli_module, "MAX_REPORT_RECORDS", 3)
    yielded = 0

    def oversized_records(config):
        nonlocal yielded
        for index in range(5):
            yielded += 1
            yield cli_module.make_record(
                config,
                run_id=f"run-{index}",
                timestamp_utc="2026-01-01T00:00:00Z",
                machine={},
                phase="total",
                seconds=0.0,
            )

    monkeypatch.setattr(cli_module.Profiler, "run", lambda self, config: oversized_records(config))

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "oversized stream",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "2",
            "--guidance",
            "2.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 1
    assert yielded == 4
    records = _read_jsonl(jsonl_path)
    assert len(records) == 1
    assert records[0]["phase"] == "total"
    assert "profile record limit exceeded" in records[0]["error"]


def test_profile_summary_bounds_text_and_ignores_non_finite_metrics():
    long_error = "failed: " + ("z" * 1_000)
    records = [
        {
            "run_id": "summary-run",
            "preset": "smoke",
            "variant_label": "bad|" + ("v" * 1_000),
            "phase": "total",
            "seconds": float("inf"),
            "peak_memory": -1,
            "error": long_error,
        },
        {
            "run_id": "summary-run",
            "preset": "smoke",
            "variant_label": "bad|" + ("v" * 1_000),
            "phase": "denoise_total",
            "seconds": float("nan"),
            "peak_memory": "not-an-int",
            "error": None,
        },
        {
            "run_id": "summary-run",
            "preset": "smoke",
            "variant_label": "bad|" + ("v" * 1_000),
            "phase": "decode",
            "seconds": -5,
            "peak_memory": 2048,
            "error": None,
        },
    ]

    rows = cli_module._profile_summary_rows(records)
    recommendation = cli_module._profile_recommendation(records)

    assert rows[0]["total"] == 0.0
    assert rows[0]["denoise_avg"] == 0.0
    assert rows[0]["peak_memory"] == "2048"
    assert rows[0]["variant"].startswith("bad/")
    assert "<truncated>" in rows[0]["variant"]
    assert "z" * 300 not in recommendation
    assert "inf" not in recommendation.lower()
    assert "nan" not in recommendation.lower()


def test_profile_command_runs_full_wan_suite_and_writes_comparison_report(tmp_path, capsys):
    results_dir = tmp_path / "profiles"

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "profile suite",
            "--seed",
            "11",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--results-dir",
            str(results_dir),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    jsonl_files = list(results_dir.glob("*T*_wan2.2.jsonl"))
    assert exit_code == 0
    assert len(jsonl_files) == 1
    report_path = jsonl_files[0].with_suffix(".md")
    assert report_path.exists()
    records = _read_jsonl(jsonl_files[0])
    totals = [record for record in records if record["phase"] == "total"]
    assert {record["preset"] for record in totals} == {
        "smoke",
        "small-baseline",
        "quality-threshold",
        "cache-experiment",
        "compile-experiment",
        "stress",
    }
    assert all(record["profile_id"] for record in records)
    assert all(record["profile_name"] == "wan2.2-full-preset-suite" for record in records)
    assert "Profile suite summary" in output
    assert "Recommended next bottleneck" in output
    report = report_path.read_text(encoding="utf-8")
    assert "Preset Comparison" in report
    assert "average denoise step" in report


def test_profile_command_skips_ltx23_stress(tmp_path, capsys):
    jsonl_path = tmp_path / "ltx-profile.jsonl"

    exit_code = main(
        [
            "profile",
            "--model",
            "ltx2.3",
            "--backend",
            "stub",
            "--prompt",
            "profile suite",
            "--seed",
            "12",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    records = _read_jsonl(jsonl_path)
    stress_records = [record for record in records if record["preset"] == "stress"]
    assert exit_code == 0
    assert stress_records
    assert all(record["error"].startswith("skipped:") for record in stress_records)
    assert "skipped" in output


def test_mlx_profile_guard_error_preserves_model_candidate(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "mlx-profile.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(
        cli_module,
        "_mlx_pre_run_guard",
        lambda label, config=None: "Memory guard blocked run: test pressure",
    )

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "profile suite",
            "--seed",
            "12",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    records = _read_jsonl(jsonl_path)
    assert exit_code == 1
    assert len(records) == 1
    assert records[0]["error"].startswith("Memory guard blocked run")
    assert records[0]["model_path"] == str(model_path.resolve())


def test_mlx_scaffold_runs_cli_memory_guard_before_failed_schema_records(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: calls.append(f"pre:{label}") or None)
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: calls.append(f"post:{label}") or {"cleanup": {"mlx_cache_cleared": False, "mlx_cleanup_error": None}},
    )

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx scaffold",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 1
    assert calls == ["pre:manual", "post:manual"]
    records = _read_jsonl(jsonl_path)
    assert records
    assert any(record["error"] for record in records)
    assert all(record["backend"] == "mlx" for record in records)
    assert all(record["model_path"] == str(model_path.resolve()) for record in records)


def test_mlx_scaffold_cannot_bypass_pre_run_memory_guard(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: True)
    monkeypatch.setattr(
        cli_module,
        "_mlx_pre_run_guard",
        lambda label, config=None: "Memory guard blocked run: scaffold blocked",
    )
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: (_ for _ in ()).throw(AssertionError("blocked pre-run must not run cleanup")),
    )

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx guarded scaffold",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    records = _read_jsonl(jsonl_path)
    assert exit_code == 1
    assert len(records) == 1
    assert records[0]["error"] == "Memory guard blocked run: scaffold blocked"


def test_mlx_run_applies_pre_run_memory_guard(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(
        cli_module,
        "_mlx_pre_run_guard",
        lambda label, config=None: "Memory guard blocked run: test pressure",
    )

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx guarded",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert len(records) == 1
    assert records[0]["phase"] == "total"
    assert "Memory guard blocked run" in records[0]["error"]


def test_mlx_real_backend_requires_parent_process_opt_in(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module.Profiler,
        "run",
        lambda self, config: (_ for _ in ()).throw(AssertionError("parent Profiler.run must be blocked")),
    )

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx parent blocked",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    records = _read_jsonl(jsonl_path)
    assert exit_code == 1
    assert len(records) == 1
    assert records[0]["phase"] == "total"
    assert cli_module.ALLOW_PARENT_MLX_ENV in records[0]["error"]
    assert "parent process" in records[0]["error"]


def test_mlx_profile_real_backend_requires_parent_process_opt_in(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "mlx-profile.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module.Profiler,
        "run",
        lambda self, config: (_ for _ in ()).throw(AssertionError("parent Profiler.run must be blocked")),
    )

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "profile parent blocked",
            "--seed",
            "12",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    records = _read_jsonl(jsonl_path)
    assert exit_code == 1
    assert len(records) == 1
    assert records[0]["phase"] == "total"
    assert records[0]["model_path"] == str(model_path.resolve())
    assert cli_module.ALLOW_PARENT_MLX_ENV in records[0]["error"]
    assert "parent process" in records[0]["error"]


def test_mlx_runtime_abort_records_cleanup_status(tmp_path, monkeypatch):
    from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setenv(cli_module.ALLOW_PARENT_MLX_ENV, "1")
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: {"cleanup": {"mlx_cache_cleared": True, "mlx_cleanup_error": None}},
    )
    monkeypatch.setattr(
        cli_module.Profiler,
        "run",
        lambda self, config: (_ for _ in ()).throw(RuntimeMemoryAbort("runtime stop")),
    )

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx abort",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert records[0]["error"] == "Runtime memory abort: runtime stop"
    assert records[0]["machine"]["mlx_guard_cleanup"] == {
        "mlx_cache_cleared": True,
        "mlx_cleanup_error": None,
    }


def test_mlx_run_cleanup_failure_fails_closed_after_success(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setenv(cli_module.ALLOW_PARENT_MLX_ENV, "1")
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: {
            "cleanup": {
                "mlx_loaded": True,
                "mlx_cache_cleared": False,
                "mlx_cleanup_error": "failed to clear MLX cache",
            },
            "run_number": 1,
        },
    )

    def fake_run(self, config):
        return [
            cli_module.make_record(
                config,
                run_id="run",
                timestamp_utc="2026-01-01T00:00:00Z",
                machine={},
                phase="total",
                seconds=0.0,
            )
        ]

    monkeypatch.setattr(cli_module.Profiler, "run", fake_run)

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx cleanup",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert records[0]["phase"] == "total"
    assert records[0]["error"] == (
        "Memory guard blocked run: MLX post-run cleanup failed: failed to clear MLX cache"
    )
    assert records[0]["machine"]["mlx_guard_cleanup"] == {
        "mlx_loaded": True,
        "mlx_cache_cleared": False,
        "mlx_cleanup_error": "failed to clear MLX cache",
    }


def test_mlx_profile_cleanup_failure_fails_closed_after_success(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "mlx-profile.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setenv(cli_module.ALLOW_PARENT_MLX_ENV, "1")
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: {
            "cleanup": {
                "mlx_loaded": True,
                "mlx_cache_cleared": False,
                "mlx_cleanup_error": "failed to clear MLX cache",
            },
            "run_number": 1,
        },
    )

    def fake_run(self, config):
        return [
            cli_module.make_record(
                config,
                run_id="profile-run",
                timestamp_utc="2026-01-01T00:00:00Z",
                machine={},
                phase="total",
                seconds=0.0,
            )
        ]

    monkeypatch.setattr(cli_module.Profiler, "run", fake_run)

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "profile cleanup",
            "--seed",
            "12",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert records[0]["phase"] == "total"
    assert records[0]["error"] == (
        "Memory guard blocked run: MLX post-run cleanup failed: failed to clear MLX cache"
    )
    assert records[0]["machine"]["mlx_guard_cleanup"] == {
        "mlx_loaded": True,
        "mlx_cache_cleared": False,
        "mlx_cleanup_error": "failed to clear MLX cache",
    }


def test_profile_command_fails_closed_before_materializing_oversized_backend_records(tmp_path, monkeypatch):
    from fastgen_profiler.reports.markdown import MAX_REPORT_RECORDS

    jsonl_path = tmp_path / "profile.jsonl"

    class OversizedRecords:
        def __len__(self):
            return MAX_REPORT_RECORDS + 1

        def __iter__(self):
            raise AssertionError("oversized profile records must be rejected before iteration")

    monkeypatch.setattr(cli_module.Profiler, "run", lambda self, config: OversizedRecords())

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "oversized records",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--fps",
            "4",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--result-jsonl",
            str(jsonl_path),
            "--results-dir",
            str(tmp_path / "profiles"),
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert len(records) == 1
    assert records[0]["phase"] == "total"
    assert "profile record limit exceeded" in records[0]["error"]


def test_profile_command_fails_closed_on_oversized_record_stream(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "profile.jsonl"
    monkeypatch.setattr(cli_module, "MAX_REPORT_RECORDS", 3)
    yielded = 0

    def oversized_records(config):
        nonlocal yielded
        for index in range(5):
            yielded += 1
            yield cli_module.make_record(
                config,
                run_id=f"run-{index}",
                timestamp_utc="2026-01-01T00:00:00Z",
                machine={},
                phase="total",
                seconds=0.0,
            )

    monkeypatch.setattr(cli_module.Profiler, "run", lambda self, config: oversized_records(config))

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "oversized stream",
            "--negative-prompt",
            "",
            "--seed",
            "3",
            "--fps",
            "4",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--result-jsonl",
            str(jsonl_path),
            "--results-dir",
            str(tmp_path / "profiles"),
        ]
    )

    assert exit_code == 1
    assert yielded == 4
    records = _read_jsonl(jsonl_path)
    assert len(records) == 1
    assert records[0]["phase"] == "total"
    assert "profile record limit exceeded" in records[0]["error"]


def test_mlx_inner_memory_guard_error_records_cleanup_status(tmp_path, monkeypatch):
    from fastgen_profiler.mlx_guard import MemoryGuardError

    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setenv(cli_module.ALLOW_PARENT_MLX_ENV, "1")
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: {"cleanup": {"mlx_cache_cleared": False, "mlx_cleanup_error": None}},
    )
    monkeypatch.setattr(
        cli_module.Profiler,
        "run",
        lambda self, config: (_ for _ in ()).throw(MemoryGuardError("file preflight blocked")),
    )

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "mlx guard",
            "--seed",
            "3",
            "--width",
            "256",
            "--height",
            "256",
            "--frames",
            "4",
            "--fps",
            "4",
            "--steps",
            "1",
            "--guidance",
            "1.0",
            "--quant",
            "none",
            "--cache",
            "none",
            "--compile",
            "off",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--no-save-video",
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert records[0]["error"] == "Memory guard blocked run: file preflight blocked"
    assert records[0]["machine"]["mlx_guard_cleanup"] == {
        "mlx_cache_cleared": False,
        "mlx_cleanup_error": None,
    }


def test_mlx_profile_inner_memory_guard_error_preserves_model_candidate(tmp_path, monkeypatch):
    from fastgen_profiler.mlx_guard import MemoryGuardError

    jsonl_path = tmp_path / "mlx-profile.jsonl"
    model_path = tmp_path / "wan-model"
    model_path.mkdir()
    monkeypatch.setenv(cli_module.ALLOW_PARENT_MLX_ENV, "1")
    monkeypatch.setattr(cli_module, "_backend_is_scaffold_only", lambda backend: False)
    monkeypatch.setattr(cli_module, "_mlx_pre_run_guard", lambda label, config=None: None)
    monkeypatch.setattr(
        cli_module,
        "_mlx_post_run_cleanup",
        lambda label: {"cleanup": {"mlx_cache_cleared": False, "mlx_cleanup_error": "failed"}},
    )
    monkeypatch.setattr(
        cli_module.Profiler,
        "run",
        lambda self, config: (_ for _ in ()).throw(MemoryGuardError("adapter guard blocked")),
    )

    exit_code = main(
        [
            "profile",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-path",
            str(model_path),
            "--prompt",
            "profile guard",
            "--seed",
            "12",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    records = _read_jsonl(jsonl_path)
    assert exit_code == 1
    assert records[0]["error"] == "Memory guard blocked run: adapter guard blocked"
    assert records[0]["model_path"] == str(model_path.resolve())
    assert records[0]["machine"]["mlx_guard_cleanup"] == {
        "mlx_cache_cleared": False,
        "mlx_cleanup_error": "failed",
    }


def test_mlx_pre_run_guard_checks_shape_budget_before_system_recovery(tmp_path, monkeypatch):
    import fastgen_profiler.mlx_guard as mlx_guard

    calls: list[str] = []
    monkeypatch.setattr(mlx_guard, "should_restart_process", lambda: False)
    monkeypatch.setattr(mlx_guard, "run_counter", lambda: 0)
    monkeypatch.setattr(
        mlx_guard,
        "check_run_allocation_budget",
        lambda **kwargs: calls.append("budget") or {"shape_floor_gb": 1},
    )
    monkeypatch.setattr(
        mlx_guard,
        "check_text_prompt_budget",
        lambda **kwargs: calls.append("prompt") or {"prompt_chars": 6},
    )
    monkeypatch.setattr(
        mlx_guard,
        "inter_run_system_recovery",
        lambda label: calls.append("recovery") or {
            "free_gb": 100,
            "freed_gb": 0,
            "run_number": 1,
        },
    )
    monkeypatch.setattr(
        mlx_guard,
        "configure_mlx_resource_limits",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("CLI pre-run must not import/configure MLX")),
    )

    config = RunConfig(
        model="ltx2.3",
        backend="mlx",
        model_path=str(tmp_path),
        model_id=None,
        model_source_root=None,
        prompt="prompt",
        negative_prompt="",
        seed=1,
        width=512,
        height=512,
        frames=9,
        fps=24,
        steps=1,
        guidance=1.0,
        quant="none",
        cache="none",
        compile="off",
        output_dir=tmp_path / "outputs",
        result_jsonl=tmp_path / "results.jsonl",
        save_video=False,
        dry_run=False,
    )

    assert cli_module._mlx_pre_run_guard("order", config=config) is None
    assert calls == ["budget", "prompt", "recovery"]


def test_mlx_pre_run_guard_blocks_oversized_prompt_before_recovery(tmp_path, monkeypatch):
    import fastgen_profiler.mlx_guard as mlx_guard
    from fastgen_profiler.mlx_guard import MemoryGuardError

    calls: list[str] = []
    monkeypatch.setattr(mlx_guard, "should_restart_process", lambda: False)
    monkeypatch.setattr(mlx_guard, "run_counter", lambda: 0)
    monkeypatch.setattr(
        mlx_guard,
        "check_run_allocation_budget",
        lambda **kwargs: calls.append("budget") or {"shape_floor_gb": 1},
    )
    monkeypatch.setattr(
        mlx_guard,
        "check_text_prompt_budget",
        lambda **kwargs: calls.append("prompt") or (_ for _ in ()).throw(MemoryGuardError("prompt too large")),
    )
    monkeypatch.setattr(
        mlx_guard,
        "inter_run_system_recovery",
        lambda label: calls.append("recovery") or {"free_gb": 100},
    )

    config = RunConfig(
        model="ltx2.3",
        backend="mlx",
        model_path=str(tmp_path),
        model_id=None,
        model_source_root=None,
        prompt="x" * 9000,
        negative_prompt="",
        seed=1,
        width=512,
        height=512,
        frames=9,
        fps=24,
        steps=1,
        guidance=1.0,
        quant="none",
        cache="none",
        compile="off",
        output_dir=tmp_path / "outputs",
        result_jsonl=tmp_path / "results.jsonl",
        save_video=False,
        dry_run=False,
    )

    error = cli_module._mlx_pre_run_guard("prompt", config=config)

    assert error == "Memory guard blocked run: prompt too large"
    assert calls == ["budget", "prompt"]


def test_mlx_pre_run_guard_blocks_after_one_completed_run(monkeypatch):
    import fastgen_profiler.mlx_guard as mlx_guard

    monkeypatch.setattr(mlx_guard, "run_counter", lambda: 1)
    monkeypatch.setattr(mlx_guard, "should_restart_process", lambda: True)

    error = cli_module._mlx_pre_run_guard("second")

    assert error is not None
    assert "process restart required after 1 consecutive MLX runs" in error


def test_mlx_pre_run_guard_fails_closed_when_guard_unavailable(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fastgen_profiler.mlx_guard":
            raise ImportError("missing guard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    error = cli_module._mlx_pre_run_guard("missing-guard")

    assert error is not None
    assert "mlx_guard unavailable before MLX run" in error


def test_mlx_post_run_cleanup_reports_guard_import_failure(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fastgen_profiler.mlx_guard":
            raise ImportError("missing guard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    status = cli_module._mlx_post_run_cleanup("missing-guard")

    assert status is not None
    assert status["cleanup"] == {
        "mlx_loaded": None,
        "mlx_cache_cleared": False,
        "mlx_cleanup_error": "mlx_guard unavailable after MLX run",
    }


def test_mlx_post_run_cleanup_keeps_cleanup_when_snapshot_fails(monkeypatch):
    import fastgen_profiler.mlx_guard as mlx_guard

    monkeypatch.setattr(mlx_guard, "mlx_cleanup", lambda: {"freed_gb": 0.5})
    monkeypatch.setattr(mlx_guard, "increment_run_counter", lambda: 1)
    monkeypatch.setattr(mlx_guard, "system_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("snapshot failed")))

    status = cli_module._mlx_post_run_cleanup("snapshot-failed")

    assert status == {
        "snapshot": None,
        "cleanup": {"freed_gb": 0.5},
        "run_number": 1,
    }


def test_smoke_preset_applies_requested_shape_and_defaults(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "preset smoke",
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    total = [record for record in records if record["phase"] == "total"]
    assert len(total) == 1
    assert total[0]["width"] == 384
    assert total[0]["height"] == 384
    assert total[0]["frames"] == 16
    assert total[0]["steps"] == 8
    assert total[0]["cache"] == "none"
    assert total[0]["compile"] == "off"
    assert all(record["output_path"] is None for record in records)


def test_small_baseline_preset_appends_guidance_and_quant_variants(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "small-baseline",
            "--model",
            "ltx2.3",
            "--backend",
            "stub",
            "--prompt",
            "small baseline",
            "--seed",
            "2",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    totals = [record for record in _read_jsonl(jsonl_path) if record["phase"] == "total"]
    assert len(totals) == 4
    assert {(record["guidance"], record["quant"]) for record in totals} == {
        (1.0, "none"),
        (1.0, "q8p"),
        (3.5, "none"),
        (3.5, "q8p"),
    }
    assert all(record["width"] == 512 and record["height"] == 512 for record in totals)


def test_quality_threshold_preset_saves_video_for_step_variants(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "quality-threshold",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "quality threshold",
            "--seed",
            "3",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    totals = [record for record in _read_jsonl(jsonl_path) if record["phase"] == "total"]
    assert [record["steps"] for record in totals] == [16, 24, 32, 40]
    assert all(record["output_path"] for record in totals)


def test_cache_and_compile_presets_expand_variants(tmp_path):
    cache_jsonl = tmp_path / "cache.jsonl"
    compile_jsonl = tmp_path / "compile.jsonl"
    common_args = [
        "--model",
        "wan2.2",
        "--backend",
        "stub",
        "--prompt",
        "variant",
        "--seed",
        "4",
        "--output-dir",
        str(tmp_path / "outputs"),
    ]

    assert main(["run", "--preset", "cache-experiment", *common_args, "--result-jsonl", str(cache_jsonl)]) == 0
    assert main(["run", "--preset", "compile-experiment", *common_args, "--result-jsonl", str(compile_jsonl)]) == 0

    cache_totals = [record for record in _read_jsonl(cache_jsonl) if record["phase"] == "total"]
    compile_totals = [record for record in _read_jsonl(compile_jsonl) if record["phase"] == "total"]
    assert [record["cache"] for record in cache_totals] == ["none", "prompt", "feature", "all"]
    assert [record["compile"] for record in compile_totals] == ["off", "on"]


def test_stress_preset_rejects_ltx23_until_backend_stabilizes(tmp_path):
    try:
        main(
            [
                "run",
                "--preset",
                "stress",
                "--model",
                "ltx2.3",
                "--backend",
                "stub",
                "--prompt",
                "stress",
                "--seed",
                "5",
                "--output-dir",
                str(tmp_path / "outputs"),
                "--result-jsonl",
                str(tmp_path / "stress.jsonl"),
            ]
        )
    except SystemExit as exc:
        assert "wan2.2" in str(exc)
    else:
        raise AssertionError("stress preset should reject ltx2.3")


def test_missing_preset_prompts_for_selection_when_interactive(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    class InteractiveStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: "1")

    exit_code = main(
        [
            "run",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--prompt",
            "interactive preset",
            "--seed",
            "6",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    totals = [record for record in _read_jsonl(jsonl_path) if record["phase"] == "total"]
    assert len(totals) == 1
    assert totals[0]["width"] == 384


def test_models_list_uses_env_and_cli_model_dirs(tmp_path, capsys):
    env_root = tmp_path / "env-root"
    cli_root = tmp_path / "cli-root"
    env_model = env_root / "wan-env"
    cli_model = cli_root / "wan-cli"
    ltx_model = env_root / "ltx-env"
    env_model.mkdir(parents=True)
    cli_model.mkdir(parents=True)
    ltx_model.mkdir(parents=True)
    (env_model / "config.json").write_text("{}", encoding="utf-8")
    (cli_model / "model.safetensors").write_text("", encoding="utf-8")
    (ltx_model / "model.safetensors").write_text("", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(f"FASTGEN_MODEL_DIRS={env_root}\n", encoding="utf-8")

    exit_code = main(
        [
            "models",
            "list",
            "--env-file",
            str(env_file),
            "--model-dir",
            str(cli_root),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-env" in output
    assert "wan-cli" in output
    assert "ltx-env" in output


def test_run_records_direct_model_path(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_path = tmp_path / "direct-wan"
    model_path.mkdir()

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--model-path",
            str(model_path),
            "--prompt",
            "direct model",
            "--seed",
            "7",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert records
    assert all(record["model_path"] == str(model_path.resolve()) for record in records)
    assert all(record["model_id"] == model_path.name for record in records)


def test_run_selects_model_id_from_env_dirs(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_root = tmp_path / "models"
    model_path = model_root / "nested" / "wan-local"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(f"FASTGEN_MODEL_DIRS={model_root}\n", encoding="utf-8")

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "stub",
            "--model-id",
            "nested/wan-local",
            "--env-file",
            str(env_file),
            "--prompt",
            "model id",
            "--seed",
            "8",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 0
    records = _read_jsonl(jsonl_path)
    assert all(record["model_id"] == "nested/wan-local" for record in records)
    assert all(record["model_source_root"] == str(model_root.resolve()) for record in records)


def test_interactive_mlx_model_selection(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    model_root = tmp_path / "models"
    first = model_root / "wan-first"
    second = model_root / "wan-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "config.json").write_text("{}", encoding="utf-8")
    (second / "config.json").write_text("{}", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: "2")

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--model-dir",
            str(model_root),
            "--prompt",
            "interactive model",
            "--seed",
            "9",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert all(record["model_id"] == "wan-second" for record in records)
    assert any("wan-second" in record["error"] for record in records if record["error"])


def test_non_interactive_mlx_without_model_selection_writes_error_record(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"

    exit_code = main(
        [
            "run",
            "--preset",
            "smoke",
            "--model",
            "wan2.2",
            "--backend",
            "mlx",
            "--prompt",
            "missing model",
            "--seed",
            "10",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--result-jsonl",
            str(jsonl_path),
        ]
    )

    assert exit_code == 1
    records = _read_jsonl(jsonl_path)
    assert len(records) == 1
    assert records[0]["phase"] == "model_load"
    assert records[0]["model_path"] is None
    assert "model selection required" in records[0]["error"]


def test_models_import_dry_run_discovers_dirs_without_writing_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    hf_dir = tmp_path / ".cache/huggingface/hub"
    hf_model = hf_dir / "models--owner--wan2.2" / "snapshots" / "abc123"
    hf_model.mkdir(parents=True)
    (hf_model / "model_index.json").write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "all",
            "--env-file",
            str(env_file),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Discovered app roots" in output
    assert "Generation model directories to register" in output
    assert str(hf_dir.resolve()) in output
    assert str(hf_model.resolve()) in output
    assert "Dry run" in output
    assert not env_file.exists()


def test_models_import_writes_env_non_interactive(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    comfy_dir = tmp_path / "ComfyUI/models"
    comfy_model = comfy_dir / "diffusion_models" / "wan2.2-video"
    old_dir = tmp_path / "old-models"
    comfy_model.mkdir(parents=True)
    (comfy_model / "model.safetensors").write_text("", encoding="utf-8")
    old_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"# existing\nFASTGEN_MODEL_DIRS={old_dir}\nOTHER=value\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "comfyui",
            "--env-file",
            str(env_file),
        ]
    )

    content = env_file.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "# existing" in content
    assert "OTHER=value" in content
    assert f"FASTGEN_MODEL_DIRS={comfy_model.resolve()}" in content
    assert str(old_dir) not in content


def test_models_import_fails_when_roots_have_no_generation_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    lmstudio_dir = tmp_path / ".cache/lm-studio/models"
    llm_dir = lmstudio_dir / "owner" / "chat-model"
    llm_dir.mkdir(parents=True)
    (llm_dir / "model.gguf").write_text("", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=value\n", encoding="utf-8")

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "lmstudio",
            "--env-file",
            str(env_file),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "No Wan2.2/LTX2.3 generation model candidates found" in output
    assert env_file.read_text(encoding="utf-8") == "OTHER=value\n"


def test_models_import_fails_when_no_directories_found(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    env_file = tmp_path / ".env"

    exit_code = main(
        [
            "models",
            "import",
            "--source",
            "all",
            "--env-file",
            str(env_file),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "No known model directories found." in output
    assert not env_file.exists()


def test_interactive_main_menu_can_import_model_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    draw_dir = tmp_path / "Documents/Draw Things/Models"
    draw_model = draw_dir / "wan2.2-draw"
    draw_model.mkdir(parents=True)
    (draw_model / "model.safetensors").write_text("", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["3", "y"])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main([])

    assert exit_code == 0
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert str(draw_model.resolve()) in content


def test_interactive_main_menu_run_profile_creates_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["1", "", "", "", "", "", "", ""])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main([])

    assert exit_code == 0
    records = _read_jsonl(tmp_path / "artifacts/results.jsonl")
    assert records
    assert records[0]["backend"] == "stub"
    assert records[0]["model"] == "wan2.2"


def test_interactive_main_menu_list_models_outputs_all_candidates_without_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model_root = tmp_path / "models"
    wan_path = model_root / "wan-menu"
    ltx_path = model_root / "ltx-menu"
    wan_path.mkdir(parents=True)
    ltx_path.mkdir(parents=True)
    (wan_path / "config.json").write_text("{}", encoding="utf-8")
    (ltx_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(f"FASTGEN_MODEL_DIRS={model_root}\n", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["2"])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-menu" in output
    assert "ltx-menu" in output


def test_run_command_prompts_for_missing_required_values_interactively(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class InteractiveStdin:
        def isatty(self):
            return True

    answers = iter(["", "", "", "", "", "", ""])
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    exit_code = main(["run"])

    assert exit_code == 0
    assert (tmp_path / "artifacts/results.jsonl").exists()


def test_models_command_without_subcommand_lists_interactively(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model_root = tmp_path / "models"
    model_path = model_root / "wan-models-command"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text(f"FASTGEN_MODEL_DIRS={model_root}\n", encoding="utf-8")

    class InteractiveStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    monkeypatch.setattr("builtins.input", lambda _: "")

    exit_code = main(["models"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-models-command" in output


def test_models_list_without_model_outputs_all_candidates(tmp_path, capsys):
    model_root = tmp_path / "models"
    wan_path = model_root / "wan-list-command"
    ltx_path = model_root / "ltx-list-command"
    wan_path.mkdir(parents=True)
    ltx_path.mkdir(parents=True)
    (wan_path / "config.json").write_text("{}", encoding="utf-8")
    (ltx_path / "config.json").write_text("{}", encoding="utf-8")

    exit_code = main(["models", "list", "--model-dir", str(model_root)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "wan-list-command" in output
    assert "ltx-list-command" in output


def test_run_command_missing_required_values_fails_non_interactively():
    try:
        main(["run"])
    except SystemExit as exc:
        assert "--model" in str(exc)
    else:
        raise AssertionError("run without required values should fail when non-interactive")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
