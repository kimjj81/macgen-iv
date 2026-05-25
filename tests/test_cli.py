from __future__ import annotations

import json

from fastgen_profiler.cli import main


def test_cli_dry_run_writes_jsonl(tmp_path):
    jsonl_path = tmp_path / "benchmarks.jsonl"
    report_path = tmp_path / "report.md"

    exit_code = main(
        [
            "--model",
            "wan2.2",
            "--prompt",
            "a test prompt",
            "--seed",
            "7",
            "--width",
            "512",
            "--height",
            "288",
            "--frames",
            "16",
            "--steps",
            "4",
            "--jsonl",
            str(jsonl_path),
            "--report",
            str(report_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    line = jsonl_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["config"]["model"] == "wan2.2"
    assert record["phase_timings"][0]["name"] == "total"
    assert report_path.exists()
