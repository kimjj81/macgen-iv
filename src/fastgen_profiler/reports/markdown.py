"""Markdown report rendering."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def render_markdown_report(records: list[dict[str, Any]]) -> str:
    runs = _group_by_run(records)
    lines = ["# FastGen Profile Report", ""]

    if not runs:
        return "# FastGen Profile Report\n\nNo records found.\n"

    lines.extend(
        [
            "## Total Time By Run",
            "",
            "| run | preset | variant | model | backend | total seconds | status |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for run_id, run_records in runs.items():
        first = run_records[0]
        total = _phase_seconds(run_records, "total")
        status = _status(run_records)
        lines.append(
            f"| `{run_id}` | `{first.get('preset') or 'manual'}` | "
            f"`{first.get('variant_label') or 'manual'}` | `{first['model']}` | "
            f"`{first['backend']}` | {total:.6f} | {status} |"
        )

    lines.extend(["", "## Preset Comparison", ""])
    lines.extend(_preset_comparison_lines(runs))

    lines.extend(["", "## Phase Time Breakdown", ""])
    for run_id, run_records in runs.items():
        lines.extend([f"### `{run_id}`", "", "| phase | seconds |", "| --- | ---: |"])
        for phase, seconds in _phase_breakdown(run_records):
            lines.append(f"| `{phase}` | {seconds:.6f} |")
        average_step = _average_denoise_step(run_records)
        slowest_phase = _slowest_phase(run_records)
        peak_memory = _peak_memory(run_records)
        lines.extend(
            [
                "",
                f"- average denoise step time: `{average_step:.6f}` seconds",
                f"- slowest phase: `{slowest_phase[0]}` at `{slowest_phase[1]:.6f}` seconds",
                f"- peak memory: `{_format_memory(peak_memory)}`",
                "",
            ]
        )

    skipped_runs = [
        (run_id, _errors(run_records))
        for run_id, run_records in runs.items()
        if _status(run_records) == "skipped"
    ]
    failed_runs = [
        (run_id, _errors(run_records))
        for run_id, run_records in runs.items()
        if _status(run_records) == "failed"
    ]
    lines.extend(["## Skipped Runs", ""])
    if skipped_runs:
        for run_id, errors in skipped_runs:
            lines.append(f"- `{run_id}`: {errors[0]}")
    else:
        lines.append("No skipped runs.")

    lines.extend(["## Failed Runs", ""])
    if failed_runs:
        for run_id, errors in failed_runs:
            lines.append(f"- `{run_id}`: {errors[0]}")
    else:
        lines.append("No failed runs.")

    lines.extend(["", "## Recommended Next Bottleneck To Inspect", ""])
    lines.append(_recommendation(runs))
    return "\n".join(lines) + "\n"


def _group_by_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["run_id"])].append(record)
    return dict(grouped)


def _phase_seconds(records: list[dict[str, Any]], phase: str) -> float:
    return sum(float(record["seconds"]) for record in records if record["phase"] == phase)


def _phase_breakdown(records: list[dict[str, Any]]) -> list[tuple[str, float]]:
    phases: dict[str, float] = defaultdict(float)
    for record in records:
        if record["phase"] == "denoise_step":
            continue
        phases[str(record["phase"])] += float(record["seconds"])
    return sorted(phases.items(), key=lambda item: item[0])


def _preset_comparison_lines(runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| preset | variant | total seconds | denoise total | average denoise step | slowest phase | peak memory | status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run_records in runs.values():
        first = run_records[0]
        slowest_phase = _slowest_phase(run_records)
        lines.append(
            f"| `{first.get('preset') or 'manual'}` | `{first.get('variant_label') or 'manual'}` | "
            f"{_phase_seconds(run_records, 'total'):.6f} | "
            f"{_phase_seconds(run_records, 'denoise_total'):.6f} | "
            f"{_average_denoise_step(run_records):.6f} | "
            f"`{slowest_phase[0]}` | `{_format_memory(_peak_memory(run_records))}` | "
            f"{_status(run_records)} |"
        )
    return lines


def _average_denoise_step(records: list[dict[str, Any]]) -> float:
    steps = [float(record["seconds"]) for record in records if record["phase"] == "denoise_step"]
    if not steps:
        return 0.0
    return sum(steps) / len(steps)


def _slowest_phase(records: list[dict[str, Any]]) -> tuple[str, float]:
    breakdown = [(phase, seconds) for phase, seconds in _phase_breakdown(records) if phase != "total"]
    if not breakdown:
        return ("none", 0.0)
    return max(breakdown, key=lambda item: item[1])


def _peak_memory(records: list[dict[str, Any]]) -> int | None:
    values = [record["peak_memory"] for record in records if record.get("peak_memory") is not None]
    if not values:
        return None
    return max(int(value) for value in values)


def _errors(records: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for record in records:
        error = record.get("error")
        if error and error not in seen:
            seen.append(str(error))
    return seen


def _status(records: list[dict[str, Any]]) -> str:
    errors = _errors(records)
    if any(error.startswith("skipped:") for error in errors):
        return "skipped"
    if errors:
        return "failed"
    return "ok"


def _recommendation(runs: dict[str, list[dict[str, Any]]]) -> str:
    for run_id, records in runs.items():
        if _status(records) == "skipped":
            continue
        errors = _errors(records)
        if errors:
            return f"Fix failed run `{run_id}` first: {errors[0]}"

    phase_totals: dict[str, float] = defaultdict(float)
    denoise_total = 0.0
    total_time = 0.0
    for records in runs.values():
        total_time += _phase_seconds(records, "total")
        denoise_total += _phase_seconds(records, "denoise_total")
        for phase, seconds in _phase_breakdown(records):
            if phase != "total":
                phase_totals[phase] += seconds

    if not phase_totals:
        return "No phase timing data is available."

    phase, seconds = max(phase_totals.items(), key=lambda item: item[1])
    if total_time > 0 and phase == "denoise_total":
        share = denoise_total / total_time * 100.0
        return f"Inspect `denoise_total` first; it accounts for {share:.1f}% of recorded total time."
    return f"Inspect `{phase}` first; it is the slowest recorded non-total phase at {seconds:.6f} seconds."


def _format_memory(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value} bytes"
