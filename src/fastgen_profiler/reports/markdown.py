"""Markdown report rendering."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any


MAX_REPORT_FIELD_CHARS = 256
MAX_REPORT_NUMERIC_FIELD_CHARS = 64
MAX_REPORT_RECORDS = 10_000
MAX_REPORT_RUNS = 1_000


def render_markdown_report(records: list[dict[str, Any]]) -> str:
    _validate_report_size(records)
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
            f"| `{_report_text(run_id)}` | `{_report_text(first.get('preset') or 'manual')}` | "
            f"`{_report_text(first.get('variant_label') or 'manual')}` | `{_report_text(first['model'])}` | "
            f"`{_report_text(first['backend'])}` | {total:.6f} | {status} |"
        )

    lines.extend(["", "## Preset Comparison", ""])
    lines.extend(_preset_comparison_lines(runs))

    lines.extend(["", "## Phase Time Breakdown", ""])
    for run_id, run_records in runs.items():
        lines.extend([f"### `{_report_text(run_id)}`", "", "| phase | seconds |", "| --- | ---: |"])
        for phase, seconds in _phase_breakdown(run_records):
            lines.append(f"| `{_report_text(phase)}` | {seconds:.6f} |")
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
            lines.append(f"- `{_report_text(run_id)}`: {_report_text(errors[0])}")
    else:
        lines.append("No skipped runs.")

    lines.extend(["## Failed Runs", ""])
    if failed_runs:
        for run_id, errors in failed_runs:
            lines.append(f"- `{_report_text(run_id)}`: {_report_text(errors[0])}")
    else:
        lines.append("No failed runs.")

    lines.extend(["", "## Recommended Next Bottleneck To Inspect", ""])
    lines.append(_recommendation(runs))
    return "\n".join(lines) + "\n"


def _validate_report_size(records: list[dict[str, Any]]) -> None:
    if len(records) > MAX_REPORT_RECORDS:
        raise ValueError(
            f"report record limit exceeded: {len(records)} records > {MAX_REPORT_RECORDS}"
        )
    run_ids: set[str] = set()
    for record in records:
        run_ids.add(_report_text(record["run_id"]))
        if len(run_ids) > MAX_REPORT_RUNS:
            raise ValueError(
                f"report run limit exceeded: more than {MAX_REPORT_RUNS} runs"
            )


def _group_by_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_report_text(record["run_id"])].append(record)
    return dict(grouped)


def _phase_seconds(records: list[dict[str, Any]], phase: str) -> float:
    return sum(_finite_seconds(record) for record in records if record["phase"] == phase)


def _phase_breakdown(records: list[dict[str, Any]]) -> list[tuple[str, float]]:
    phases: dict[str, float] = defaultdict(float)
    for record in records:
        if record["phase"] == "denoise_step":
            continue
        phases[_report_text(record["phase"])] += _finite_seconds(record)
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
            f"| `{_report_text(first.get('preset') or 'manual')}` | `{_report_text(first.get('variant_label') or 'manual')}` | "
            f"{_phase_seconds(run_records, 'total'):.6f} | "
            f"{_phase_seconds(run_records, 'denoise_total'):.6f} | "
            f"{_average_denoise_step(run_records):.6f} | "
            f"`{_report_text(slowest_phase[0])}` | `{_format_memory(_peak_memory(run_records))}` | "
            f"{_status(run_records)} |"
        )
    return lines


def _average_denoise_step(records: list[dict[str, Any]]) -> float:
    steps = [_finite_seconds(record) for record in records if record["phase"] == "denoise_step"]
    if not steps:
        return 0.0
    return sum(steps) / len(steps)


def _slowest_phase(records: list[dict[str, Any]]) -> tuple[str, float]:
    breakdown = [(phase, seconds) for phase, seconds in _phase_breakdown(records) if phase != "total"]
    if not breakdown:
        return ("none", 0.0)
    return max(breakdown, key=lambda item: item[1])


def _peak_memory(records: list[dict[str, Any]]) -> int | None:
    values = [_non_negative_int(record["peak_memory"]) for record in records if record.get("peak_memory") is not None]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def _errors(records: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for record in records:
        error = record.get("error")
        sanitized_error = _report_text(error) if error else ""
        if sanitized_error and sanitized_error not in seen:
            seen.append(sanitized_error)
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
            return f"Fix failed run `{_report_text(run_id)}` first: {_report_text(errors[0])}"

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
    return f"Inspect `{_report_text(phase)}` first; it is the slowest recorded non-total phase at {seconds:.6f} seconds."


def _format_memory(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value} bytes"


def _finite_seconds(record: dict[str, Any]) -> float:
    raw = record["seconds"]
    if type(raw) not in {int, float, str}:
        return 0.0
    if isinstance(raw, str):
        raw = _bounded_numeric_text(raw)
        if raw is None:
            return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value


def _non_negative_int(value: Any) -> int | None:
    if type(value) not in {int, str}:
        return None
    if isinstance(value, str):
        value = _bounded_numeric_text(value)
        if value is None:
            return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    if converted < 0:
        return None
    return converted


def _bounded_numeric_text(value: str) -> str | None:
    text = value.strip()
    if len(text) > MAX_REPORT_NUMERIC_FIELD_CHARS:
        return None
    return text


def _report_text(value: Any) -> str:
    text = _safe_report_text(value).replace("\n", " ").replace("\r", " ").replace("|", "/")
    if len(text) <= MAX_REPORT_FIELD_CHARS:
        return text
    return f"{text[: MAX_REPORT_FIELD_CHARS - 12]}...<truncated>"


def _safe_report_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"
