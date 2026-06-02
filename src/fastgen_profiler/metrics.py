"""Benchmark metric schema and JSONL serialization."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import uuid


REQUIRED_PHASES = (
    "model_load",
    "prompt_prepare",
    "text_encoder",
    "latent_init",
    "denoise_total",
    "denoise_step",
    "vae_decode",
    "video_encode",
    "file_write",
    "total",
)

DEFAULT_JSONL_READ_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_JSONL_READ_MAX_RECORDS = 100_000
MAX_METRIC_TEXT_FIELD_CHARS = 2_048
MAX_METRIC_COLLECTION_ITEMS = 256
MAX_METRIC_RECORD_JSON_BYTES = 256 * 1024
MAX_MACHINE_METADATA_OUTPUT_BYTES = 16 * 1024
MAX_RUN_DIMENSION = 4096
MAX_RUN_FRAMES = 257
MAX_RUN_FPS = 240
MAX_RUN_STEPS = 512


@dataclass(slots=True)
class RunConfig:
    model: str
    backend: str
    model_path: str | None
    model_id: str | None
    model_source_root: str | None
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    guidance: float
    quant: str
    cache: str
    compile: str
    output_dir: Path
    result_jsonl: Path
    save_video: bool
    dry_run: bool
    profile_id: str | None = None
    profile_name: str | None = None
    preset: str | None = None
    variant_label: str | None = None


@dataclass(slots=True)
class MeasurementRecord:
    run_id: str
    timestamp_utc: str
    model: str
    backend: str
    model_path: str | None
    model_id: str | None
    model_source_root: str | None
    prompt_hash: str
    negative_prompt_hash: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    guidance: float
    quant: str
    cache: str
    compile: str
    phase: str
    step_index: int | None
    seconds: float
    peak_memory: int | None
    active_memory: int | None
    cache_memory: int | None
    output_path: str | None
    error: str | None
    profile_id: str | None
    profile_name: str | None
    preset: str | None
    variant_label: str | None
    machine: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Avoid dataclasses.asdict(): it recursively deep-copies nested machine
        # metadata before we can enforce JSONL bounds.
        return {
            record_field.name: _bound_metric_value(getattr(self, record_field.name))
            for record_field in fields(self)
        }


def new_run_id() -> str:
    return str(uuid.uuid4())


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def machine_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "python_version": sys.version.split()[0],
        "mlx_version": _mlx_version(),
        "processor": platform.processor() or None,
        "machine": platform.machine() or None,
        "apple_silicon": platform.machine() == "arm64" and sys.platform == "darwin",
        "chip": _sysctl_value("machdep.cpu.brand_string"),
        "total_memory": _sysctl_int("hw.memsize"),
    }
    return metadata


def make_record(
    config: RunConfig,
    *,
    run_id: str,
    timestamp_utc: str,
    machine: dict[str, Any],
    phase: str,
    seconds: float,
    step_index: int | None = None,
    peak_memory: int | None = None,
    active_memory: int | None = None,
    cache_memory: int | None = None,
    output_path: str | None = None,
    error: str | None = None,
) -> MeasurementRecord:
    return MeasurementRecord(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        model=config.model,
        backend=config.backend,
        model_path=config.model_path,
        model_id=config.model_id,
        model_source_root=config.model_source_root,
        prompt_hash=prompt_hash(config.prompt),
        negative_prompt_hash=prompt_hash(config.negative_prompt),
        seed=config.seed,
        width=config.width,
        height=config.height,
        frames=config.frames,
        fps=config.fps,
        steps=config.steps,
        guidance=config.guidance,
        quant=config.quant,
        cache=config.cache,
        compile=config.compile,
        phase=phase,
        step_index=step_index,
        seconds=seconds,
        peak_memory=peak_memory,
        active_memory=active_memory,
        cache_memory=cache_memory,
        output_path=output_path,
        error=error,
        profile_id=config.profile_id,
        profile_name=config.profile_name,
        preset=config.preset,
        variant_label=config.variant_label,
        machine=machine,
    )


def validate_run_config_safety(config: RunConfig) -> None:
    """Validate core run bounds before any backend can allocate or emit records."""
    _validate_positive_capped_int(config.width, "width", MAX_RUN_DIMENSION)
    _validate_positive_capped_int(config.height, "height", MAX_RUN_DIMENSION)
    _validate_positive_capped_int(config.frames, "frames", MAX_RUN_FRAMES)
    _validate_positive_capped_int(config.fps, "fps", MAX_RUN_FPS)
    _validate_positive_capped_int(config.steps, "steps", MAX_RUN_STEPS)


def _validate_positive_capped_int(value: object, name: str, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > max_value:
        raise ValueError(f"{name} must be no greater than {max_value}")


def append_jsonl(
    path: Path,
    records: Iterable[MeasurementRecord],
    *,
    max_records: int = DEFAULT_JSONL_READ_MAX_RECORDS,
    max_record_bytes: int = MAX_METRIC_RECORD_JSON_BYTES,
) -> None:
    if max_records <= 0:
        raise ValueError("JSONL write record limit must be positive")
    if max_record_bytes <= 0:
        raise ValueError("JSONL write record byte limit must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if index >= max_records:
                raise ValueError(
                    f"{path} exceeds JSONL write record limit: more than {max_records} records"
                )
            line = json.dumps(record.to_dict(), sort_keys=True)
            line_bytes = len(line.encode("utf-8")) + 1
            if line_bytes > max_record_bytes:
                raise ValueError(
                    f"{path} JSONL record exceeds write byte limit: "
                    f"{line_bytes} bytes > {max_record_bytes} bytes"
                )
            handle.write(line + "\n")


def _bound_metric_value(value: Any) -> Any:
    if isinstance(value, str):
        return _bound_metric_text(value)
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_METRIC_COLLECTION_ITEMS:
                bounded["__truncated_items__"] = True
                break
            bounded[_safe_metric_text(key)] = _bound_metric_value(item)
        return bounded
    if isinstance(value, list):
        return _bound_metric_sequence(value)
    if isinstance(value, tuple):
        return _bound_metric_sequence(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_metric_text(value)


def _safe_metric_text(value: Any) -> str:
    if isinstance(value, str):
        return _bound_metric_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return _bound_metric_text(str(value))
    value_type = type(value)
    return _bound_metric_text(f"<{value_type.__module__}.{value_type.__qualname__}>")


def _bound_metric_text(value: str) -> str:
    if len(value) <= MAX_METRIC_TEXT_FIELD_CHARS:
        return value
    suffix = "...<truncated>"
    return value[: MAX_METRIC_TEXT_FIELD_CHARS - len(suffix)] + suffix


def _bound_metric_sequence(value: Iterable[Any]) -> list[Any]:
    bounded = []
    for index, item in enumerate(value):
        if index >= MAX_METRIC_COLLECTION_ITEMS:
            bounded.append({"__truncated_items__": True})
            break
        bounded.append(_bound_metric_value(item))
    return bounded


def read_jsonl(
    path: Path,
    *,
    max_bytes: int = DEFAULT_JSONL_READ_MAX_BYTES,
    max_records: int = DEFAULT_JSONL_READ_MAX_RECORDS,
) -> list[dict[str, Any]]:
    if max_bytes <= 0:
        raise ValueError("JSONL read byte limit must be positive")
    if max_records <= 0:
        raise ValueError("JSONL read record limit must be positive")
    file_size = path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(
            f"{path} exceeds JSONL read limit: {file_size} bytes > {max_bytes} bytes"
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if len(records) >= max_records:
                raise ValueError(
                    f"{path} exceeds JSONL record limit: more than {max_records} records"
                )
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return records


def _mlx_version() -> str | None:
    try:
        return metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return None


def _sysctl_value(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        returncode, stdout = _run_bounded_stdout(
            ["sysctl", "-n", name],
            max_bytes=MAX_MACHINE_METADATA_OUTPUT_BYTES,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if returncode != 0:
        return None
    return stdout.strip() or None


def _sysctl_int(name: str) -> int | None:
    value = _sysctl_value(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _run_bounded_stdout(args: list[str], *, max_bytes: int, timeout: float) -> tuple[int, str]:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError(f"max_bytes must be a positive integer, got {max_bytes!r}")

    with tempfile.TemporaryFile() as stdout:
        result = subprocess.run(
            args,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        stdout.seek(0, os.SEEK_END)
        size = stdout.tell()
        if size > max_bytes:
            raise OSError(
                f"{args[0]} output exceeded metadata limit: {size} bytes > {max_bytes} bytes"
            )
        stdout.seek(0)
        data = stdout.read(max_bytes)
    return result.returncode, data.decode("utf-8", errors="replace")
