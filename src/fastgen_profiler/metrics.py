"""Benchmark metric schema and JSONL serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
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
        return asdict(self)


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


def append_jsonl(path: Path, records: Iterable[MeasurementRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
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
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _sysctl_int(name: str) -> int | None:
    value = _sysctl_value(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
