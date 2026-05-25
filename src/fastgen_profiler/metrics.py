"""Benchmark metric schema and JSONL serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BenchmarkConfig:
    model: str
    backend: str
    prompt: str
    seed: int
    width: int
    height: int
    frames: int
    steps: int
    precision: str
    guidance: float | None
    cache_enabled: bool
    compile_enabled: bool


@dataclass(slots=True)
class PhaseTiming:
    name: str
    duration_seconds: float
    synchronization: str
    notes: str | None = None


@dataclass(slots=True)
class StepTiming:
    step_index: int
    duration_seconds: float
    synchronization: str
    sampler: str | None = None
    guidance: float | None = None
    cache_enabled: bool | None = None
    compile_enabled: bool | None = None


@dataclass(slots=True)
class BenchmarkResult:
    config: BenchmarkConfig
    phase_timings: list[PhaseTiming]
    step_timings: list[StepTiming] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
