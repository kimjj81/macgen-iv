"""Common backend adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import sys
from time import perf_counter
from typing import Iterator

from fastgen_profiler.metrics import MeasurementRecord, RunConfig, make_record


class BackendAdapter(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        config: RunConfig,
        *,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
    ) -> list[MeasurementRecord]:
        """Run one profile and return flat measurement records."""

    def record(
        self,
        config: RunConfig,
        *,
        run_id: str,
        timestamp_utc: str,
        machine: dict[str, object],
        phase: str,
        seconds: float,
        step_index: int | None = None,
        output_path: str | None = None,
        error: str | None = None,
    ) -> MeasurementRecord:
        memory = mlx_memory_snapshot()
        return make_record(
            config,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            machine=machine,
            phase=phase,
            step_index=step_index,
            seconds=seconds,
            active_memory=memory["active_memory"],
            peak_memory=memory["peak_memory"],
            cache_memory=memory["cache_memory"],
            output_path=output_path,
            error=error,
        )


@contextmanager
def timed_section(sync_target: object | None = None) -> Iterator[dict[str, float]]:
    synchronize_mlx(sync_target)
    started = perf_counter()
    timing: dict[str, float] = {}
    try:
        yield timing
    finally:
        synchronize_mlx(sync_target)
        timing["seconds"] = perf_counter() - started


def synchronize_mlx(target: object | None = None) -> None:
    if target is None:
        return

    # Synchronization must not be the operation that initializes MLX/Metal.
    # Backend adapters are responsible for running the memory guard before
    # importing mlx.core.
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return

    try:
        mx.eval(target)
        return
    except Exception as exc:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort, mlx_cleanup

        target = None
        _clear_traceback_frames(exc)
        mlx_cleanup()
        _detach_exception(exc)

    from fastgen_profiler.mlx_guard import RuntimeMemoryAbort

    raise RuntimeMemoryAbort(
        "Runtime memory abort [synchronize]: MLX synchronization failed; "
        "aborting because Metal runtime state may be unsafe."
    )


def mlx_memory_snapshot() -> dict[str, int | None]:
    # Importing MLX can initialize native runtime state. Only read MLX counters
    # when a backend has already imported mlx.core in this process.
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return {"active_memory": None, "peak_memory": None, "cache_memory": None}
    try:
        return {
            "active_memory": _memory_counter_or_none(mx.get_active_memory()),
            "peak_memory": _memory_counter_or_none(mx.get_peak_memory()),
            "cache_memory": _memory_counter_or_none(mx.get_cache_memory()),
        }
    except Exception:
        return {"active_memory": None, "peak_memory": None, "cache_memory": None}


def _memory_counter_or_none(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _clear_traceback_frames(exc: BaseException) -> None:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        tb = current.__traceback__
        while tb is not None:
            try:
                tb.tb_frame.clear()
            except RuntimeError:
                pass
            tb = tb.tb_next
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)


def _detach_exception(exc: BaseException) -> None:
    try:
        exc.__traceback__ = None
    except Exception:
        pass
    try:
        exc.__cause__ = None
    except Exception:
        pass
    try:
        exc.__context__ = None
    except Exception:
        pass
