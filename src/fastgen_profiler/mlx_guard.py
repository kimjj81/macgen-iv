"""MLX memory and system guardian for repeated benchmark runs.

Prevents watchdog timeout kernel panics by monitoring memory pressure,
enforcing inter-run cleanup, and limiting consecutive runs per process.

Three guard layers:
  1. Pre-run system resource check (memory, swap, pressure)
  2. Adaptive batch sizing (start small, grow or shrink based on headroom)
  3. Runtime per-step memory watchdog (abort before kernel panic)
"""

from __future__ import annotations

import gc
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum free memory (bytes) required before starting a run.
# Keep this aligned with the system reserve used for host allocations.
DEFAULT_MIN_FREE_BYTES = 8 * 1024 ** 3  # 8 GB

# Maximum MLX runs allowed in one process before forcing a restart.
# MLX/Metal resource state can survive Python-level cleanup, so the safe
# default is one heavy run per process. Suite runners should respawn children
# instead of reusing a process for multiple model executions.
MAX_CONSECUTIVE_RUNS = 1

# Seconds to sleep between runs for thermal/Metal cooldown.
COOLDOWN_SECONDS = 5

# Maximum number of swap files before we refuse to start a run.
MAX_SWAP_FILES = 20

# Memory pressure fraction threshold (0.0-1.0) for runtime watchdog.
# Above this, the current step loop is aborted to prevent kernel panic.
RUNTIME_PRESSURE_ABORT_THRESHOLD = 0.92
RUNTIME_MLX_LIMIT_ABORT_FRACTION = 0.92

# Free memory threshold for runtime watchdog.
# Below this, abort before unified memory pressure can destabilize the OS.
RUNTIME_MIN_FREE_BYTES = 4 * 1024 ** 3  # 4 GB

# Keep memory outside MLX so the OS remains responsive. MLX's default memory
# limit can be larger than the device working set, so profiler runs set an
# explicit process-local cap before touching model weights.
DEFAULT_SYSTEM_RESERVE_BYTES = 8 * 1024 ** 3  # 8 GB
DEFAULT_MLX_MEMORY_FRACTION = 0.80
DEFAULT_MLX_CACHE_LIMIT_BYTES = 1 * 1024 ** 3  # 1 GB
MIN_MLX_MEMORY_LIMIT_BYTES = 512 * 1024 ** 2  # 512 MiB
DEFAULT_MAX_PROMPT_CHARS = 8192
MAX_PROMPT_CHARS = 65_536
MAX_TELEMETRY_OUTPUT_BYTES = 64 * 1024

# Adaptive batch sizing defaults.
ADAPTIVE_INITIAL_FRAMES = 5
ADAPTIVE_INITIAL_STEPS = 4
ADAPTIVE_MAX_GROWTH_FACTOR = 2.0
ADAPTIVE_HEADROOM_THRESHOLD = 0.3  # grow if >30% memory free after probe
ADAPTIVE_SHRINK_THRESHOLD = 0.15   # shrink if <15% memory free after probe
MAX_ADAPTIVE_BATCH_HISTORY = 256

logger = logging.getLogger("fastgen_profiler.mlx_guard")
_MLX_IMPORT_PROBE_ENV = "FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE"
_current_mlx_memory_limit_bytes: int | None = None


# ---------------------------------------------------------------------------
# Memory introspection (macOS vm_stat)
# ---------------------------------------------------------------------------

def _vm_stat() -> dict[str, int]:
    """Parse macOS vm_stat output into a dict of page counts."""
    try:
        returncode, stdout = _run_bounded_stdout(["vm_stat"], timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if returncode != 0:
        return {}

    pages: dict[str, int] = {}
    for line in stdout.splitlines():
        if "page size of" in line.lower():
            parts = line.replace(".", "").split()
            for index, part in enumerate(parts):
                if part == "of" and index + 1 < len(parts):
                    try:
                        pages["__page_size__"] = int(parts[index + 1].replace(",", ""))
                    except ValueError:
                        pass
                    break
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".").replace(",", "")
        try:
            pages[key.strip()] = int(value)
        except ValueError:
            continue
    return pages


def free_memory_bytes() -> int | None:
    """Return free + inactive memory in bytes on macOS, or None if unknown."""
    pages = _vm_stat()
    if not pages:
        return None
    page_size = pages.get("__page_size__", 16384)
    free = pages.get("Pages free", 0)
    inactive = pages.get("Pages inactive", 0)
    return (free + inactive) * page_size


def total_memory_bytes() -> int | None:
    """Return total physical memory in bytes, or None if unknown."""
    if sys.platform != "darwin":
        return None
    try:
        returncode, stdout = _run_bounded_stdout(["sysctl", "-n", "hw.memsize"], timeout=5)
        if returncode != 0:
            return None
        return int(stdout.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def swap_file_count() -> int | None:
    """Return number of active swapfiles on macOS, or None if unknown."""
    if sys.platform != "darwin":
        return None
    swap_dir = Path("/private/var/vm")
    try:
        if not swap_dir.is_dir():
            return 0
        return sum(1 for f in swap_dir.iterdir() if f.name.startswith("swapfile"))
    except OSError:
        return None


def memory_pressure_fraction() -> float | None:
    """Return memory pressure as 0.0-1.0 on macOS, or None if unknown."""
    try:
        returncode, stdout = _run_bounded_stdout(["memory_pressure"], timeout=10)
        if returncode != 0:
            return None
        # Output: "System-wide memory free percentage: 72%"
        for line in stdout.splitlines():
            if "percentage" in line.lower():
                pct_str = line.rsplit(":", 1)[-1].strip().rstrip("%")
                try:
                    used_pct = 100 - float(pct_str)
                    return used_pct / 100.0
                except ValueError:
                    continue
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _run_bounded_stdout(
    args: list[str],
    *,
    timeout: float,
    max_bytes: int = MAX_TELEMETRY_OUTPUT_BYTES,
) -> tuple[int, str]:
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
                f"{args[0]} output exceeded telemetry limit: {size} bytes > {max_bytes} bytes"
            )
        stdout.seek(0)
        data = stdout.read(max_bytes)
    return result.returncode, data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# System resource snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    """Point-in-time view of system memory and swap state."""
    free_bytes: int | None
    total_bytes: int | None
    pressure: float | None
    swap_files: int | None
    free_fraction: float | None  # free/total if both known

    def free_gb(self) -> str:
        if self.free_bytes is None:
            return "?"
        return f"{self.free_bytes / 1e9:.1f}"

    def summary(self) -> str:
        parts = [f"free={self.free_gb()}GB"]
        if self.total_bytes is not None and self.free_fraction is not None:
            parts.append(f"({self.free_fraction * 100:.0f}% avail)")
        if self.pressure is not None:
            parts.append(f"pressure={self.pressure * 100:.0f}%")
        if self.swap_files is not None:
            parts.append(f"swap={self.swap_files}")
        return " ".join(parts)


def system_snapshot() -> SystemSnapshot:
    """Capture current system memory state."""
    try:
        free = free_memory_bytes()
    except Exception:
        free = None
    try:
        total = total_memory_bytes()
    except Exception:
        total = None
    try:
        pressure = memory_pressure_fraction()
    except Exception:
        pressure = None
    try:
        swap = swap_file_count()
    except Exception:
        swap = None

    free_frac = None
    if (
        _is_non_negative_int(free)
        and _is_positive_int(total)
        and free <= total
    ):
        free_frac = free / total

    return SystemSnapshot(
        free_bytes=free,
        total_bytes=total,
        pressure=pressure,
        swap_files=swap,
        free_fraction=free_frac,
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_unit_interval_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _invalid_system_snapshot_reasons(snap: SystemSnapshot) -> list[str]:
    """Return impossible memory telemetry values that must fail closed."""
    reasons: list[str] = []

    if snap.free_bytes is not None and not _is_non_negative_int(snap.free_bytes):
        reasons.append(f"free_bytes must be a non-negative integer, got {snap.free_bytes!r}")
    if snap.total_bytes is not None and not _is_positive_int(snap.total_bytes):
        reasons.append(f"total_bytes must be a positive integer, got {snap.total_bytes!r}")
    if (
        _is_non_negative_int(snap.free_bytes)
        and _is_positive_int(snap.total_bytes)
        and snap.free_bytes > snap.total_bytes
    ):
        reasons.append(
            f"free_bytes cannot exceed total_bytes ({snap.free_bytes!r} > {snap.total_bytes!r})"
        )
    if snap.pressure is not None and not _is_unit_interval_number(snap.pressure):
        reasons.append(f"pressure must be a finite number in [0, 1], got {snap.pressure!r}")
    if snap.swap_files is not None and not _is_non_negative_int(snap.swap_files):
        reasons.append(f"swap_files must be a non-negative integer, got {snap.swap_files!r}")
    if snap.free_fraction is not None and not _is_unit_interval_number(snap.free_fraction):
        reasons.append(
            f"free_fraction must be a finite number in [0, 1], got {snap.free_fraction!r}"
        )

    return reasons


def _invalid_system_snapshot_message(label: str, reasons: list[str]) -> str:
    return (
        f"Memory guard [{label}]: invalid memory telemetry: "
        + "; ".join(reasons)
    )


# ---------------------------------------------------------------------------
# MLX cleanup
# ---------------------------------------------------------------------------

def mlx_cleanup() -> dict[str, object]:
    """Aggressive MLX + Python cleanup between benchmark runs.

    Returns a status dict with memory info for logging.
    """
    before_free = None
    mlx_loaded = False
    mlx_cache_cleared = False
    mlx_cleanup_error = None
    memory_telemetry_error = None

    try:
        before_free = free_memory_bytes()
    except Exception:
        memory_telemetry_error = "failed to read free memory before cleanup"

    # Force Python garbage collection (multiple passes)
    for _ in range(3):
        gc.collect()

    # Clear MLX Metal cache only if this process has already initialized MLX.
    # Cleanup must not open a fresh Metal runtime while handling a memory abort.
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        mlx_loaded = True
        try:
            mx.clear_cache()
            mlx_cache_cleared = True
        except Exception:
            mlx_cleanup_error = "failed to clear MLX cache"

    # Another GC pass after MLX cleanup
    gc.collect()

    after_free = None
    try:
        after_free = free_memory_bytes()
    except Exception:
        if memory_telemetry_error is None:
            memory_telemetry_error = "failed to read free memory after cleanup"

    return {
        "free_before_gb": round(before_free / 1e9, 2) if before_free else None,
        "free_after_gb": round(after_free / 1e9, 2) if after_free else None,
        "freed_gb": round((after_free - before_free) / 1e9, 2)
        if (before_free is not None and after_free is not None)
        else None,
        "mlx_loaded": mlx_loaded,
        "mlx_cache_cleared": mlx_cache_cleared,
        "mlx_cleanup_error": mlx_cleanup_error,
        "memory_telemetry_error": memory_telemetry_error,
    }


def _env_gb_to_bytes(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        gb = float(value)
    except ValueError as exc:
        raise MemoryGuardError(f"{name} must be a number of GB, got {value!r}") from exc
    if not math.isfinite(gb) or gb <= 0:
        raise MemoryGuardError(f"{name} must be a finite number of GB greater than zero")
    bytes_value = int(gb * 1024 ** 3)
    if bytes_value <= 0:
        raise MemoryGuardError(f"{name} is too small to represent at byte precision")
    return bytes_value


def _system_reserve_bytes(default: int = DEFAULT_SYSTEM_RESERVE_BYTES) -> int:
    override = _env_gb_to_bytes("FASTGEN_SYSTEM_RESERVE_GB")
    if override is None:
        return default
    return max(default, override)


def _default_mlx_memory_limit(total_bytes: int | None) -> int | None:
    if total_bytes is None:
        return None
    reserved_limit = total_bytes - _system_reserve_bytes()
    fraction_limit = int(total_bytes * DEFAULT_MLX_MEMORY_FRACTION)
    return min(reserved_limit, fraction_limit)


def _free_memory_mlx_limit(free_bytes: int | None) -> int | None:
    if free_bytes is None:
        return None
    return free_bytes - _system_reserve_bytes()


def _validate_pre_run_snapshot(
    snap: SystemSnapshot,
    *,
    label: str,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    max_swap_files: int = MAX_SWAP_FILES,
) -> None:
    """Fail closed before any MLX import or allocator probe."""
    invalid_reasons = _invalid_system_snapshot_reasons(snap)
    if invalid_reasons:
        raise MemoryGuardError(_invalid_system_snapshot_message(label, invalid_reasons))

    if sys.platform == "darwin" and snap.free_bytes is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read vm_stat free memory. "
            "Refusing to start an MLX/Metal run without free memory telemetry."
        )
    if sys.platform == "darwin" and snap.swap_files is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read macOS swap file state. "
            "Refusing to start an MLX/Metal run without swap telemetry."
        )
    if sys.platform == "darwin" and snap.pressure is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read macOS memory pressure. "
            "Refusing to start an MLX/Metal run without pressure telemetry."
        )

    if snap.free_bytes is not None and snap.free_bytes < min_free_bytes:
        raise MemoryGuardError(
            f"Memory guard [{label}]: only {round(snap.free_bytes / 1e9, 1)}GB free, "
            f"need >= {round(min_free_bytes / 1e9, 1)}GB before importing MLX/Metal."
        )

    if snap.pressure is not None and snap.pressure > 0.95:
        raise MemoryGuardError(
            f"Memory guard [{label}]: pressure at {round(snap.pressure * 100, 0)}%, "
            "system is near OOM. Refusing to import MLX/Metal."
        )

    if snap.swap_files is not None and snap.swap_files > max_swap_files:
        raise MemoryGuardError(
            f"Memory guard [{label}]: {snap.swap_files} swap files active "
            f"(max {max_swap_files}). Refusing to import MLX/Metal while heavily swapping."
        )


def configure_mlx_resource_limits(
    *,
    snapshot: SystemSnapshot | None = None,
    label: str = "",
) -> dict[str, object]:
    """Set conservative MLX allocator limits for this process.

    Environment overrides, in GB:
      FASTGEN_MLX_MEMORY_LIMIT_GB
      FASTGEN_MLX_CACHE_LIMIT_GB
      FASTGEN_MLX_WIRED_LIMIT_GB
    """
    snap = snapshot or system_snapshot()
    _validate_pre_run_snapshot(snap, label=label)
    default_memory_limit = _default_mlx_memory_limit(snap.total_bytes)
    free_memory_limit = _free_memory_mlx_limit(snap.free_bytes)
    if default_memory_limit is None and sys.platform == "darwin":
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot derive MLX memory limit because "
            "total memory telemetry is unavailable. Refusing to trust explicit "
            "allocator overrides without a system-reserve clamp; fix sysctl "
            "hw.memsize access before starting MLX/Metal."
        )
    if free_memory_limit is None and sys.platform == "darwin":
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot derive MLX memory limit because "
            "free memory telemetry is unavailable. Refusing to import MLX/Metal "
            "without preserving the system reserve from current headroom."
        )
    memory_limit = _env_gb_to_bytes("FASTGEN_MLX_MEMORY_LIMIT_GB")
    if memory_limit is None:
        memory_limit = default_memory_limit
    elif default_memory_limit is not None:
        memory_limit = min(memory_limit, default_memory_limit)
    if memory_limit is not None and free_memory_limit is not None:
        memory_limit = min(memory_limit, free_memory_limit)
    if memory_limit is not None and memory_limit < MIN_MLX_MEMORY_LIMIT_BYTES:
        raise MemoryGuardError(
            f"Memory guard [{label}]: safe MLX memory limit would be "
            f"{memory_limit / 1e9:.2f}GB, below minimum "
            f"{MIN_MLX_MEMORY_LIMIT_BYTES / 1e9:.2f}GB after preserving system reserve."
        )
    cache_limit = _env_gb_to_bytes("FASTGEN_MLX_CACHE_LIMIT_GB")
    if cache_limit is None:
        cache_limit = DEFAULT_MLX_CACHE_LIMIT_BYTES
    if memory_limit is not None:
        cache_limit = min(cache_limit, memory_limit)
    wired_limit = _env_gb_to_bytes("FASTGEN_MLX_WIRED_LIMIT_GB")
    if wired_limit is None and memory_limit is not None and sys.platform == "darwin":
        wired_limit = memory_limit
    elif wired_limit is not None and memory_limit is not None:
        wired_limit = min(wired_limit, memory_limit)

    status: dict[str, object] = {
        "label": label,
        "memory_limit_gb": round(memory_limit / 1e9, 2) if memory_limit else None,
        "cache_limit_gb": round(cache_limit / 1e9, 2) if cache_limit else None,
        "wired_limit_gb": round(wired_limit / 1e9, 2) if wired_limit else None,
        "previous_memory_limit_gb": None,
        "previous_cache_limit_gb": None,
        "previous_wired_limit_gb": None,
        "wired_limit_error": None,
    }

    _probe_mlx_import(label)

    try:
        import mlx.core as mx
    except Exception as exc:
        mlx_cleanup()
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot import mlx.core to set allocator limits"
        ) from exc

    try:
        if memory_limit is not None:
            previous = mx.set_memory_limit(memory_limit)
            status["previous_memory_limit_gb"] = round(previous / 1e9, 2)
        if cache_limit is not None:
            previous = mx.set_cache_limit(cache_limit)
            status["previous_cache_limit_gb"] = round(previous / 1e9, 2)
        if wired_limit is not None:
            try:
                previous = mx.set_wired_limit(wired_limit)
                status["previous_wired_limit_gb"] = round(previous / 1e9, 2)
            except Exception as exc:
                status["wired_limit_error"] = str(exc)
                mlx_cleanup()
                raise MemoryGuardError(
                    f"Memory guard [{label}]: failed to set MLX wired memory limit: {exc}"
                ) from exc
    except MemoryGuardError:
        raise
    except Exception as exc:
        mlx_cleanup()
        raise MemoryGuardError(
            f"Memory guard [{label}]: failed to set MLX memory limits: {exc}"
        ) from exc

    global _current_mlx_memory_limit_bytes
    _current_mlx_memory_limit_bytes = memory_limit
    return status


def _probe_mlx_import(label: str) -> None:
    """Check MLX/Metal availability in a child process before importing here."""
    if _test_skip_mlx_import_probe_enabled():
        return
    code = (
        "import mlx.core as mx\n"
        "mx.set_cache_limit(0)\n"
        "mx.clear_cache()\n"
    )
    env = dict(os.environ)
    env[_MLX_IMPORT_PROBE_ENV] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot verify MLX/Metal availability: {exc}"
        ) from exc
    if result.returncode != 0:
        raise MemoryGuardError(
            f"Memory guard [{label}]: MLX/Metal is unavailable before run "
            f"(probe exit {result.returncode})."
        )


def _test_skip_mlx_import_probe_enabled() -> bool:
    """Allow MLX import probe bypass only inside pytest."""
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    return (
        os.environ.get(_MLX_IMPORT_PROBE_ENV) == "1"
        and "pytest" in sys.modules
        and "::" in current_test
        and current_test.endswith((" (setup)", " (call)", " (teardown)"))
    )


# ---------------------------------------------------------------------------
# Pre-run checks (Guard 1: System resource gate)
# ---------------------------------------------------------------------------

class MemoryGuardError(RuntimeError):
    """Raised when system memory is too low to safely start a benchmark run."""


def check_memory_guard(
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    max_swap_files: int = MAX_SWAP_FILES,
    label: str = "",
) -> dict[str, object]:
    """Check if system has enough free memory for a benchmark run.

    Checks: free memory, memory pressure, swap file count.
    Returns status dict. Raises MemoryGuardError if insufficient.
    """
    snap = system_snapshot()
    invalid_reasons = _invalid_system_snapshot_reasons(snap)
    if invalid_reasons:
        raise MemoryGuardError(_invalid_system_snapshot_message(label, invalid_reasons))

    status: dict[str, object] = {
        "label": label,
        "free_gb": round(snap.free_bytes / 1e9, 2) if snap.free_bytes else None,
        "pressure": round(snap.pressure, 3) if snap.pressure is not None else None,
        "swap_files": snap.swap_files,
        "min_free_gb": round(min_free_bytes / 1e9, 2),
    }

    if sys.platform == "darwin" and snap.free_bytes is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read vm_stat free memory. "
            "Refusing to start an MLX/Metal run without free memory telemetry."
        )
    if sys.platform == "darwin" and snap.swap_files is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read macOS swap file state. "
            "Refusing to start an MLX/Metal run without swap telemetry."
        )
    if sys.platform == "darwin" and snap.pressure is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read macOS memory pressure. "
            "Refusing to start an MLX/Metal run without pressure telemetry."
        )

    if snap.free_bytes is not None and snap.free_bytes < min_free_bytes:
        raise MemoryGuardError(
            f"Memory guard [{label}]: only {round(snap.free_bytes / 1e9, 1)}GB free, "
            f"need >= {round(min_free_bytes / 1e9, 1)}GB. "
            "Skipping run to prevent kernel panic. "
            "Try reducing frames/steps or wait for memory to free up."
        )

    if snap.pressure is not None and snap.pressure > 0.95:
        raise MemoryGuardError(
            f"Memory guard [{label}]: pressure at {round(snap.pressure * 100, 0)}%, "
            "system is near OOM. Skipping run."
        )

    if snap.swap_files is not None and snap.swap_files > max_swap_files:
        raise MemoryGuardError(
            f"Memory guard [{label}]: {snap.swap_files} swap files active "
            f"(max {max_swap_files}). System is swapping heavily. Skipping run."
        )

    return status


# ---------------------------------------------------------------------------
# Runtime memory watchdog (Guard 3: Per-step abort)
# ---------------------------------------------------------------------------

class RuntimeMemoryAbort(RuntimeError):
    """Raised during a step loop when memory crosses the danger threshold."""


def check_runtime_memory(label: str = "") -> SystemSnapshot:
    """Check memory state during a running step loop.

    Returns the snapshot for logging.
    Raises RuntimeMemoryAbort if the system is about to OOM.
    """
    _check_mlx_runtime_limit(label)
    try:
        snap = system_snapshot()
    except Exception as exc:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot capture system memory telemetry "
            "during MLX/Metal run. Aborting because runtime memory state is unknown."
        ) from exc

    invalid_reasons = _invalid_system_snapshot_reasons(snap)
    if invalid_reasons:
        mlx_cleanup()
        raise RuntimeMemoryAbort(_invalid_system_snapshot_message(label, invalid_reasons))

    if sys.platform == "darwin" and snap.free_bytes is None:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot read vm_stat free memory "
            "during MLX/Metal run. Aborting because free memory telemetry is unavailable."
        )
    if sys.platform == "darwin" and snap.swap_files is None:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot read macOS swap file state "
            "during MLX/Metal run. Aborting because swap telemetry is unavailable."
        )
    if sys.platform == "darwin" and snap.pressure is None:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot read macOS memory pressure "
            "during MLX/Metal run. Aborting because pressure telemetry is unavailable."
        )

    if snap.free_bytes is not None and snap.free_bytes < RUNTIME_MIN_FREE_BYTES:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: only {snap.free_gb()}GB free "
            f"(threshold {RUNTIME_MIN_FREE_BYTES / 1e9:.0f}GB). "
            "Aborting run to prevent kernel watchdog timeout."
        )

    if snap.pressure is not None and snap.pressure > RUNTIME_PRESSURE_ABORT_THRESHOLD:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: pressure at {snap.pressure * 100:.0f}% "
            f"(threshold {RUNTIME_PRESSURE_ABORT_THRESHOLD * 100:.0f}%). "
            "Aborting run to prevent kernel watchdog timeout."
        )

    return snap


def _check_mlx_runtime_limit(label: str) -> None:
    if _current_mlx_memory_limit_bytes is None:
        return
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return
    try:
        active = _mlx_counter_bytes(mx.get_active_memory(), "active")
        cache = _mlx_counter_bytes(mx.get_cache_memory(), "cache")
    except Exception as exc:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot read MLX allocator memory "
            "while a configured memory limit is active."
        ) from exc
    used = active + cache
    threshold = int(_current_mlx_memory_limit_bytes * RUNTIME_MLX_LIMIT_ABORT_FRACTION)
    if used >= threshold:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: MLX active+cache memory "
            f"{used / 1e9:.1f}GB is above {RUNTIME_MLX_LIMIT_ABORT_FRACTION * 100:.0f}% "
            f"of configured limit {_current_mlx_memory_limit_bytes / 1e9:.1f}GB."
        )


def _mlx_counter_bytes(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"MLX {name} memory counter must be a non-negative integer, got {value!r}")
    return value


def check_host_allocation_headroom(
    required_bytes: int,
    *,
    label: str = "",
    reserve_bytes: int = DEFAULT_SYSTEM_RESERVE_BYTES,
) -> SystemSnapshot:
    """Abort before large CPU-side allocations when system headroom is tight."""
    if not isinstance(required_bytes, int) or isinstance(required_bytes, bool) or required_bytes <= 0:
        raise MemoryGuardError(
            f"Memory guard [{label}]: required_bytes must be a positive integer, got {required_bytes!r}"
        )
    reserve_bytes = _system_reserve_bytes(reserve_bytes)
    snap = system_snapshot()
    required_with_reserve = required_bytes + reserve_bytes
    invalid_reasons = _invalid_system_snapshot_reasons(snap)
    if invalid_reasons:
        mlx_cleanup()
        raise RuntimeMemoryAbort(_invalid_system_snapshot_message(label, invalid_reasons))

    if sys.platform == "darwin" and snap.swap_files is None:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot read macOS swap file state before "
            f"allocating {required_bytes / 1e9:.1f}GB on host."
        )
    if sys.platform == "darwin" and snap.pressure is None:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: cannot read macOS memory pressure before "
            f"allocating {required_bytes / 1e9:.1f}GB on host."
        )
    if snap.free_bytes is None:
        if sys.platform == "darwin":
            mlx_cleanup()
            raise RuntimeMemoryAbort(
                f"Runtime memory abort [{label}]: cannot read free memory before "
                f"allocating {required_bytes / 1e9:.1f}GB on host."
            )
        return snap

    if snap.pressure is not None and snap.pressure > RUNTIME_PRESSURE_ABORT_THRESHOLD:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: pressure at {snap.pressure * 100:.0f}% "
            f"before allocating {required_bytes / 1e9:.1f}GB on host."
        )

    if snap.free_bytes < required_with_reserve:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: need {required_bytes / 1e9:.1f}GB "
            f"host allocation plus {reserve_bytes / 1e9:.1f}GB reserve, "
            f"only {snap.free_gb()}GB free."
        )
    return snap


def estimate_video_run_floor_bytes(
    *,
    width: int,
    height: int,
    frames: int,
    guidance: float,
) -> int:
    """Estimate a conservative non-weight memory floor for one video run.

    This deliberately excludes model weights because those vary by adapter and
    quantization. It captures the allocations driven directly by user shape:
    high-channel latents, CFG duplication, denoise temporaries, and decoded
    host frames.
    """
    for name, value in (("width", width), ("height", height), ("frames", frames)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MemoryGuardError(
                f"Memory guard [shape budget]: {name} must be a positive integer, got {value!r}"
            )
    try:
        guidance_value = float(guidance)
    except (TypeError, ValueError) as exc:
        raise MemoryGuardError(
            f"Memory guard [shape budget]: guidance must be a finite number, got {guidance!r}"
        ) from exc
    if not math.isfinite(guidance_value):
        raise MemoryGuardError(
            f"Memory guard [shape budget]: guidance must be a finite number, got {guidance!r}"
        )

    latent_h = max(1, (height + 7) // 8)
    latent_w = max(1, (width + 7) // 8)
    cfg_factor = 2 if guidance_value > 1.0 else 1
    latent_channels = 128
    bytes_per_float = 4
    latent_bytes = frames * latent_h * latent_w * latent_channels * bytes_per_float
    denoise_temporaries = latent_bytes * max(6, cfg_factor * 4)
    decoded_float = frames * height * width * 3 * bytes_per_float
    decoded_uint8 = frames * height * width * 3
    return int(denoise_temporaries + decoded_float + decoded_uint8)


def check_run_allocation_budget(
    *,
    width: int,
    height: int,
    frames: int,
    guidance: float,
    label: str = "",
) -> dict[str, object]:
    """Check shape-driven allocation floor before starting an MLX run."""
    required = estimate_video_run_floor_bytes(
        width=width,
        height=height,
        frames=frames,
        guidance=guidance,
    )
    snap = check_host_allocation_headroom(required, label=f"{label} shape budget")
    return {
        "shape_floor_gb": round(required / 1e9, 2),
        "free_gb": round(snap.free_bytes / 1e9, 2) if snap.free_bytes else None,
    }


def _max_prompt_chars() -> int:
    raw = os.environ.get("FASTGEN_MAX_PROMPT_CHARS")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_PROMPT_CHARS
    try:
        value = int(raw)
    except ValueError as exc:
        raise MemoryGuardError(f"FASTGEN_MAX_PROMPT_CHARS must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise MemoryGuardError(f"FASTGEN_MAX_PROMPT_CHARS must be a positive integer, got {raw!r}")
    if value > MAX_PROMPT_CHARS:
        raise MemoryGuardError(
            f"FASTGEN_MAX_PROMPT_CHARS must be no greater than {MAX_PROMPT_CHARS}, got {value}"
        )
    return value


def check_text_prompt_budget(
    *,
    prompt: str,
    negative_prompt: str | None = None,
    label: str = "",
) -> dict[str, object]:
    """Reject unexpectedly large text inputs before tokenizer/model allocation.

    Tokenizers can allocate proportionally to raw text length before model-side
    max token limits are applied. Keep this as a pre-tokenization gate so a
    malformed CLI or script input cannot start a large host allocation.
    """
    max_chars = _max_prompt_chars()
    negative = negative_prompt or ""
    prompt_chars = len(prompt)
    negative_chars = len(negative)
    largest = max(prompt_chars, negative_chars)
    if largest > max_chars:
        raise MemoryGuardError(
            f"Memory guard [{label}]: prompt text is {largest} chars, "
            f"max {max_chars}. Refusing to tokenize oversized text input."
        )

    # Python stores text wider than one byte when needed, and tokenizers often
    # copy buffers internally. This estimate is intentionally conservative.
    required = (prompt_chars + negative_chars) * 16
    check_host_allocation_headroom(required, label=f"{label} prompt text")
    return {
        "prompt_chars": prompt_chars,
        "negative_prompt_chars": negative_chars,
        "max_prompt_chars": max_chars,
    }


def check_token_sequence_budget(
    *,
    token_count: int,
    max_tokens: int,
    hidden_size: int,
    label: str = "",
) -> dict[str, object]:
    """Reject oversized token sequences before text-encoder MLX arrays."""
    for name, value in (
        ("token_count", token_count),
        ("max_tokens", max_tokens),
        ("hidden_size", hidden_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MemoryGuardError(
                f"Memory guard [{label}]: {name} must be a positive integer, got {value!r}"
            )
    if token_count > max_tokens:
        raise MemoryGuardError(
            f"Memory guard [{label}]: token sequence is {token_count} tokens, "
            f"max {max_tokens}. Refusing to encode oversized text input."
        )

    # Hidden states are at least token_count * hidden_size * fp32 bytes.
    # Use a multiplier for common temporary activations and projection buffers.
    required = token_count * hidden_size * 4 * 4
    check_host_allocation_headroom(required, label=f"{label} token hidden states")
    return {
        "token_count": token_count,
        "max_tokens": max_tokens,
        "hidden_size": hidden_size,
        "token_hidden_state_floor_gb": round(required / 1e9, 2),
    }


# ---------------------------------------------------------------------------
# Adaptive batch manager (Guard 2: Start small, grow or shrink)
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveBatchConfig:
    """Configuration for adaptive batch sizing."""
    # Initial probe size (small)
    initial_frames: int = ADAPTIVE_INITIAL_FRAMES
    initial_steps: int = ADAPTIVE_INITIAL_STEPS
    # Target size (what the user asked for)
    target_frames: int = 25
    target_steps: int = 16
    # Growth controls
    max_growth_factor: float = ADAPTIVE_MAX_GROWTH_FACTOR
    headroom_grow_threshold: float = ADAPTIVE_HEADROOM_THRESHOLD
    headroom_shrink_threshold: float = ADAPTIVE_SHRINK_THRESHOLD
    # Minimum sizes
    min_frames: int = 5
    min_steps: int = 4

    def __post_init__(self) -> None:
        for name in (
            "initial_frames",
            "initial_steps",
            "target_frames",
            "target_steps",
            "min_frames",
            "min_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MemoryGuardError(
                    f"Memory guard [adaptive batch]: {name} must be a positive integer, got {value!r}"
                )
        for name in ("headroom_grow_threshold", "headroom_shrink_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise MemoryGuardError(
                    f"Memory guard [adaptive batch]: {name} must be a finite number in [0, 1], got {value!r}"
                )
            if value < 0 or value > 1:
                raise MemoryGuardError(
                    f"Memory guard [adaptive batch]: {name} must be in [0, 1], got {value!r}"
                )
        if (
            not isinstance(self.max_growth_factor, (int, float))
            or isinstance(self.max_growth_factor, bool)
            or not math.isfinite(self.max_growth_factor)
            or self.max_growth_factor <= 1
        ):
            raise MemoryGuardError(
                "Memory guard [adaptive batch]: max_growth_factor must be a finite number greater than 1, "
                f"got {self.max_growth_factor!r}"
            )
        if self.min_frames > self.target_frames:
            raise MemoryGuardError(
                "Memory guard [adaptive batch]: min_frames cannot exceed target_frames "
                f"({self.min_frames} > {self.target_frames})"
            )
        if self.min_steps > self.target_steps:
            raise MemoryGuardError(
                "Memory guard [adaptive batch]: min_steps cannot exceed target_steps "
                f"({self.min_steps} > {self.target_steps})"
            )
        if self.initial_frames > self.target_frames:
            raise MemoryGuardError(
                "Memory guard [adaptive batch]: initial_frames cannot exceed target_frames "
                f"({self.initial_frames} > {self.target_frames})"
            )
        if self.initial_steps > self.target_steps:
            raise MemoryGuardError(
                "Memory guard [adaptive batch]: initial_steps cannot exceed target_steps "
                f"({self.initial_steps} > {self.target_steps})"
            )


@dataclass
class BatchDecision:
    """Result of adaptive batch sizing for one iteration."""
    frames: int
    steps: int
    phase: str  # "probe" | "grow" | "steady" | "shrink" | "final" | "max_reached"
    reason: str
    snapshot: SystemSnapshot | None = None


class AdaptiveBatchManager:
    """Manages adaptive batch sizing across probe → grow → final phases.

    Flow:
      1. Run a small probe (initial_frames, initial_steps)
      2. Check memory after probe
      3. If headroom is good → grow toward target (up to 2x)
      4. If headroom is tight → stay or shrink
      5. Eventually reach target → run "final" at target size
    """

    def __init__(self, config: AdaptiveBatchConfig) -> None:
        self.config = config
        self._iteration = 0
        self._current_frames = config.initial_frames
        self._current_steps = config.initial_steps
        self._reached_target = False
        self._last_snapshot: SystemSnapshot | None = None
        self._history: list[BatchDecision] = []

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def history(self) -> list[BatchDecision]:
        return list(self._history)

    def next_batch(self, snapshot: SystemSnapshot | None = None) -> BatchDecision:
        """Determine the next batch size based on memory state.

        Call this before each batch iteration. Pass the post-cleanup snapshot
        from the previous iteration (or None for the first probe).
        """
        self._iteration += 1
        self._last_snapshot = snapshot

        # First iteration: always probe small
        if self._iteration == 1:
            decision = BatchDecision(
                frames=self.config.initial_frames,
                steps=self.config.initial_steps,
                phase="probe",
                reason=f"Initial probe: {self.config.initial_frames} frames, {self.config.initial_steps} steps",
                snapshot=snapshot,
            )
            self._append_history(decision)
            self._current_frames = decision.frames
            self._current_steps = decision.steps
            return decision

        # If we already reached target, keep running at target
        if self._reached_target:
            decision = BatchDecision(
                frames=self.config.target_frames,
                steps=self.config.target_steps,
                phase="steady",
                reason="Running at target size",
                snapshot=snapshot,
            )
            self._append_history(decision)
            return decision

        # Determine headroom from snapshot
        headroom = self._compute_headroom(snapshot)

        # Decide: grow, shrink, or reach target
        if headroom is not None and headroom < self.config.headroom_shrink_threshold:
            # Tight: shrink but don't go below minimum
            new_frames = max(
                self.config.min_frames,
                int(self._current_frames / self.config.max_growth_factor),
            )
            new_steps = max(
                self.config.min_steps,
                int(self._current_steps / self.config.max_growth_factor),
            )
            decision = BatchDecision(
                frames=new_frames,
                steps=new_steps,
                phase="shrink",
                reason=f"Low headroom ({headroom * 100:.0f}%): shrinking from {self._current_frames}f/{self._current_steps}s",
                snapshot=snapshot,
            )
        elif self._current_frames >= self.config.target_frames and self._current_steps >= self.config.target_steps:
            # Reached target
            self._reached_target = True
            decision = BatchDecision(
                frames=self.config.target_frames,
                steps=self.config.target_steps,
                phase="final",
                reason=f"Reached target: {self.config.target_frames} frames, {self.config.target_steps} steps",
                snapshot=snapshot,
            )
        elif headroom is not None and headroom >= self.config.headroom_grow_threshold:
            # Good headroom: grow toward target
            new_frames = min(
                self.config.target_frames,
                int(self._current_frames * self.config.max_growth_factor),
            )
            new_steps = min(
                self.config.target_steps,
                int(self._current_steps * self.config.max_growth_factor),
            )
            # Cap at target
            new_frames = min(new_frames, self.config.target_frames)
            new_steps = min(new_steps, self.config.target_steps)

            phase = "grow"
            reason = f"Good headroom ({headroom * 100:.0f}%): growing to {new_frames}f/{new_steps}s"
            if new_frames >= self.config.target_frames and new_steps >= self.config.target_steps:
                self._reached_target = True
                phase = "final"
                reason = f"Reached target: {self.config.target_frames} frames, {self.config.target_steps} steps"

            decision = BatchDecision(
                frames=new_frames,
                steps=new_steps,
                phase=phase,
                reason=reason,
                snapshot=snapshot,
            )
        else:
            # Moderate headroom: stay at current size
            # Check if current already equals target
            if self._current_frames >= self.config.target_frames and self._current_steps >= self.config.target_steps:
                self._reached_target = True
                decision = BatchDecision(
                    frames=self.config.target_frames,
                    steps=self.config.target_steps,
                    phase="final",
                    reason=f"Reached target: {self.config.target_frames} frames, {self.config.target_steps} steps",
                    snapshot=snapshot,
                )
            else:
                # Try growing by smaller increments
                new_frames = min(
                    self.config.target_frames,
                    self._current_frames + max(1, int(self._current_frames * 0.5)),
                )
                new_steps = min(
                    self.config.target_steps,
                    self._current_steps + max(1, int(self._current_steps * 0.5)),
                )
                decision = BatchDecision(
                    frames=new_frames,
                    steps=new_steps,
                    phase="grow",
                    reason=f"Moderate headroom: cautiously growing to {new_frames}f/{new_steps}s",
                    snapshot=snapshot,
                )

        self._append_history(decision)
        self._current_frames = decision.frames
        self._current_steps = decision.steps
        return decision

    def _append_history(self, decision: BatchDecision) -> None:
        self._history.append(decision)
        overflow = len(self._history) - MAX_ADAPTIVE_BATCH_HISTORY
        if overflow > 0:
            del self._history[:overflow]

    def _compute_headroom(self, snapshot: SystemSnapshot | None) -> float | None:
        """Compute memory headroom fraction (0.0 = full, 1.0 = empty)."""
        if snapshot is None:
            return None
        if snapshot.free_fraction is not None:
            return snapshot.free_fraction
        # Fallback: use pressure as inverse
        if snapshot.pressure is not None:
            return 1.0 - snapshot.pressure
        return None


def adaptive_batch_config_from_run(
    target_frames: int,
    target_steps: int,
) -> AdaptiveBatchConfig:
    """Create an AdaptiveBatchConfig targeting the given frame/step counts."""
    for name, value in (("target_frames", target_frames), ("target_steps", target_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MemoryGuardError(
                f"Memory guard [adaptive batch]: {name} must be a positive integer, got {value!r}"
            )
    return AdaptiveBatchConfig(
        initial_frames=min(ADAPTIVE_INITIAL_FRAMES, target_frames),
        initial_steps=min(ADAPTIVE_INITIAL_STEPS, target_steps),
        target_frames=target_frames,
        target_steps=target_steps,
        min_frames=min(5, target_frames),
        min_steps=min(4, target_steps),
    )


# ---------------------------------------------------------------------------
# Run counter (process-level state)
# ---------------------------------------------------------------------------

_run_counter = 0


def run_counter() -> int:
    """Return the number of runs completed in this process."""
    return _run_counter


def increment_run_counter() -> int:
    """Increment and return the run counter."""
    global _run_counter
    _run_counter += 1
    return _run_counter


def reset_run_counter() -> None:
    """Reset process-local run counter. Intended for tests and fresh orchestration."""
    global _run_counter
    _run_counter = 0


def should_restart_process() -> bool:
    """Return True if this process has run enough times to risk instability.

    After MAX_CONSECUTIVE_RUNS, MLX Metal resource leaks may accumulate and
    trigger watchdog timeouts. The caller must stop MLX work in this process
    and let an orchestrator respawn a fresh process.
    """
    return _run_counter >= MAX_CONSECUTIVE_RUNS


# ---------------------------------------------------------------------------
# Convenience: full inter-run cycle
# ---------------------------------------------------------------------------

def inter_run_recovery(
    label: str = "",
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, object]:
    """Run full inter-run recovery: cleanup + memory check + cooldown.

    Returns combined status dict.
    Raises MemoryGuardError if memory is insufficient.
    """
    check_memory_guard(min_free_bytes=min_free_bytes, label=label)
    cleanup_status = mlx_cleanup()

    time.sleep(COOLDOWN_SECONDS)

    guard_status = check_memory_guard(min_free_bytes=min_free_bytes, label=label)
    limit_status = configure_mlx_resource_limits(label=label)
    guard_status.update(cleanup_status)
    guard_status.update(limit_status)
    guard_status["run_number"] = run_counter() + 1
    guard_status["should_restart"] = should_restart_process()

    return guard_status


def inter_run_system_recovery(
    label: str = "",
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, object]:
    """Run cleanup, cooldown, and system checks without importing MLX.

    Use this in callers that have not yet completed model-local asset/config
    preflight. Backend adapters should configure MLX allocator limits after
    they know the local model structure is safe to load.
    """
    check_memory_guard(min_free_bytes=min_free_bytes, label=label)
    cleanup_status = mlx_cleanup()

    time.sleep(COOLDOWN_SECONDS)

    guard_status = check_memory_guard(min_free_bytes=min_free_bytes, label=label)
    guard_status.update(cleanup_status)
    guard_status["run_number"] = run_counter() + 1
    guard_status["should_restart"] = should_restart_process()

    return guard_status
