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
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum free memory (bytes) required before starting a run.
# M4 Max with 128GB: 4GB headroom. Scale down for smaller configs.
DEFAULT_MIN_FREE_BYTES = 4 * 1024 ** 3  # 4 GB

# Maximum consecutive runs before forcing a process restart recommendation.
# After this many runs, accumulated Metal/MLX resource leaks may destabilize
# the system. The caller should exit and let the orchestrator respawn.
MAX_CONSECUTIVE_RUNS = 8

# Seconds to sleep between runs for thermal/Metal cooldown.
COOLDOWN_SECONDS = 5

# Maximum number of swap files before we refuse to start a run.
MAX_SWAP_FILES = 20

# Memory pressure fraction threshold (0.0-1.0) for runtime watchdog.
# Above this, the current step loop is aborted to prevent kernel panic.
RUNTIME_PRESSURE_ABORT_THRESHOLD = 0.92

# Free memory threshold for runtime watchdog.
# Below this, we abort the current run.
RUNTIME_MIN_FREE_BYTES = 2 * 1024 ** 3  # 2 GB

# Keep memory outside MLX so the OS remains responsive. MLX's default memory
# limit can be larger than the device working set, so profiler runs set an
# explicit process-local cap before touching model weights.
DEFAULT_SYSTEM_RESERVE_BYTES = 8 * 1024 ** 3  # 8 GB
DEFAULT_MLX_MEMORY_FRACTION = 0.80
DEFAULT_MLX_CACHE_LIMIT_BYTES = 1 * 1024 ** 3  # 1 GB

# Adaptive batch sizing defaults.
ADAPTIVE_INITIAL_FRAMES = 5
ADAPTIVE_INITIAL_STEPS = 4
ADAPTIVE_MAX_GROWTH_FACTOR = 2.0
ADAPTIVE_HEADROOM_THRESHOLD = 0.3  # grow if >30% memory free after probe
ADAPTIVE_SHRINK_THRESHOLD = 0.15   # shrink if <15% memory free after probe

logger = logging.getLogger("fastgen_profiler.mlx_guard")


# ---------------------------------------------------------------------------
# Memory introspection (macOS vm_stat)
# ---------------------------------------------------------------------------

def _vm_stat() -> dict[str, int]:
    """Parse macOS vm_stat output into a dict of page counts."""
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    pages: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
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
    page_size = 16384  # Apple Silicon ARM64 page size
    free = pages.get("Pages free", 0)
    inactive = pages.get("Pages inactive", 0)
    return (free + inactive) * page_size


def total_memory_bytes() -> int | None:
    """Return total physical memory in bytes, or None if unknown."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
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
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Output: "System-wide memory free percentage: 72%"
        for line in result.stdout.splitlines():
            if "percentage" in line.lower():
                pct_str = line.rsplit(":", 1)[-1].strip().rstrip("%")
                try:
                    used_pct = 100 - float(pct_str)
                    return used_pct / 100.0
                except ValueError:
                    continue
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


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
    free = free_memory_bytes()
    total = total_memory_bytes()
    pressure = memory_pressure_fraction()
    swap = swap_file_count()

    free_frac = None
    if free is not None and total is not None and total > 0:
        free_frac = free / total

    return SystemSnapshot(
        free_bytes=free,
        total_bytes=total,
        pressure=pressure,
        swap_files=swap,
        free_fraction=free_frac,
    )


# ---------------------------------------------------------------------------
# MLX cleanup
# ---------------------------------------------------------------------------

def mlx_cleanup() -> dict[str, object]:
    """Aggressive MLX + Python cleanup between benchmark runs.

    Returns a status dict with memory info for logging.
    """
    before_free = free_memory_bytes()

    # Force Python garbage collection (multiple passes)
    for _ in range(3):
        gc.collect()

    # Clear MLX Metal cache
    try:
        import mlx.core as mx
        mx.clear_cache()
        # Synchronize to flush pending GPU commands
        mx.eval(mx.array(0))
    except Exception:
        pass

    # Another GC pass after MLX cleanup
    gc.collect()

    after_free = free_memory_bytes()

    return {
        "free_before_gb": round(before_free / 1e9, 2) if before_free else None,
        "free_after_gb": round(after_free / 1e9, 2) if after_free else None,
        "freed_gb": round((after_free - before_free) / 1e9, 2)
        if (before_free is not None and after_free is not None)
        else None,
    }


def _env_gb_to_bytes(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        gb = float(value)
    except ValueError as exc:
        raise MemoryGuardError(f"{name} must be a number of GB, got {value!r}") from exc
    if gb <= 0:
        raise MemoryGuardError(f"{name} must be greater than zero")
    return int(gb * 1024 ** 3)


def _default_mlx_memory_limit(total_bytes: int | None) -> int | None:
    if total_bytes is None:
        return None
    reserved_limit = max(0, total_bytes - DEFAULT_SYSTEM_RESERVE_BYTES)
    fraction_limit = int(total_bytes * DEFAULT_MLX_MEMORY_FRACTION)
    return max(1 * 1024 ** 3, min(reserved_limit, fraction_limit))


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
    memory_limit = _env_gb_to_bytes("FASTGEN_MLX_MEMORY_LIMIT_GB")
    if memory_limit is None:
        memory_limit = _default_mlx_memory_limit(snap.total_bytes)
    cache_limit = _env_gb_to_bytes("FASTGEN_MLX_CACHE_LIMIT_GB")
    if cache_limit is None:
        cache_limit = DEFAULT_MLX_CACHE_LIMIT_BYTES
    wired_limit = _env_gb_to_bytes("FASTGEN_MLX_WIRED_LIMIT_GB")
    if wired_limit is None and memory_limit is not None and sys.platform == "darwin":
        wired_limit = memory_limit

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

    try:
        import mlx.core as mx
    except Exception as exc:
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
    except Exception as exc:
        raise MemoryGuardError(
            f"Memory guard [{label}]: failed to set MLX memory limits: {exc}"
        ) from exc

    return status


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

    status: dict[str, object] = {
        "label": label,
        "free_gb": round(snap.free_bytes / 1e9, 2) if snap.free_bytes else None,
        "pressure": round(snap.pressure, 3) if snap.pressure is not None else None,
        "swap_files": snap.swap_files,
        "min_free_gb": round(min_free_bytes / 1e9, 2),
    }

    if sys.platform == "darwin" and snap.free_bytes is None and snap.pressure is None:
        raise MemoryGuardError(
            f"Memory guard [{label}]: cannot read vm_stat or memory_pressure. "
            "Refusing to start an MLX/Metal run without memory telemetry."
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
    snap = system_snapshot()

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


def check_host_allocation_headroom(
    required_bytes: int,
    *,
    label: str = "",
    reserve_bytes: int = DEFAULT_SYSTEM_RESERVE_BYTES,
) -> SystemSnapshot:
    """Abort before large CPU-side allocations when system headroom is tight."""
    snap = system_snapshot()
    required_with_reserve = required_bytes + reserve_bytes
    if snap.free_bytes is None:
        if sys.platform == "darwin":
            mlx_cleanup()
            raise RuntimeMemoryAbort(
                f"Runtime memory abort [{label}]: cannot read free memory before "
                f"allocating {required_bytes / 1e9:.1f}GB on host."
            )
        return snap

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
    latent_h = max(1, height // 8)
    latent_w = max(1, width // 8)
    cfg_factor = 2 if guidance > 1.0 else 1
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
            self._history.append(decision)
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
            self._history.append(decision)
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

        self._history.append(decision)
        self._current_frames = decision.frames
        self._current_steps = decision.steps
        return decision

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

    After MAX_CONSECUTIVE_RUNS, MLX Metal resource leaks may accumulate
    and trigger watchdog timeouts. The caller should exit with a special
    code and let the orchestrator respawn a fresh process.
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
    cleanup_status = mlx_cleanup()

    import time
    time.sleep(COOLDOWN_SECONDS)

    guard_status = check_memory_guard(min_free_bytes=min_free_bytes, label=label)
    limit_status = configure_mlx_resource_limits(label=label)
    guard_status.update(cleanup_status)
    guard_status.update(limit_status)
    guard_status["run_number"] = run_counter() + 1
    guard_status["should_restart"] = should_restart_process()

    return guard_status
