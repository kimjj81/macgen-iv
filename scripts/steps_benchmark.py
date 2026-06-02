#!/usr/bin/env python3
"""Steps benchmark: measure denoise + VAE decode time across step counts for LTX2.3.

Includes MLX memory guard to prevent kernel panic from repeated GPU-heavy runs.
Defaults to a single minimal step; set FASTGEN_STEPS_VALUES=24,32,36,40,44
explicitly for the full historical sweep.
Real model execution requires FASTGEN_STEPS_ALLOW_HEAVY=1.
Multiple heavy child runs also require FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY=1.
"""

import sys
import time
import json
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import Any, Iterable

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastgen_profiler.mlx_guard import (
    MemoryGuardError,
    RuntimeMemoryAbort,
    check_memory_guard,
    check_host_allocation_headroom,
    check_run_allocation_budget,
    check_runtime_memory,
    check_text_prompt_budget,
    increment_run_counter,
    mlx_cleanup,
    run_counter,
    should_restart_process,
    DEFAULT_MAX_PROMPT_CHARS,
    MAX_PROMPT_CHARS as HARD_MAX_PROMPT_CHARS,
    MAX_CONSECUTIVE_RUNS,
    COOLDOWN_SECONDS,
)


PNG_FRAME_ALLOCATION_MULTIPLIER = 4
VIDEO_STATS_ALLOCATION_MULTIPLIER = 4
MAX_DIMENSION = 4096
MAX_FRAMES = 257
MAX_FPS = 240
MAX_STEPS = 512
MAX_STEP_VALUES = 16
MAX_CHILD_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_SEED = 2**32 - 1


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return _positive_int_value(name, raw)


def _env_capped_positive_int(name: str, default: int, max_value: int) -> int:
    value = _env_positive_int(name, default)
    if value > max_value:
        raise MemoryGuardError(f"{name} must be no greater than {max_value}, got {value}")
    return value


def _positive_int_value(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise MemoryGuardError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise MemoryGuardError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _env_step_values(name: str, default: str) -> list[int]:
    raw_values = [value.strip() for value in os.environ.get(name, default).split(",") if value.strip()]
    if len(raw_values) > MAX_STEP_VALUES:
        raise MemoryGuardError(f"{name} may contain at most {MAX_STEP_VALUES} values")
    return [_capped_positive_int_value(name, value, MAX_STEPS) for value in raw_values]


def _env_bounded_text(name: str, default: str, *, max_chars: int) -> str:
    value = os.environ.get(name, default)
    if len(value) > max_chars:
        raise MemoryGuardError(f"{name} must be no longer than {max_chars} chars")
    return value


def _capped_positive_int_value(name: str, raw: str, max_value: int) -> int:
    value = _positive_int_value(name, raw)
    if value > max_value:
        raise MemoryGuardError(f"{name} must be no greater than {max_value}, got {value}")
    return value


def _eval_mlx(mx, target, *, label: str) -> None:
    try:
        mx.eval(target)
    except Exception as exc:
        mlx_cleanup()
        raise RuntimeMemoryAbort(
            f"Runtime memory abort [{label}]: MLX eval failed; "
            "aborting because Metal runtime state may be unsafe."
        ) from exc


MODEL_PATH = Path(os.environ.get(
    "FASTGEN_STEPS_MODEL_PATH",
    str(REPO_ROOT / "artifacts" / "models" / "LTX-2.3-distilled-mlx"),
))
OUTPUT_BASE = Path(os.environ.get(
    "FASTGEN_STEPS_OUTPUT_BASE",
    str(REPO_ROOT / "artifacts" / "steps_benchmark"),
))
MAX_PROMPT_CHARS = _env_capped_positive_int(
    "FASTGEN_MAX_PROMPT_CHARS",
    DEFAULT_MAX_PROMPT_CHARS,
    HARD_MAX_PROMPT_CHARS,
)
PROMPT = _env_bounded_text(
    "FASTGEN_STEPS_PROMPT",
    "A golden retriever running through a sunlit meadow, cinematic, slow motion",
    max_chars=MAX_PROMPT_CHARS,
)
NEGATIVE_PROMPT = _env_bounded_text(
    "FASTGEN_STEPS_NEGATIVE_PROMPT",
    "",
    max_chars=MAX_PROMPT_CHARS,
)
WIDTH = _env_capped_positive_int("FASTGEN_STEPS_WIDTH", 512, MAX_DIMENSION)
HEIGHT = _env_capped_positive_int("FASTGEN_STEPS_HEIGHT", 512, MAX_DIMENSION)
FRAMES = _env_capped_positive_int("FASTGEN_STEPS_FRAMES", 9, MAX_FRAMES)
FPS = _env_capped_positive_int("FASTGEN_STEPS_FPS", 24, MAX_FPS)
SEED = _env_capped_positive_int("FASTGEN_STEPS_SEED", 42, MAX_SEED)
STEP_VALUES = _env_step_values("FASTGEN_STEPS_VALUES", "1")
ALLOW_HEAVY = os.environ.get("FASTGEN_STEPS_ALLOW_HEAVY") == "1"
ALLOW_MULTIPLE_HEAVY = os.environ.get("FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY") == "1"
RESULTS_JSONL = OUTPUT_BASE / "results.jsonl"
CHILD_MODE_ENV = "FASTGEN_STEPS_CHILD"
CHILD_STEP_ENV = "FASTGEN_STEPS_CHILD_STEP"
CHILD_RESULT_ENV = "FASTGEN_STEPS_CHILD_RESULT"
CHILD_TIMEOUT_SECONDS = _env_capped_positive_int(
    "FASTGEN_STEPS_CHILD_TIMEOUT_SECONDS",
    60 * 60,
    MAX_CHILD_TIMEOUT_SECONDS,
)
CHILD_IO_MAX_BYTES = 1024 * 1024
CHILD_LOG_TAIL_BYTES = _env_capped_positive_int(
    "FASTGEN_STEPS_CHILD_LOG_TAIL_BYTES",
    64 * 1024,
    CHILD_IO_MAX_BYTES,
)
CHILD_RESULT_MAX_BYTES = _env_capped_positive_int(
    "FASTGEN_STEPS_CHILD_RESULT_MAX_BYTES",
    CHILD_IO_MAX_BYTES,
    CHILD_IO_MAX_BYTES,
)
STEPS_RESULT_TEXT_FIELD_MAX_CHARS = 2_048
STEPS_RESULT_COLLECTION_MAX_ITEMS = 256

# Exit code to signal orchestrator that the process should be restarted.
EXIT_RESTART = 10


def run_single(steps: int):
    import gc

    label = f"steps_{steps}"
    if not ALLOW_HEAVY:
        return {
            "steps": steps,
            "skipped": True,
            "error": (
                "skipped: real MLX benchmark requires FASTGEN_STEPS_ALLOW_HEAVY=1 "
                "after reviewing memory limits and model paths"
            ),
        }

    # Pre-run memory guard
    guard = check_memory_guard(label=label)
    budget = check_run_allocation_budget(
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        guidance=1.0,
        label=label,
    )
    check_text_prompt_budget(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        label=label,
    )
    if importlib.util.find_spec("mlx_video") is None:
        raise MemoryGuardError(
            "dependency unavailable before MLX import: mlx_video is required "
            "for the LTX2.3 steps benchmark"
        )
    run_num = run_counter() + 1
    print(f"  [guard] run #{run_num} free={guard.get('free_gb', '?')}GB "
          f"pressure={guard.get('pressure', '?')} "
          f"shape_floor={budget.get('shape_floor_gb', '?')}GB")

    from fastgen_profiler.backends.ltx23_mlx_adapter import create_ltx23_pipeline

    step_dir = OUTPUT_BASE / f"steps_{steps}"
    step_dir.mkdir(parents=True, exist_ok=True)

    result = {"steps": steps, "width": WIDTH, "height": HEIGHT, "frames": FRAMES}

    pipe = create_ltx23_pipeline(
        model_path=MODEL_PATH,
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        steps=steps,
        fps=FPS,
        guidance=1.0,
        save_video=False,
    )

    # Load model
    t0 = time.perf_counter()
    pipe.load_model()
    t1 = time.perf_counter()
    result["load_model_s"] = round(t1 - t0, 2)
    print(f"  load_model: {result['load_model_s']}s")

    import mlx.core as mx

    # Text encode
    t0 = time.perf_counter()
    prepared = pipe.prepare_prompt(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT)
    context = pipe.encode_text(prepared)
    _eval_mlx(mx, context, label=f"{label} text_encode")
    t1 = time.perf_counter()
    result["text_encode_s"] = round(t1 - t0, 2)
    print(f"  text_encode: {result['text_encode_s']}s")

    # Init latents
    latents = pipe.init_latents(seed=SEED, width=WIDTH, height=HEIGHT, frames=FRAMES)
    _eval_mlx(mx, latents, label=f"{label} latent_init")

    # Denoise
    denoise_times = []
    for i in range(steps):
        check_runtime_memory(label=f"{label} denoise {i+1}/{steps} before")
        _eval_mlx(mx, latents, label=f"{label} denoise {i+1}/{steps} input")
        t0 = time.perf_counter()
        latents = pipe.denoise_step(
            latents, step_index=i, steps=steps, guidance=1.0, cache="none"
        )
        _eval_mlx(mx, latents, label=f"{label} denoise {i+1}/{steps} output")
        check_runtime_memory(label=f"{label} denoise {i+1}/{steps} after")
        t1 = time.perf_counter()
        step_s = t1 - t0
        denoise_times.append(step_s)
        if (i + 1) % 8 == 0 or i == 0:
            print(f"  denoise {i+1}/{steps} ({step_s:.2f}s)")

    result["denoise_total_s"] = round(sum(denoise_times), 2)
    result["denoise_avg_s"] = round(sum(denoise_times) / len(denoise_times), 2)
    result["denoise_min_s"] = round(min(denoise_times), 2)
    result["denoise_max_s"] = round(max(denoise_times), 2)

    # VAE decode
    _eval_mlx(mx, latents, label=f"{label} vae_decode input")
    check_runtime_memory(label=f"{label} vae_decode before")
    t0 = time.perf_counter()
    video = pipe.decode(latents)
    _check_decoded_video_shape(video, label=label)
    check_runtime_memory(label=f"{label} vae_decode after")
    t1 = time.perf_counter()
    result["vae_decode_s"] = round(t1 - t0, 2)

    # Quality metrics
    check_host_allocation_headroom(
        _video_frame_budget_bytes(video, multiplier=VIDEO_STATS_ALLOCATION_MULTIPLIER),
        label=f"{label} quality metrics",
    )
    result["video_shape"] = [FRAMES, HEIGHT, WIDTH, 3]
    result["pixel_min"] = int(video.min())
    result["pixel_max"] = int(video.max())
    result["pixel_mean"] = round(float(video.mean()), 2)
    result["pixel_std"] = round(float(video.std()), 2)

    # Save frames as PNG
    check_host_allocation_headroom(
        _video_frame_budget_bytes(video, multiplier=PNG_FRAME_ALLOCATION_MULTIPLIER),
        label=f"{label} png frames",
    )
    from PIL import Image
    img = None
    for idx in range(FRAMES):
        img = Image.fromarray(video[idx])
        img.save(str(step_dir / f"frame_{idx:03d}.png"))

    total = result["denoise_total_s"] + result["vae_decode_s"]
    print(f"  DONE: denoise={result['denoise_total_s']}s  vae={result['vae_decode_s']}s  "
          f"total={round(total,1)}s  pixels=[{result['pixel_min']},{result['pixel_max']}] "
          f"mean={result['pixel_mean']}")

    # Aggressive cleanup to prevent resource accumulation
    del pipe, latents, video, context, prepared
    if img is not None:
        del img
    gc.collect()
    cleanup = mlx_cleanup()
    print(f"  [guard] cleanup: freed={cleanup.get('freed_gb', '?')}GB "
          f"now_free={cleanup.get('free_after_gb', '?')}GB")
    increment_run_counter()

    return result


def _check_decoded_video_shape(video, *, label: str) -> None:
    actual = _bounded_shape_tuple(video, expected_rank=4, label=label)
    expected = (FRAMES, HEIGHT, WIDTH, 3)
    if actual != expected:
        raise RuntimeMemoryAbort(
            f"decoded benchmark video must have shape {expected}, got {actual} for {label}"
        )


def _bounded_shape_tuple(value: Any, *, expected_rank: int, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise RuntimeMemoryAbort(
            f"decoded benchmark video has no shape for {label}; refusing unbounded shape inspection"
        )
    dims: list[int] = []
    try:
        iterator = iter(shape)
    except TypeError as exc:
        raise RuntimeMemoryAbort(
            f"decoded benchmark video shape is not iterable for {label}; refusing unbounded shape inspection"
        ) from exc
    for dim in iterator:
        if len(dims) >= expected_rank:
            raise RuntimeMemoryAbort(
                f"decoded benchmark video shape rank exceeds {expected_rank} for {label}; "
                "refusing unbounded shape inspection"
            )
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise RuntimeMemoryAbort(
                "decoded benchmark video shape contains non-integer dimension "
                f"{_shape_dim_text(dim)} for {label}"
            )
        dims.append(dim)
    if len(dims) != expected_rank:
        raise RuntimeMemoryAbort(
            f"decoded benchmark video shape rank is {len(dims)}, expected {expected_rank} for {label}; "
            "refusing under-bounded shape inspection"
        )
    return tuple(dims)


def _shape_dim_text(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return str(value)
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _video_frame_budget_bytes(video: Any, *, multiplier: int) -> int:
    shape_floor = FRAMES * HEIGHT * WIDTH * 3 * 4
    reported_nbytes = getattr(video, "nbytes", 0)
    if not isinstance(reported_nbytes, int) or isinstance(reported_nbytes, bool) or reported_nbytes < 0:
        reported_nbytes = 0
    return max(reported_nbytes, shape_floor) * multiplier


def run_child() -> int:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    try:
        result_path = _child_result_path_from_env()
        steps = _positive_int_value(CHILD_STEP_ENV, os.environ[CHILD_STEP_ENV])
    except (KeyError, MemoryGuardError) as exc:
        error = _safe_exception_text(exc)
        print(f"  [guard] child BLOCKED: invalid child environment: {error}")
        return 1
    try:
        result = run_single(steps)
    except MemoryGuardError as e:
        error = _safe_exception_text(e)
        print(f"  [guard] steps={steps} BLOCKED: {error}")
        result = {"steps": steps, "error": error, "skipped": True, "guard_blocked": True}
    except RuntimeMemoryAbort as e:
        error = _safe_exception_text(e)
        print(f"  [guard] steps={steps} ABORTED: {error}")
        result = {"steps": steps, "error": error, "aborted": True}
        result["cleanup"] = mlx_cleanup()
    except Exception as e:
        error = _safe_exception_text(e)
        print(f"  steps={steps} FAILED: {error}")
        print("  [guard] traceback suppressed to avoid unbounded exception formatting")
        result = {"steps": steps, "error": error}
        result["cleanup"] = mlx_cleanup()

    temp_result_path = result_path.with_suffix(result_path.suffix + ".tmp")
    temp_result_path.write_text(json.dumps(_bound_steps_result(result)) + "\n", encoding="utf-8")
    temp_result_path.replace(result_path)
    return 0


def _child_result_path_from_env() -> Path:
    try:
        raw = os.environ[CHILD_RESULT_ENV]
    except KeyError as exc:
        raise MemoryGuardError(f"{CHILD_RESULT_ENV} is required in child mode") from exc
    result_path = Path(raw)
    output_base = OUTPUT_BASE.resolve()
    resolved = result_path.resolve()
    if resolved.parent != output_base:
        raise MemoryGuardError(
            f"{CHILD_RESULT_ENV} must point inside {output_base}, got {resolved}"
        )
    return resolved


def run_step_in_child(steps: int) -> dict:
    child_result = OUTPUT_BASE / f"steps_{steps}.child.json"
    child_log = OUTPUT_BASE / f"steps_{steps}.child.log"
    child_result.unlink(missing_ok=True)
    child_log.unlink(missing_ok=True)
    env = dict(os.environ)
    env[CHILD_MODE_ENV] = "1"
    env[CHILD_STEP_ENV] = str(steps)
    env[CHILD_RESULT_ENV] = str(child_result)
    try:
        with child_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=CHILD_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        _print_child_log_tail(child_log)
        record = {
            "steps": steps,
            "error": f"child process timed out after {CHILD_TIMEOUT_SECONDS}s",
            "aborted": True,
            "log_path": str(child_log),
        }
        _cleanup_child_artifacts(child_result, child_log)
        return record
    _print_child_log_tail(child_log)
    if child_result.exists():
        result_size = child_result.stat().st_size
        if result_size > CHILD_RESULT_MAX_BYTES:
            record = {
                "steps": steps,
                "error": (
                    f"child result file is {result_size} bytes, "
                    f"exceeds limit {CHILD_RESULT_MAX_BYTES} bytes"
                ),
                "aborted": True,
                "log_path": str(child_log),
            }
            _cleanup_child_artifacts(child_result, child_log)
            return record
        try:
            record = _read_last_child_result(child_result)
        except json.JSONDecodeError as exc:
            record = {
                "steps": steps,
                "error": f"child result file is not valid JSONL: {exc}",
                "aborted": True,
                "log_path": str(child_log),
            }
            _cleanup_child_artifacts(child_result, child_log)
            return record
        if record is None:
            record = {
                "steps": steps,
                "error": "child result file did not contain a result record",
                "aborted": True,
                "log_path": str(child_log),
            }
            _cleanup_child_artifacts(child_result, child_log)
            return record
        record = _bound_steps_result(record)
        record.setdefault("log_path", str(child_log))
        return record
    record = {
        "steps": steps,
        "error": f"child process exited {result.returncode} without a result record",
        "aborted": True,
        "log_path": str(child_log),
    }
    _cleanup_child_artifacts(child_result, child_log)
    return record


def _cleanup_child_artifacts(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _print_child_log_tail(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= 0:
        return
    with path.open("rb") as handle:
        if size > CHILD_LOG_TAIL_BYTES:
            handle.seek(-CHILD_LOG_TAIL_BYTES, os.SEEK_END)
            print(f"\n[guard] child log tail from {path} ({CHILD_LOG_TAIL_BYTES} bytes):")
        data = handle.read()
    print(data.decode("utf-8", errors="replace"), end="")


def _read_last_child_result(path: Path) -> dict[str, Any] | None:
    last_record = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            last_record = json.loads(stripped)
    return last_record


def _bound_steps_result(value: Any) -> Any:
    if isinstance(value, str):
        return _bound_steps_text(value)
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= STEPS_RESULT_COLLECTION_MAX_ITEMS:
                bounded["__truncated_items__"] = True
                break
            bounded[_safe_steps_text(key)] = _bound_steps_result(item)
        return bounded
    if isinstance(value, list):
        return _bound_steps_sequence(value)
    if isinstance(value, tuple):
        return _bound_steps_sequence(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_steps_text(value)


def _safe_steps_text(value: Any) -> str:
    if isinstance(value, str):
        return _bound_steps_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return _bound_steps_text(str(value))
    value_type = type(value)
    return _bound_steps_text(f"<{value_type.__module__}.{value_type.__qualname__}>")


def _safe_exception_text(exc: BaseException) -> str:
    exc_type = type(exc)
    parts = []
    for index, arg in enumerate(exc.args):
        if index >= STEPS_RESULT_COLLECTION_MAX_ITEMS:
            parts.append("...<truncated>")
            break
        parts.append(_safe_steps_text(arg))
    if not parts:
        return _bound_steps_text(f"<{exc_type.__module__}.{exc_type.__qualname__}>")
    if len(parts) == 1:
        return parts[0]
    return _bound_steps_text(f"{exc_type.__module__}.{exc_type.__qualname__}: {', '.join(parts)}")


def _bound_steps_text(value: str) -> str:
    if len(value) <= STEPS_RESULT_TEXT_FIELD_MAX_CHARS:
        return value
    suffix = "...<truncated>"
    return value[: STEPS_RESULT_TEXT_FIELD_MAX_CHARS - len(suffix)] + suffix


def _bound_steps_sequence(value: Iterable[Any]) -> list[Any]:
    bounded = []
    for index, item in enumerate(value):
        if index >= STEPS_RESULT_COLLECTION_MAX_ITEMS:
            bounded.append({"__truncated_items__": True})
            break
        bounded.append(_bound_steps_result(item))
    return bounded


def _write_steps_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    max_records: int = MAX_STEP_VALUES,
) -> None:
    if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records <= 0:
        raise MemoryGuardError(f"steps result record limit must be a positive integer, got {max_records!r}")
    with open(path, "w", encoding="utf-8") as f:
        for index, record in enumerate(records):
            if index >= max_records:
                raise MemoryGuardError(
                    f"steps result record limit exceeded: more than {max_records} records"
                )
            f.write(json.dumps(_bound_steps_result(record)) + "\n")


def parent_inter_child_recovery(label: str) -> dict[str, object]:
    """System-only recovery between child MLX processes.

    The parent process must not import MLX here; each child owns MLX runtime
    setup and teardown.
    """
    status = check_memory_guard(label=label)
    time.sleep(COOLDOWN_SECONDS)
    return status


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    if os.environ.get(CHILD_MODE_ENV) == "1":
        return run_child()

    if ALLOW_HEAVY and len(STEP_VALUES) > 1 and not ALLOW_MULTIPLE_HEAVY:
        error = (
            "skipped: multiple heavy MLX child runs require "
            "FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY=1 after reviewing cooldown, "
            "memory reserve, and model paths"
        )
        _write_steps_jsonl(
            RESULTS_JSONL,
            ({"steps": steps, "error": error, "skipped": True} for steps in STEP_VALUES),
        )
        print(f"  [guard] {error}")
        print(f"Results: {RESULTS_JSONL}")
        return 1

    # Clean previous partial results
    for steps in STEP_VALUES:
        d = OUTPUT_BASE / f"steps_{steps}"
        if d.exists():
            import shutil
            shutil.rmtree(d)

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    all_results = []
    completed_child_runs = 0
    for steps in STEP_VALUES:
        print(f"\n{'='*60}")
        process_run = completed_child_runs + 1 if ALLOW_HEAVY else run_counter() + 1
        print(f"Running steps={steps}  (process run #{process_run}/{MAX_CONSECUTIVE_RUNS} max)")
        print(f"{'='*60}")

        # Inter-run recovery: cleanup + cooldown + memory check.
        # Heavy mode runs MLX in child processes, so the parent uses a
        # system-only check/cooldown and avoids importing MLX itself.
        if ALLOW_HEAVY and completed_child_runs > 0:
            try:
                recovery = parent_inter_child_recovery(label=f"pre-steps_{steps}")
                print(f"  [guard] parent recovery: free={recovery.get('free_gb', '?')}GB")
            except MemoryGuardError as e:
                error = _safe_exception_text(e)
                print(f"  [guard] SKIPPING steps={steps}: {error}")
                all_results.append({"steps": steps, "error": error, "skipped": True})
                continue
        elif run_counter() > 0:
            try:
                recovery = parent_inter_child_recovery(label=f"pre-steps_{steps}")
                print(f"  [guard] recovery: free={recovery.get('free_gb', '?')}GB")
            except MemoryGuardError as e:
                error = _safe_exception_text(e)
                print(f"  [guard] SKIPPING steps={steps}: {error}")
                all_results.append({"steps": steps, "error": error, "skipped": True})
                continue

        # Check if process should restart to avoid Metal leak accumulation
        if not ALLOW_HEAVY and should_restart_process():
            print(f"  [guard] Process has run {run_counter()} times. "
                  f"Recommending restart to prevent Metal resource leak.")
            # Save what we have so far
            _write_steps_jsonl(RESULTS_JSONL, all_results)
            print(f"  [guard] Partial results saved to {RESULTS_JSONL}")
            print(f"  [guard] Remaining steps: {[s for s in STEP_VALUES if not any(r.get('steps') == s and 'error' not in r for r in all_results)]}")
            sys.exit(EXIT_RESTART)

        try:
            r = run_step_in_child(steps) if ALLOW_HEAVY else run_single(steps)
            all_results.append(r)
            if ALLOW_HEAVY and not r.get("skipped") and not r.get("aborted") and "error" not in r:
                completed_child_runs += 1
            if r.get("skipped"):
                print(f"  [guard] steps={steps} SKIPPED: {r['error']}")
            if ALLOW_HEAVY and (
                r.get("aborted")
                or r.get("guard_blocked")
                or ("error" in r and not r.get("skipped"))
            ):
                print(
                    "  [guard] stopping remaining heavy steps after child "
                    "abort/error to avoid repeated MLX/Metal initialization."
                )
                break
        except MemoryGuardError as e:
            error = _safe_exception_text(e)
            print(f"  [guard] steps={steps} BLOCKED: {error}")
            all_results.append({"steps": steps, "error": error, "skipped": True})
        except RuntimeMemoryAbort as e:
            error = _safe_exception_text(e)
            print(f"  [guard] steps={steps} ABORTED: {error}")
            all_results.append({"steps": steps, "error": error, "aborted": True})
            increment_run_counter()
            mlx_cleanup()
        except Exception as e:
            error = _safe_exception_text(e)
            print(f"  steps={steps} FAILED: {error}")
            print("  [guard] traceback suppressed to avoid unbounded exception formatting")
            all_results.append({"steps": steps, "error": error})
            # Treat unknown failures as a consumed MLX process slot: the error
            # may have happened after Metal state was initialized.
            increment_run_counter()
            mlx_cleanup()

    # Write JSONL
    _write_steps_jsonl(RESULTS_JSONL, all_results)

    print(f"\n\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {RESULTS_JSONL}")
    exit_code = 0
    for r in all_results:
        if "error" in r:
            if r.get("skipped"):
                status = "SKIPPED"
            elif r.get("aborted"):
                status = "ABORTED"
            else:
                status = "ERROR"
            if ALLOW_HEAVY or r.get("aborted") or not r.get("skipped"):
                exit_code = 1
            print(f"  steps={r['steps']}: {status} - {r['error']}")
        else:
            total = round(r['denoise_total_s'] + r['vae_decode_s'], 1)
            print(f"  steps={r['steps']}: denoise={r['denoise_total_s']}s  vae={r['vae_decode_s']}s  "
                  f"total={total}s  pixels=[{r['pixel_min']},{r['pixel_max']}] mean={r['pixel_mean']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
