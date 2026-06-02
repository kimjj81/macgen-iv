"""Command line interface for fastgen-profile."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Annotated
import uuid

logger = logging.getLogger("fastgen_profiler.cli")

import click
import typer

from .backends import create_backend
from .metrics import (
    MAX_RUN_DIMENSION,
    MAX_RUN_FPS,
    MAX_RUN_FRAMES,
    MAX_RUN_STEPS,
    MeasurementRecord,
    RunConfig,
    append_jsonl,
    make_record,
    machine_metadata,
    new_run_id,
    read_jsonl,
    utc_timestamp,
)
from .models import (
    IMPORT_SOURCES,
    ModelCandidate,
    candidate_to_dict,
    discover_generation_model_dirs,
    direct_model_candidate,
    discover_import_dirs,
    discover_models,
    model_dirs_from_sources,
    replace_model_dirs_in_env,
    resolve_model_candidate,
)
from .profiler import Profiler
from .reports.markdown import MAX_REPORT_RECORDS, render_markdown_report


PRESET_CHOICES = (
    "smoke",
    "small-baseline",
    "quality-threshold",
    "stress",
    "cache-experiment",
    "compile-experiment",
)
PROFILE_PRESETS = (
    "smoke",
    "small-baseline",
    "quality-threshold",
    "cache-experiment",
    "compile-experiment",
    "stress",
)
MODEL_CHOICES = ("wan2.2", "ltx2.3")
BACKEND_CHOICES = ("mlx", "stub")
QUANT_CHOICES = ("none", "q8", "q8p", "q4")
CACHE_CHOICES = ("none", "prompt", "feature", "all")
COMPILE_CHOICES = ("off", "on")

DEFAULT_GUIDANCE = 3.5
DEFAULT_FPS = 12
MAX_SUMMARY_FIELD_CHARS = 256
MAX_CLI_DIMENSION = MAX_RUN_DIMENSION
MAX_CLI_FRAMES = MAX_RUN_FRAMES
MAX_CLI_STEPS = MAX_RUN_STEPS
MAX_CLI_FPS = MAX_RUN_FPS
ALLOW_PARENT_MLX_ENV = "FASTGEN_CLI_ALLOW_PARENT_MLX"

app = typer.Typer(
    help="Profile MLX video generation experiments and write benchmark JSONL.",
    invoke_without_command=True,
    no_args_is_help=False,
)
models_app = typer.Typer(
    help="Inspect local model directories.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(models_app, name="models")


@dataclass(frozen=True, slots=True)
class PresetRun:
    width: int
    height: int
    frames: int
    steps: int
    guidance: float
    quant: str
    cache: str
    compile: str
    save_video: bool


@dataclass(frozen=True, slots=True)
class ProfileRunSpec:
    preset: str
    variant_label: str
    run: PresetRun


@dataclass(frozen=True, slots=True)
class RunOptions:
    preset: str | None
    model: str
    backend: str
    model_dir: list[Path]
    model_path: Path | None
    model_id: str | None
    env_file: Path
    prompt: str
    negative_prompt: str
    seed: int
    width: int | None
    height: int | None
    frames: int | None
    fps: int
    steps: int | None
    guidance: float | None
    quant: str | None
    cache: str | None
    compile: str | None
    output_dir: Path
    result_jsonl: Path
    save_video: bool | None
    dry_run: bool


@app.callback()
def root_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if sys.stdin.isatty():
        raise typer.Exit(interactive_main_menu())
    typer.echo(ctx.get_help())
    raise typer.Exit(0)


@models_app.callback()
def models_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if sys.stdin.isatty():
        raise typer.Exit(interactive_list_models())
    typer.echo(ctx.get_help())
    raise typer.Exit(0)


@app.command()
def run(
    preset: Annotated[
        str | None,
        typer.Option("--preset", case_sensitive=False, help="Benchmark preset name."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", case_sensitive=False)] = None,
    backend: Annotated[str | None, typer.Option("--backend", case_sensitive=False)] = None,
    model_dir: Annotated[list[Path] | None, typer.Option("--model-dir")] = None,
    model_path: Annotated[Path | None, typer.Option("--model-path")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    prompt: Annotated[str | None, typer.Option("--prompt")] = None,
    negative_prompt: Annotated[str, typer.Option("--negative-prompt")] = "",
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    width: Annotated[int | None, typer.Option("--width")] = None,
    height: Annotated[int | None, typer.Option("--height")] = None,
    frames: Annotated[int | None, typer.Option("--frames")] = None,
    fps: Annotated[int, typer.Option("--fps")] = DEFAULT_FPS,
    steps: Annotated[int | None, typer.Option("--steps")] = None,
    guidance: Annotated[float | None, typer.Option("--guidance")] = None,
    quant: Annotated[str | None, typer.Option("--quant", case_sensitive=False)] = None,
    cache: Annotated[str | None, typer.Option("--cache", case_sensitive=False)] = None,
    compile: Annotated[str | None, typer.Option("--compile", case_sensitive=False)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    result_jsonl: Annotated[Path | None, typer.Option("--result-jsonl")] = None,
    save_video: Annotated[bool | None, typer.Option("--save-video/--no-save-video")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    options = _complete_run_options(
        preset=preset,
        model=model,
        backend=backend,
        model_dir=model_dir or [],
        model_path=model_path,
        model_id=model_id,
        env_file=env_file,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        steps=steps,
        guidance=guidance,
        quant=quant,
        cache=cache,
        compile=compile,
        output_dir=output_dir,
        result_jsonl=result_jsonl,
        save_video=save_video,
        dry_run=dry_run,
    )
    raise typer.Exit(run_command(options))


@app.command()
def profile(
    model: Annotated[str | None, typer.Option("--model", case_sensitive=False)] = None,
    backend: Annotated[str | None, typer.Option("--backend", case_sensitive=False)] = None,
    model_dir: Annotated[list[Path] | None, typer.Option("--model-dir")] = None,
    model_path: Annotated[Path | None, typer.Option("--model-path")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    prompt: Annotated[str | None, typer.Option("--prompt")] = None,
    negative_prompt: Annotated[str, typer.Option("--negative-prompt")] = "",
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    fps: Annotated[int, typer.Option("--fps")] = DEFAULT_FPS,
    guidance: Annotated[float | None, typer.Option("--guidance")] = None,
    quant: Annotated[str | None, typer.Option("--quant", case_sensitive=False)] = None,
    cache: Annotated[str | None, typer.Option("--cache", case_sensitive=False)] = None,
    compile: Annotated[str | None, typer.Option("--compile", case_sensitive=False)] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("artifacts/profiles"),
    result_jsonl: Annotated[Path | None, typer.Option("--result-jsonl")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    options = _complete_profile_options(
        model=model,
        backend=backend,
        model_dir=model_dir or [],
        model_path=model_path,
        model_id=model_id,
        env_file=env_file,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        fps=fps,
        guidance=guidance,
        quant=quant,
        cache=cache,
        compile=compile,
        output_dir=output_dir,
        results_dir=results_dir,
        result_jsonl=result_jsonl,
        dry_run=dry_run,
    )
    raise typer.Exit(profile_command(options))


@app.command()
def report(
    input: Annotated[Path, typer.Option("--input")] = ...,
    output: Annotated[Path, typer.Option("--output")] = ...,
) -> None:
    records = read_jsonl(input, max_records=MAX_REPORT_RECORDS)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(records), encoding="utf-8")


@models_app.command("list")
def models_list(
    model: Annotated[str | None, typer.Option("--model", case_sensitive=False)] = None,
    model_dir: Annotated[list[Path] | None, typer.Option("--model-dir")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
) -> None:
    _print_model_candidates(model=model, model_dir=model_dir or [], env_file=env_file)


@models_app.command("import")
def models_import(
    source: Annotated[str, typer.Option("--source", case_sensitive=False)] = "all",
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    source = _validate_choice(source, IMPORT_SOURCES, "source")
    raise typer.Exit(
        _import_model_dirs(
            source=source,
            env_file=env_file,
            dry_run=dry_run,
            require_confirmation=False,
        )
    )


def run_command(options: RunOptions) -> int:
    preset = options.preset or _select_preset_if_needed(options)
    preset_runs = _preset_runs(preset, options) if preset else [_manual_run(options)]
    candidate = _select_model_candidate(options)
    if options.backend == "mlx" and candidate is None:
        _append_model_selection_error(options, preset_runs)
        return 1

    backend = create_backend(options.backend)
    mlx_runtime_required = _mlx_runtime_required(options, backend)
    for preset_run in preset_runs:
        config = RunConfig(
            model=options.model,
            backend=options.backend,
            model_path=str(candidate.path) if candidate else None,
            model_id=candidate.id if candidate else None,
            model_source_root=str(candidate.source_root) if candidate else None,
            prompt=options.prompt,
            negative_prompt=options.negative_prompt,
            seed=options.seed,
            width=preset_run.width,
            height=preset_run.height,
            frames=preset_run.frames,
            fps=options.fps,
            steps=preset_run.steps,
            guidance=preset_run.guidance,
            quant=preset_run.quant,
            cache=preset_run.cache,
            compile=preset_run.compile,
            output_dir=options.output_dir,
            result_jsonl=options.result_jsonl,
            save_video=preset_run.save_video,
            dry_run=options.dry_run,
            preset=preset,
            variant_label=_variant_label(preset, preset_run) if preset else "manual",
        )
        if mlx_runtime_required:
            guard_error = _mlx_pre_run_guard(config.variant_label or "manual", config=config)
            if guard_error is not None:
                records = _error_records_for_config(config, error=guard_error)
                append_jsonl(options.result_jsonl, records)
                return 1
            parent_error = _mlx_parent_process_execution_error(options, backend)
            if parent_error is not None:
                records = _error_records_for_config(config, error=parent_error)
                append_jsonl(options.result_jsonl, records)
                return 1

        memory_aborted = False
        guard_failed = False
        runtime_abort_error = None
        guard_error = None
        cleanup_status = None
        try:
            records = Profiler(backend).run(config)
        except _memory_guard_error_type() as exc:
            guard_failed = True
            guard_error = f"Memory guard blocked run: {_safe_exception_text(exc)}"
        except _runtime_memory_abort_type() as exc:
            memory_aborted = True
            runtime_abort_error = f"Runtime memory abort: {_safe_exception_text(exc)}"
        finally:
            if mlx_runtime_required:
                cleanup_status = _mlx_post_run_cleanup(config.variant_label or "manual")
        cleanup_error = _mlx_cleanup_failure_error(cleanup_status) if mlx_runtime_required else None
        if cleanup_error is not None and not guard_failed and not memory_aborted:
            guard_failed = True
            guard_error = cleanup_error
        if guard_failed:
            records = _error_records_for_config(
                config,
                error=guard_error or "Memory guard blocked run",
                guard_context=cleanup_status,
            )
        if memory_aborted:
            records = _error_records_for_config(
                config,
                error=runtime_abort_error or "Runtime memory abort",
                guard_context=cleanup_status,
            )
        records, limit_error = _bounded_profile_records(records, current_count=0)
        if limit_error is not None:
            guard_failed = True
            records = _error_records_for_config(
                config,
                error=limit_error,
                guard_context=cleanup_status,
            )
        append_jsonl(options.result_jsonl, records)
        if options.backend == "mlx" and _records_have_errors(records):
            return 1
        if limit_error is not None or memory_aborted or guard_failed:
            return 1
    return 0


def _records_have_errors(records: list[MeasurementRecord]) -> bool:
    return any(record.error for record in records)


def _bounded_profile_records(
    records: object,
    *,
    current_count: int,
) -> tuple[list[MeasurementRecord], str | None]:
    limit_error = _profile_record_limit_error(records, current_count=current_count)
    if limit_error is not None:
        return [], limit_error
    if isinstance(records, list):
        return records, None

    remaining = MAX_REPORT_RECORDS - current_count
    bounded: list[MeasurementRecord] = []
    for index, record in enumerate(records):  # type: ignore[operator]
        if index >= remaining:
            return [], _profile_record_limit_message(current_count + index + 1)
        bounded.append(record)
    return bounded, None


def _profile_record_limit_error(records: object, *, current_count: int) -> str | None:
    try:
        record_count = len(records)  # type: ignore[arg-type]
    except TypeError:
        return None
    total_count = current_count + record_count
    if total_count <= MAX_REPORT_RECORDS:
        return None
    return _profile_record_limit_message(total_count)


def _profile_record_limit_message(total_count: int) -> str:
    return (
        f"profile record limit exceeded: {total_count} records > {MAX_REPORT_RECORDS}; "
        "refusing to materialize profile report"
    )


def _extend_profile_report_records(
    all_records: list[dict[str, object]],
    records: list[MeasurementRecord],
) -> str | None:
    for record in records:
        total_count = len(all_records) + 1
        if total_count > MAX_REPORT_RECORDS:
            return _profile_record_limit_message(total_count)
        all_records.append(record.to_dict())
    return None


def _backend_is_scaffold_only(backend: object) -> bool:
    return bool(getattr(backend, "scaffold_only", False))


def _mlx_runtime_required(options: RunOptions, backend: object) -> bool:
    if options.backend != "mlx":
        return False
    # The generic MLX scaffold does not import Metal today, but selecting the
    # MLX backend is still an opt-in heavy-runtime intent. Keep the CLI guard
    # invariant independent from adapter implementation details so replacing
    # the scaffold with a real adapter cannot silently bypass memory gates.
    return True


def _mlx_parent_process_execution_error(options: RunOptions, backend: object) -> str | None:
    if options.backend != "mlx" or _backend_is_scaffold_only(backend):
        return None
    if os.environ.get(ALLOW_PARENT_MLX_ENV) == "1":
        return None
    return (
        f"Memory guard blocked run: real MLX backend execution in the CLI parent process "
        f"requires {ALLOW_PARENT_MLX_ENV}=1 until a CLI child-process orchestrator is available. "
        "This prevents Metal state from accumulating in the long-lived parent process."
    )


def _memory_guard_error_type() -> type[Exception]:
    try:
        from fastgen_profiler.mlx_guard import MemoryGuardError
    except ImportError:
        return RuntimeError
    return MemoryGuardError


def _runtime_memory_abort_type() -> type[Exception]:
    try:
        from fastgen_profiler.mlx_guard import RuntimeMemoryAbort
    except ImportError:
        return RuntimeError
    return RuntimeMemoryAbort


def _error_records_for_config(
    config: RunConfig,
    *,
    error: str,
    guard_context: dict[str, object] | None = None,
) -> list:
    machine = machine_metadata()
    if guard_context is not None:
        machine["mlx_guard_cleanup"] = guard_context.get("cleanup")
    return [
        make_record(
            config,
            run_id=new_run_id(),
            timestamp_utc=utc_timestamp(),
            machine=machine,
            phase="total",
            seconds=0.0,
            error=error,
        )
    ]


def _mlx_pre_run_guard(label: str, *, config: RunConfig | None = None) -> str | None:
    try:
        from fastgen_profiler.mlx_guard import (
            MAX_CONSECUTIVE_RUNS,
            check_run_allocation_budget,
            check_text_prompt_budget,
            inter_run_system_recovery,
            run_counter,
            should_restart_process,
        )

        if should_restart_process():
            return (
                f"skipped: process restart required after {run_counter()} "
                "consecutive MLX runs to prevent Metal resource accumulation."
            )

        budget = None
        if config is not None:
            budget = check_run_allocation_budget(
                width=config.width,
                height=config.height,
                frames=config.frames,
                guidance=config.guidance,
                label=label,
            )
            check_text_prompt_budget(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                label=label,
            )
        recovery = inter_run_system_recovery(label=label)
        typer.echo(
            f"[guard] pre-run '{label}': "
            f"free={recovery.get('free_gb', '?')}GB "
            f"freed={recovery.get('freed_gb', '?')}GB "
            f"shape_floor={(budget or {}).get('shape_floor_gb', '?')}GB "
            f"run={recovery.get('run_number', '?')}/{MAX_CONSECUTIVE_RUNS}"
        )
    except ImportError as exc:
        return (
            "Memory guard blocked run: mlx_guard unavailable before MLX run: "
            f"{_safe_exception_text(exc)}"
        )
    except Exception as exc:
        return f"Memory guard blocked run: {_safe_exception_text(exc)}"
    return None


def _mlx_post_run_cleanup(label: str) -> dict[str, object] | None:
    try:
        from fastgen_profiler.mlx_guard import increment_run_counter, mlx_cleanup, system_snapshot

        cleanup = mlx_cleanup()
        completed_runs = increment_run_counter()
        try:
            snap = system_snapshot()
        except Exception:
            snap = None
        typer.echo(
            f"[guard] post-run '{label}': run={completed_runs} "
            f"{snap.summary() if snap is not None else 'free=?GB'} "
            f"freed={cleanup.get('freed_gb', '?')}GB"
        )
        return {"snapshot": snap, "cleanup": cleanup, "run_number": completed_runs}
    except ImportError as exc:
        typer.echo(
            f"[guard] post-run '{label}': cleanup unavailable because mlx_guard could not be imported: "
            f"{_safe_exception_text(exc)}"
        )
        return {
            "snapshot": None,
            "cleanup": {
                "mlx_loaded": None,
                "mlx_cache_cleared": False,
                "mlx_cleanup_error": "mlx_guard unavailable after MLX run",
            },
            "run_number": None,
        }


def _mlx_cleanup_failure_error(cleanup_status: dict[str, object] | None) -> str | None:
    if cleanup_status is None:
        return "Memory guard blocked run: MLX post-run cleanup did not return status"
    cleanup = cleanup_status.get("cleanup")
    if not isinstance(cleanup, dict):
        return "Memory guard blocked run: MLX post-run cleanup returned invalid status"
    cleanup_error = cleanup.get("mlx_cleanup_error")
    if cleanup_error:
        return f"Memory guard blocked run: MLX post-run cleanup failed: {cleanup_error}"
    if cleanup.get("mlx_loaded") is True and cleanup.get("mlx_cache_cleared") is False:
        return "Memory guard blocked run: MLX post-run cleanup did not clear loaded MLX cache"
    return None


def _adaptive_adjust_spec(
    spec: ProfileRunSpec,
    adaptive_state: dict[str, object],
    *,
    backend_name: str,
) -> ProfileRunSpec:
    """Adjust a ProfileRunSpec based on memory headroom from previous runs.

    If the previous post-run snapshot showed low memory headroom, shrink
    frames/steps to prevent kernel watchdog timeout. If headroom is good,
    leave the spec unchanged (or grow toward original target).

    This only applies to the mlx backend — stub runs are never adjusted.
    """
    if backend_name != "mlx":
        return spec

    last_snap = adaptive_state.get("last_snapshot")
    if last_snap is None:
        # No previous snapshot — first spec runs at original size.
        return spec

    from fastgen_profiler.mlx_guard import SystemSnapshot

    if not isinstance(last_snap, SystemSnapshot):
        return spec

    # Compute headroom: free_fraction if available, else inverse of pressure.
    headroom: float | None = None
    if last_snap.free_fraction is not None:
        headroom = last_snap.free_fraction
    elif last_snap.pressure is not None:
        headroom = 1.0 - last_snap.pressure

    if headroom is None:
        return spec

    original_frames = spec.run.frames
    original_steps = spec.run.steps

    # Thresholds
    SHRINK_BELOW = 0.15  # <15% free → halve frames/steps
    GROW_ABOVE = 0.35    # >35% free → grow toward original if we shrunk before

    shrunk_specs: set = adaptive_state.get("shrunk_specs", set())  # type: ignore[assignment]

    if headroom < SHRINK_BELOW:
        # Danger zone: cut frames and steps in half (minimum 4 frames, 2 steps).
        new_frames = max(4, original_frames // 2)
        new_steps = max(2, original_steps // 2)
        shrunk_specs.add(spec.variant_label)
        typer.echo(
            f"[adaptive] low headroom ({headroom * 100:.0f}%): "
            f"shrinking {spec.variant_label} "
            f"{original_frames}f/{original_steps}s -> {new_frames}f/{new_steps}s"
        )
    elif headroom > GROW_ABOVE and spec.variant_label in shrunk_specs:
        # Recovered: restore original size.
        new_frames = original_frames
        new_steps = original_steps
        shrunk_specs.discard(spec.variant_label)
        typer.echo(
            f"[adaptive] good headroom ({headroom * 100:.0f}%): "
            f"restoring {spec.variant_label} to {new_frames}f/{new_steps}s"
        )
    elif headroom < 0.25 and spec.variant_label not in shrunk_specs:
        # Moderate pressure: reduce by 25%.
        new_frames = max(4, int(original_frames * 0.75))
        new_steps = max(2, int(original_steps * 0.75))
        shrunk_specs.add(spec.variant_label)
        typer.echo(
            f"[adaptive] moderate pressure ({headroom * 100:.0f}%): "
            f"reducing {spec.variant_label} "
            f"{original_frames}f/{original_steps}s -> {new_frames}f/{new_steps}s"
        )
    else:
        return spec

    return ProfileRunSpec(
        preset=spec.preset,
        variant_label=spec.variant_label,
        run=PresetRun(
            width=spec.run.width,
            height=spec.run.height,
            frames=new_frames,
            steps=new_steps,
            guidance=spec.run.guidance,
            quant=spec.run.quant,
            cache=spec.run.cache,
            compile=spec.run.compile,
            save_video=spec.run.save_video,
        ),
    )


def profile_command(options: RunOptions) -> int:
    profile_id = str(uuid.uuid4())
    profile_name = f"{options.model}-full-preset-suite"
    specs, skipped_specs = _profile_run_specs(options)
    candidate = _select_model_candidate(options)
    if options.backend == "mlx" and candidate is None:
        all_records = []
        for spec in specs:
            records = _profile_error_records(
                options=options,
                profile_id=profile_id,
                profile_name=profile_name,
                spec=spec,
                error=(
                    "model selection required: pass --model-path, --model-id, --model-dir, "
                    "or configure FASTGEN_MODEL_DIRS in .env"
                ),
            )
            append_jsonl(options.result_jsonl, records)
            limit_error = _extend_profile_report_records(all_records, records)
            if limit_error is not None:
                typer.echo(f"[guard] report limit blocked: {limit_error}")
                break
        report_path = options.result_jsonl.with_suffix(".md")
        report_path.write_text(render_markdown_report(all_records), encoding="utf-8")
        _print_profile_summary(all_records, options.result_jsonl, report_path)
        return 1

    backend = create_backend(options.backend)
    mlx_runtime_required = _mlx_runtime_required(options, backend)
    all_records = []

    # Guard state: tracks memory headroom across specs.
    adaptive_state: dict[str, object] = {
        "last_snapshot": None,
        "shrunk_specs": set(),
    }
    memory_guard_failed = False

    for spec in specs:
        spec_label = spec.variant_label

        # Guard 2: Adaptive batch sizing.
        effective_spec = _adaptive_adjust_spec(
            spec, adaptive_state, backend_name=options.backend,
        )

        config = _profile_run_config(
            options=options,
            candidate=candidate,
            profile_id=profile_id,
            profile_name=profile_name,
            spec=effective_spec,
        )

        if mlx_runtime_required:
            guard_error = _mlx_pre_run_guard(spec_label, config=config)
            if guard_error is not None:
                memory_guard_failed = True
                typer.echo(f"[guard] pre-run '{spec_label}' blocked: {guard_error}")
                records = _profile_error_records(
                    options=options,
                    profile_id=profile_id,
                    profile_name=profile_name,
                    spec=effective_spec,
                    error=guard_error,
                    candidate=candidate,
                )
                append_jsonl(options.result_jsonl, records)
                limit_error = _extend_profile_report_records(all_records, records)
                if limit_error is not None:
                    typer.echo(f"[guard] report limit blocked: {limit_error}")
                break
            parent_error = _mlx_parent_process_execution_error(options, backend)
            if parent_error is not None:
                memory_guard_failed = True
                records = _profile_error_records(
                    options=options,
                    profile_id=profile_id,
                    profile_name=profile_name,
                    spec=effective_spec,
                    error=parent_error,
                    candidate=candidate,
                )
                append_jsonl(options.result_jsonl, records)
                limit_error = _extend_profile_report_records(all_records, records)
                if limit_error is not None:
                    typer.echo(f"[guard] report limit blocked: {limit_error}")
                break

        memory_aborted = False
        guard_failed = False
        runtime_abort_error = None
        guard_error = None
        cleanup_status = None
        try:
            records = Profiler(backend).run(config)
        except _memory_guard_error_type() as exc:
            guard_failed = True
            memory_guard_failed = True
            safe_error = _safe_exception_text(exc)
            logger.warning(f"Memory guard failure for '{spec_label}': {safe_error}")
            guard_error = f"Memory guard blocked run: {safe_error}"
        except _runtime_memory_abort_type() as exc:
            memory_aborted = True
            memory_guard_failed = True
            safe_error = _safe_exception_text(exc)
            logger.warning(f"Runtime memory abort for '{spec_label}': {safe_error}")
            runtime_abort_error = f"Runtime memory abort: {safe_error}"
        finally:
            if mlx_runtime_required:
                cleanup_status = _mlx_post_run_cleanup(spec_label)
                snap = cleanup_status.get("snapshot") if cleanup_status is not None else None
                if snap is not None:
                    adaptive_state["last_snapshot"] = snap
        cleanup_error = _mlx_cleanup_failure_error(cleanup_status) if mlx_runtime_required else None
        if cleanup_error is not None and not guard_failed and not memory_aborted:
            guard_failed = True
            memory_guard_failed = True
            guard_error = cleanup_error
        if guard_failed:
            records = _profile_error_records(
                options=options,
                profile_id=profile_id,
                profile_name=profile_name,
                spec=effective_spec,
                error=guard_error or "Memory guard blocked run",
                candidate=candidate,
                guard_context=cleanup_status,
            )
        if memory_aborted:
            records = _profile_error_records(
                options=options,
                profile_id=profile_id,
                profile_name=profile_name,
                spec=effective_spec,
                error=runtime_abort_error or "Runtime memory abort",
                candidate=candidate,
                guard_context=cleanup_status,
            )

        records, limit_error = _bounded_profile_records(records, current_count=len(all_records))
        if limit_error is not None:
            memory_guard_failed = True
            records = _profile_error_records(
                options=options,
                profile_id=profile_id,
                profile_name=profile_name,
                spec=effective_spec,
                error=limit_error,
                candidate=candidate,
                guard_context=cleanup_status,
            )

        append_jsonl(options.result_jsonl, records)
        report_limit_error = _extend_profile_report_records(all_records, records)
        if report_limit_error is not None:
            memory_guard_failed = True
            typer.echo(f"[guard] report limit blocked: {report_limit_error}")
            break
        if options.backend == "mlx" and _records_have_errors(records):
            memory_guard_failed = True
        if (
            limit_error is not None
            or memory_aborted
            or guard_failed
            or (options.backend == "mlx" and _records_have_errors(records))
        ):
            break

    for spec in skipped_specs:
        records = _skipped_profile_records(
            options=options,
            profile_id=profile_id,
            profile_name=profile_name,
            spec=spec,
            reason="skipped: stress preset is currently limited to wan2.2",
        )
        append_jsonl(options.result_jsonl, records)
        limit_error = _extend_profile_report_records(all_records, records)
        if limit_error is not None:
            memory_guard_failed = True
            typer.echo(f"[guard] report limit blocked: {limit_error}")
            break

    report_path = options.result_jsonl.with_suffix(".md")
    report_path.write_text(render_markdown_report(all_records), encoding="utf-8")
    _print_profile_summary(all_records, options.result_jsonl, report_path)
    return 1 if memory_guard_failed else 0


def _print_model_candidates(*, model: str | None, model_dir: list[Path], env_file: Path) -> None:
    model = _validate_optional_choice(model, MODEL_CHOICES, "model")
    models = (model,) if model else MODEL_CHOICES
    all_candidates = []
    seen_paths: set[str] = set()
    for target_model in models:
        roots = model_dirs_from_sources(
            model=target_model,
            cli_dirs=model_dir,
            env_file=env_file,
        )
        candidates = discover_models(roots, model=target_model)
        for candidate in candidates:
            key = str(candidate.path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            all_candidates.append(candidate)

    if not all_candidates:
        target = model if model else "wan2.2/ltx2.3"
        typer.echo(f"No generation model candidates found for {target}.")
        return

    for index, candidate in enumerate(all_candidates, start=1):
        data = candidate_to_dict(candidate)
        markers = ",".join(data["markers"])
        typer.echo(
            f"{index}. {data['id']} "
            f"family={data['model_family_guess']} markers={markers} path={data['path']}"
        )


def _complete_run_options(
    *,
    preset: str | None,
    model: str | None,
    backend: str | None,
    model_dir: list[Path],
    model_path: Path | None,
    model_id: str | None,
    env_file: Path,
    prompt: str | None,
    negative_prompt: str,
    seed: int | None,
    width: int | None,
    height: int | None,
    frames: int | None,
    fps: int,
    steps: int | None,
    guidance: float | None,
    quant: str | None,
    cache: str | None,
    compile: str | None,
    output_dir: Path | None,
    result_jsonl: Path | None,
    save_video: bool | None,
    dry_run: bool,
) -> RunOptions:
    interactive = sys.stdin.isatty()
    preset = _validate_optional_choice(preset, PRESET_CHOICES, "preset")
    model = _validate_optional_choice(model, MODEL_CHOICES, "model")
    backend = _validate_optional_choice(backend, BACKEND_CHOICES, "backend")
    quant = _validate_optional_choice(quant, QUANT_CHOICES, "quant")
    cache = _validate_optional_choice(cache, CACHE_CHOICES, "cache")
    compile = _validate_optional_choice(compile, COMPILE_CHOICES, "compile")
    _validate_positive_capped_int(fps, "fps", MAX_CLI_FPS)

    if interactive:
        model = model or _prompt_choice("Model", MODEL_CHOICES, "wan2.2")
        backend = backend or _prompt_choice("Backend", BACKEND_CHOICES, "stub")
        preset = preset or _prompt_choice("Preset", PRESET_CHOICES, "smoke")
        prompt = prompt if prompt is not None else _prompt_text("Prompt", "test prompt")
        seed = seed if seed is not None else _prompt_int("Seed", 1)
        output_dir = output_dir or Path(_prompt_text("Output directory", "artifacts/videos"))
        result_jsonl = result_jsonl or Path(_prompt_text("Result JSONL", "artifacts/results.jsonl"))
    else:
        missing = []
        if model is None:
            missing.append("model")
        if backend is None:
            missing.append("backend")
        if prompt is None:
            missing.append("prompt")
        if seed is None:
            missing.append("seed")
        if output_dir is None:
            missing.append("output-dir")
        if result_jsonl is None:
            missing.append("result-jsonl")
        if missing:
            raise typer.BadParameter(
                "Missing required options: "
                + ", ".join(f"--{name}" for name in missing)
            )

    return RunOptions(
        preset=preset,
        model=model,
        backend=backend,
        model_dir=model_dir,
        model_path=model_path,
        model_id=model_id,
        env_file=env_file,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        steps=steps,
        guidance=guidance,
        quant=quant,
        cache=cache,
        compile=compile,
        output_dir=output_dir,
        result_jsonl=result_jsonl,
        save_video=save_video,
        dry_run=dry_run,
    )


def _complete_profile_options(
    *,
    model: str | None,
    backend: str | None,
    model_dir: list[Path],
    model_path: Path | None,
    model_id: str | None,
    env_file: Path,
    prompt: str | None,
    negative_prompt: str,
    seed: int | None,
    fps: int,
    guidance: float | None,
    quant: str | None,
    cache: str | None,
    compile: str | None,
    output_dir: Path | None,
    results_dir: Path,
    result_jsonl: Path | None,
    dry_run: bool,
) -> RunOptions:
    interactive = sys.stdin.isatty()
    model = _validate_optional_choice(model, MODEL_CHOICES, "model")
    backend = _validate_optional_choice(backend, BACKEND_CHOICES, "backend")
    quant = _validate_optional_choice(quant, QUANT_CHOICES, "quant")
    cache = _validate_optional_choice(cache, CACHE_CHOICES, "cache")
    compile = _validate_optional_choice(compile, COMPILE_CHOICES, "compile")
    _validate_positive_capped_int(fps, "fps", MAX_CLI_FPS)

    if interactive:
        model = model or _prompt_choice("Model", MODEL_CHOICES, "wan2.2")
        backend = backend or _prompt_choice("Backend", BACKEND_CHOICES, "stub")
        prompt = prompt if prompt is not None else _prompt_text("Prompt", "test prompt")
        seed = seed if seed is not None else _prompt_int("Seed", 1)
    else:
        missing = []
        if model is None:
            missing.append("model")
        if backend is None:
            missing.append("backend")
        if prompt is None:
            missing.append("prompt")
        if seed is None:
            missing.append("seed")
        if missing:
            raise typer.BadParameter(
                "Missing required options: "
                + ", ".join(f"--{name}" for name in missing)
            )

    output_dir = output_dir or Path("artifacts/videos")
    result_jsonl = result_jsonl or _default_profile_jsonl(results_dir=results_dir, model=model)
    return RunOptions(
        preset=None,
        model=model,
        backend=backend,
        model_dir=model_dir,
        model_path=model_path,
        model_id=model_id,
        env_file=env_file,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=None,
        height=None,
        frames=None,
        fps=fps,
        steps=None,
        guidance=guidance,
        quant=quant,
        cache=cache,
        compile=compile,
        output_dir=output_dir,
        result_jsonl=result_jsonl,
        save_video=None,
        dry_run=dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    command = typer.main.get_command(app)
    args = sys.argv[1:] if argv is None else argv
    try:
        result = command.main(
            args=args,
            prog_name="fastgen-profile",
            standalone_mode=False,
        )
    except click.BadParameter as exc:
        raise SystemExit(_safe_exception_text(exc)) from exc
    except click.exceptions.Exit as exc:
        return int(exc.exit_code or 0)
    return int(result or 0)


def interactive_main_menu() -> int:
    while True:
        typer.echo("Select command:")
        typer.echo("1. Run profile")
        typer.echo("2. List models")
        typer.echo("3. Import model directories")
        typer.echo("4. Exit")
        choice = input("Command number: ").strip()
        if choice == "1":
            return interactive_run_profile()
        if choice == "2":
            return interactive_list_models()
        if choice == "3":
            return interactive_import_model_dirs()
        if choice == "4":
            return 0
        typer.echo("Invalid command selection.")


def interactive_import_model_dirs() -> int:
    source = "all"
    found = discover_import_dirs(source)
    if not found:
        typer.echo("No known model directories found.")
        return 1
    typer.echo("Found model directories:")
    for path in found:
        typer.echo(f"- {path}")

    confirm = input("Save these directories to .env? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        typer.echo("Import cancelled.")
        return 0
    return _import_model_dirs(
        source=source,
        env_file=Path(".env"),
        dry_run=False,
        require_confirmation=False,
    )


def interactive_run_profile() -> int:
    options = _complete_run_options(
        preset=None,
        model=None,
        backend=None,
        model_dir=[],
        model_path=None,
        model_id=None,
        env_file=Path(".env"),
        prompt=None,
        negative_prompt="",
        seed=None,
        width=None,
        height=None,
        frames=None,
        fps=DEFAULT_FPS,
        steps=None,
        guidance=None,
        quant=None,
        cache=None,
        compile=None,
        output_dir=None,
        result_jsonl=None,
        save_video=None,
        dry_run=False,
    )
    return run_command(options)


def interactive_list_models() -> int:
    _print_model_candidates(model=None, model_dir=[], env_file=Path(".env"))
    return 0


def _select_preset_if_needed(options: RunOptions) -> str | None:
    if _has_complete_manual_run(options):
        return None
    if not sys.stdin.isatty():
        missing = ", ".join(_missing_manual_fields(options))
        raise typer.BadParameter(
            "Manual run is missing required fields "
            f"({missing}). Pass --preset or provide all manual run fields."
        )

    typer.echo("Select benchmark preset:")
    for index, preset in enumerate(PRESET_CHOICES, start=1):
        typer.echo(f"{index}. {preset}")
    while True:
        choice = input("Preset number: ").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(PRESET_CHOICES):
                return PRESET_CHOICES[index - 1]
        typer.echo("Invalid preset selection.")


def _has_complete_manual_run(options: RunOptions) -> bool:
    return not _missing_manual_fields(options)


def _missing_manual_fields(options: RunOptions) -> list[str]:
    required = ("width", "height", "frames", "steps", "guidance", "quant", "cache", "compile")
    missing = [name for name in required if getattr(options, name) is None]
    if options.save_video is None:
        missing.append("save-video/no-save-video")
    return missing


def _manual_run(options: RunOptions) -> PresetRun:
    missing = _missing_manual_fields(options)
    if missing:
        raise typer.BadParameter(
            "Manual run is missing required fields "
            f"({', '.join(missing)}). Pass --preset or provide all manual run fields."
        )
    return _validate_preset_run(
        PresetRun(
            width=options.width,
            height=options.height,
            frames=options.frames,
            steps=options.steps,
            guidance=options.guidance,
            quant=options.quant,
            cache=options.cache,
            compile=options.compile,
            save_video=options.save_video,
        )
    )


def _preset_runs(preset: str, options: RunOptions) -> list[PresetRun]:
    guidance = options.guidance if options.guidance is not None else DEFAULT_GUIDANCE
    quant = options.quant or "none"
    cache = options.cache or "none"
    save_video = options.save_video

    if preset == "smoke":
        return _validate_preset_runs([
            PresetRun(
                width=384,
                height=384,
                frames=16,
                steps=8,
                guidance=guidance if options.guidance is not None else 1.0,
                quant=quant,
                cache="none",
                compile="off",
                save_video=False if save_video is None else save_video,
            )
        ])

    if preset == "small-baseline":
        return _validate_preset_runs([
            PresetRun(
                width=512,
                height=512,
                frames=24,
                steps=16,
                guidance=variant_guidance,
                quant=variant_quant,
                cache="none",
                compile="off",
                save_video=False if save_video is None else save_video,
            )
            for variant_guidance in (1.0, guidance)
            for variant_quant in ("none", "q8p")
        ])

    if preset == "quality-threshold":
        return _validate_preset_runs([
            PresetRun(
                width=720,
                height=480,
                frames=48,
                steps=steps,
                guidance=guidance,
                quant=quant,
                cache="none",
                compile="off",
                save_video=True if save_video is None else save_video,
            )
            for steps in (16, 24, 32, 40)
        ])

    if preset == "stress":
        if options.model != "wan2.2":
            raise typer.BadParameter("The stress preset is currently limited to --model wan2.2.")
        return _validate_preset_runs([
            PresetRun(
                width=1280,
                height=720,
                frames=81,
                steps=steps,
                guidance=guidance,
                quant=quant,
                cache="none",
                compile="off",
                save_video=False if save_video is None else save_video,
            )
            for steps in (24, 40)
        ])

    if preset == "cache-experiment":
        return _validate_preset_runs([
            PresetRun(
                width=options.width or 512,
                height=options.height or 512,
                frames=options.frames or 24,
                steps=options.steps or 16,
                guidance=guidance,
                quant=quant,
                cache=variant_cache,
                compile="off",
                save_video=False if save_video is None else save_video,
            )
            for variant_cache in ("none", "prompt", "feature", "all")
        ])

    if preset == "compile-experiment":
        return _validate_preset_runs([
            PresetRun(
                width=options.width or 512,
                height=options.height or 512,
                frames=options.frames or 24,
                steps=options.steps or 16,
                guidance=guidance,
                quant=quant,
                cache=cache,
                compile=variant_compile,
                save_video=False if save_video is None else save_video,
            )
            for variant_compile in ("off", "on")
        ])

    raise typer.BadParameter(f"Unknown preset: {preset}")


def _validate_preset_runs(runs: list[PresetRun]) -> list[PresetRun]:
    for run in runs:
        _validate_preset_run(run)
    return runs


def _validate_preset_run(run: PresetRun) -> PresetRun:
    _validate_positive_capped_int(run.width, "width", MAX_CLI_DIMENSION)
    _validate_positive_capped_int(run.height, "height", MAX_CLI_DIMENSION)
    _validate_positive_capped_int(run.frames, "frames", MAX_CLI_FRAMES)
    _validate_positive_capped_int(run.steps, "steps", MAX_CLI_STEPS)
    return run


def _validate_positive_capped_int(value: object, name: str, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise typer.BadParameter(f"{name} must be a positive integer")
    if value > max_value:
        raise typer.BadParameter(f"{name} must be no greater than {max_value}")


def _profile_run_specs(options: RunOptions) -> tuple[list[ProfileRunSpec], list[ProfileRunSpec]]:
    specs: list[ProfileRunSpec] = []
    skipped: list[ProfileRunSpec] = []
    for preset in PROFILE_PRESETS:
        if preset == "stress" and options.model != "wan2.2":
            skipped.extend(
                ProfileRunSpec(
                    preset=preset,
                    variant_label=f"{preset}_steps-{steps}",
                    run=PresetRun(
                        width=1280,
                        height=720,
                        frames=81,
                        steps=steps,
                        guidance=options.guidance if options.guidance is not None else DEFAULT_GUIDANCE,
                        quant=options.quant or "none",
                        cache="none",
                        compile="off",
                        save_video=False,
                    ),
                )
                for steps in (24, 40)
            )
            continue
        for preset_run in _preset_runs(preset, options):
            specs.append(
                ProfileRunSpec(
                    preset=preset,
                    variant_label=_variant_label(preset, preset_run),
                    run=preset_run,
                )
            )
    return specs, skipped


def _variant_label(preset: str | None, preset_run: PresetRun) -> str:
    if preset is None:
        return "manual"
    if preset == "small-baseline":
        return f"{preset}_guidance-{preset_run.guidance:g}_quant-{preset_run.quant}"
    if preset == "quality-threshold":
        return f"{preset}_steps-{preset_run.steps}"
    if preset == "stress":
        return f"{preset}_steps-{preset_run.steps}"
    if preset == "cache-experiment":
        return f"{preset}_cache-{preset_run.cache}"
    if preset == "compile-experiment":
        return f"{preset}_compile-{preset_run.compile}"
    return preset


def _profile_run_config(
    *,
    options: RunOptions,
    candidate: ModelCandidate | None,
    profile_id: str,
    profile_name: str,
    spec: ProfileRunSpec,
) -> RunConfig:
    return RunConfig(
        model=options.model,
        backend=options.backend,
        model_path=str(candidate.path) if candidate else None,
        model_id=candidate.id if candidate else None,
        model_source_root=str(candidate.source_root) if candidate else None,
        prompt=options.prompt,
        negative_prompt=options.negative_prompt,
        seed=options.seed,
        width=spec.run.width,
        height=spec.run.height,
        frames=spec.run.frames,
        fps=options.fps,
        steps=spec.run.steps,
        guidance=spec.run.guidance,
        quant=spec.run.quant,
        cache=spec.run.cache,
        compile=spec.run.compile,
        output_dir=options.output_dir,
        result_jsonl=options.result_jsonl,
        save_video=spec.run.save_video,
        dry_run=options.dry_run,
        profile_id=profile_id,
        profile_name=profile_name,
        preset=spec.preset,
        variant_label=spec.variant_label,
    )


def _skipped_profile_records(
    *,
    options: RunOptions,
    profile_id: str,
    profile_name: str,
    spec: ProfileRunSpec,
    reason: str,
) -> list:
    return _profile_error_records(
        options=options,
        profile_id=profile_id,
        profile_name=profile_name,
        spec=spec,
        error=reason,
    )


def _profile_error_records(
    *,
    options: RunOptions,
    profile_id: str,
    profile_name: str,
    spec: ProfileRunSpec,
    error: str,
    candidate: ModelCandidate | None = None,
    guard_context: dict[str, object] | None = None,
) -> list:
    config = _profile_run_config(
        options=options,
        candidate=candidate,
        profile_id=profile_id,
        profile_name=profile_name,
        spec=spec,
    )
    machine = machine_metadata()
    if guard_context is not None:
        machine["mlx_guard_cleanup"] = guard_context.get("cleanup")
    return [
        make_record(
            config,
            run_id=new_run_id(),
            timestamp_utc=utc_timestamp(),
            machine=machine,
            phase="total",
            seconds=0.0,
            error=error,
        )
    ]


def _default_profile_jsonl(*, results_dir: Path, model: str) -> Path:
    locale = _safe_locale_name()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_filename_part(model)
    return results_dir / f"{timestamp}T{locale}_{safe_model}.jsonl"


def _safe_locale_name() -> str:
    local_zone = time.tzname[0] if time.tzname else "local"
    if time.daylight and len(time.tzname) > 1:
        local_zone = time.tzname[1]
    return _safe_filename_part(local_zone or "local")


def _safe_filename_part(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {".", "-", "_"} else "-" for character in value)
    return safe.strip("-") or "unknown"


def _print_profile_summary(records: list[dict], jsonl_path: Path, report_path: Path) -> None:
    rows = _profile_summary_rows(records)
    typer.echo("Profile suite summary:")
    typer.echo("preset | variant | total_s | slowest_phase | denoise_avg_s | peak_memory | status")
    typer.echo("--- | --- | ---: | --- | ---: | --- | ---")
    for row in rows:
        typer.echo(
            f"{row['preset']} | {row['variant']} | {row['total']:.6f} | "
            f"{row['slowest_phase']} | {row['denoise_avg']:.6f} | "
            f"{row['peak_memory']} | {row['status']}"
        )
    typer.echo(f"Recommended next bottleneck: {_profile_recommendation(records)}")
    typer.echo(f"JSONL: {jsonl_path}")
    typer.echo(f"Report: {report_path}")


def _profile_summary_rows(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(_summary_text(record["run_id"]), []).append(record)
    rows = []
    for run_records in grouped.values():
        first = run_records[0]
        errors = [_summary_text(record["error"]) for record in run_records if record.get("error")]
        status = "skipped" if any(error.startswith("skipped:") for error in errors) else ("failed" if errors else "ok")
        rows.append(
            {
                "preset": _summary_text(first.get("preset") or "manual"),
                "variant": _summary_text(first.get("variant_label") or "manual"),
                "total": _sum_phase(run_records, "total"),
                "slowest_phase": _summary_text(_slowest_recorded_phase(run_records)[0]),
                "denoise_avg": _average_phase(run_records, "denoise_step"),
                "peak_memory": _format_summary_memory(_max_memory(run_records)),
                "status": status,
            }
        )
    return rows


def _profile_recommendation(records: list[dict]) -> str:
    errors = [
        _summary_text(record["error"])
        for record in records
        if record.get("error") and not _summary_text(record["error"]).startswith("skipped:")
    ]
    if errors:
        return f"fix failed run first: {errors[0]}"
    phase_totals: dict[str, float] = {}
    total = 0.0
    for record in records:
        phase = _summary_text(record["phase"])
        seconds = _finite_record_seconds(record)
        if phase == "total":
            total += seconds
        elif phase != "denoise_step":
            phase_totals[phase] = phase_totals.get(phase, 0.0) + seconds
    if not phase_totals:
        return "no phase timing data available"
    phase, seconds = max(phase_totals.items(), key=lambda item: item[1])
    if phase == "denoise_total" and total > 0:
        return f"inspect denoise_total first ({seconds / total * 100.0:.1f}% of total time)"
    return f"inspect {phase} first ({seconds:.6f}s cumulative)"


def _sum_phase(records: list[dict], phase: str) -> float:
    return sum(_finite_record_seconds(record) for record in records if record["phase"] == phase)


def _average_phase(records: list[dict], phase: str) -> float:
    values = [_finite_record_seconds(record) for record in records if record["phase"] == phase]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _slowest_recorded_phase(records: list[dict]) -> tuple[str, float]:
    phases: dict[str, float] = {}
    for record in records:
        phase = _summary_text(record["phase"])
        if phase in {"total", "denoise_step"}:
            continue
        phases[phase] = phases.get(phase, 0.0) + _finite_record_seconds(record)
    if not phases:
        return ("none", 0.0)
    return max(phases.items(), key=lambda item: item[1])


def _max_memory(records: list[dict]) -> int | None:
    values = [
        value
        for value in (_non_negative_record_int(record, "peak_memory") for record in records)
        if value is not None
    ]
    return max(values) if values else None


def _format_summary_memory(value: int | None) -> str:
    return "unavailable" if value is None else str(value)


def _finite_record_seconds(record: dict) -> float:
    raw = record["seconds"]
    if type(raw) not in {int, float, str}:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return value


def _non_negative_record_int(record: dict, key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) not in {int, str}:
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    if converted < 0:
        return None
    return converted


def _summary_text(value: object) -> str:
    text = _safe_summary_text(value).replace("\n", " ").replace("\r", " ").replace("|", "/")
    if len(text) <= MAX_SUMMARY_FIELD_CHARS:
        return text
    return f"{text[: MAX_SUMMARY_FIELD_CHARS - 12]}...<truncated>"


def _safe_summary_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _safe_exception_text(exc: BaseException) -> str:
    parts = [_summary_text(arg) for arg in exc.args[:4]]
    if not parts:
        exc_type = type(exc)
        return _summary_text(f"<{exc_type.__module__}.{exc_type.__qualname__}>")
    if len(exc.args) > 4:
        parts.append("...<truncated>")
    if len(parts) == 1:
        return parts[0]
    exc_type = type(exc)
    return _summary_text(f"{exc_type.__module__}.{exc_type.__qualname__}: {', '.join(parts)}")


def _select_model_candidate(options: RunOptions) -> ModelCandidate | None:
    if options.model_path is not None:
        return direct_model_candidate(options.model_path, model=options.model)

    roots = model_dirs_from_sources(
        model=options.model,
        cli_dirs=options.model_dir,
        env_file=options.env_file,
    )
    candidates = discover_models(roots, model=options.model)

    if options.model_id:
        candidate = resolve_model_candidate(candidates, options.model_id)
        if candidate is None:
            raise typer.BadParameter(f"Model id not found: {options.model_id}")
        return candidate

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if options.backend != "mlx":
        return None
    if not sys.stdin.isatty():
        return None

    typer.echo("Select local model:")
    for index, candidate in enumerate(candidates, start=1):
        typer.echo(f"{index}. {candidate.id} ({candidate.model_family_guess}) {candidate.path}")
    while True:
        choice = input("Model number: ").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        typer.echo("Invalid model selection.")


def _append_model_selection_error(options: RunOptions, preset_runs: list[PresetRun]) -> None:
    run_id = new_run_id()
    timestamp = utc_timestamp()
    machine = machine_metadata()
    message = (
        "model selection required: pass --model-path, --model-id, --model-dir, "
        "or configure FASTGEN_MODEL_DIRS in .env"
    )

    for preset_run in preset_runs:
        config = RunConfig(
            model=options.model,
            backend=options.backend,
            model_path=None,
            model_id=None,
            model_source_root=None,
            prompt=options.prompt,
            negative_prompt=options.negative_prompt,
            seed=options.seed,
            width=preset_run.width,
            height=preset_run.height,
            frames=preset_run.frames,
            fps=options.fps,
            steps=preset_run.steps,
            guidance=preset_run.guidance,
            quant=preset_run.quant,
            cache=preset_run.cache,
            compile=preset_run.compile,
            output_dir=options.output_dir,
            result_jsonl=options.result_jsonl,
            save_video=preset_run.save_video,
            dry_run=options.dry_run,
        )
        append_jsonl(
            options.result_jsonl,
            [
                make_record(
                    config,
                    run_id=run_id,
                    timestamp_utc=timestamp,
                    machine=machine,
                    phase="model_load",
                    seconds=0.0,
                    error=message,
                )
            ],
        )


def _import_model_dirs(
    *,
    source: str,
    env_file: Path,
    dry_run: bool,
    require_confirmation: bool,
) -> int:
    found = discover_import_dirs(source)
    if not found:
        typer.echo("No known model directories found.")
        return 1

    typer.echo("Discovered app roots:")
    for path in found:
        typer.echo(f"- {path}")

    generation_dirs = discover_generation_model_dirs(found)
    if not generation_dirs:
        typer.echo("No Wan2.2/LTX2.3 generation model candidates found under discovered app roots.")
        return 1

    typer.echo("Generation model directories to register:")
    for path in generation_dirs:
        typer.echo(f"- {path}")
    for path in found:
        if not any(_is_relative_to(candidate, path) for candidate in generation_dirs):
            typer.echo(f"Skipped {path}: no Wan2.2/LTX2.3 generation model candidates found under this root.")
    if source in {"ollama", "lmstudio", "all"}:
        typer.echo("Note: LM Studio/Ollama LLM-only and GGUF-only directories are not registered for this video generation profiler.")

    replacement_preview = [path.expanduser().resolve() for path in generation_dirs]
    typer.echo(f"FASTGEN_MODEL_DIRS={':'.join(str(path) for path in replacement_preview)}")
    if dry_run:
        typer.echo("Dry run: .env was not modified.")
        return 0
    if require_confirmation:
        confirm = input(f"Save to {env_file}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            typer.echo("Import cancelled.")
            return 0

    replacement = replace_model_dirs_in_env(env_file, generation_dirs)
    typer.echo(f"Updated {env_file}: {len(replacement)} model directories registered.")
    return 0


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_choice(value: str | None, choices: tuple[str, ...], name: str) -> str:
    if value is None:
        raise typer.BadParameter(f"{name} is required")
    normalized = value.lower()
    if normalized not in choices:
        raise typer.BadParameter(f"{name} must be one of: {', '.join(choices)}")
    return normalized


def _validate_optional_choice(value: str | None, choices: tuple[str, ...], name: str) -> str | None:
    if value is None:
        return None
    return _validate_choice(value, choices, name)


def _prompt_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    typer.echo(f"{label}:")
    for index, choice in enumerate(choices, start=1):
        suffix = " (default)" if choice == default else ""
        typer.echo(f"{index}. {choice}{suffix}")
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        normalized = raw.lower()
        if normalized in choices:
            return normalized
        typer.echo("Invalid selection.")


def _prompt_text(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            typer.echo("Enter an integer.")


if __name__ == "__main__":
    raise SystemExit(main())
