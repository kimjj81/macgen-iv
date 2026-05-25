"""Command line interface for fastgen-profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Annotated

import click
import typer

from .backends import create_backend
from .metrics import RunConfig, append_jsonl, make_record, machine_metadata, new_run_id, read_jsonl, utc_timestamp
from .models import (
    IMPORT_SOURCES,
    ModelCandidate,
    candidate_to_dict,
    direct_model_candidate,
    discover_import_dirs,
    discover_models,
    model_dirs_from_sources,
    replace_model_dirs_in_env,
    resolve_model_candidate,
)
from .profiler import Profiler
from .reports.markdown import render_markdown_report


PRESET_CHOICES = (
    "smoke",
    "small-baseline",
    "quality-threshold",
    "stress",
    "cache-experiment",
    "compile-experiment",
)
MODEL_CHOICES = ("wan2.2", "ltx2.3")
BACKEND_CHOICES = ("mlx", "stub")
QUANT_CHOICES = ("none", "q8", "q8p", "q4")
CACHE_CHOICES = ("none", "prompt", "feature", "all")
COMPILE_CHOICES = ("off", "on")

DEFAULT_GUIDANCE = 3.5
DEFAULT_FPS = 12

app = typer.Typer(
    help="Profile MLX video generation experiments and write benchmark JSONL.",
    invoke_without_command=True,
    no_args_is_help=False,
)
models_app = typer.Typer(help="Inspect local model directories.")
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


@app.command()
def run(
    preset: Annotated[
        str | None,
        typer.Option("--preset", case_sensitive=False, help="Benchmark preset name."),
    ] = None,
    model: Annotated[str, typer.Option("--model", case_sensitive=False)] = ...,
    backend: Annotated[str, typer.Option("--backend", case_sensitive=False)] = ...,
    model_dir: Annotated[list[Path] | None, typer.Option("--model-dir")] = None,
    model_path: Annotated[Path | None, typer.Option("--model-path")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    prompt: Annotated[str, typer.Option("--prompt")] = ...,
    negative_prompt: Annotated[str, typer.Option("--negative-prompt")] = "",
    seed: Annotated[int, typer.Option("--seed")] = ...,
    width: Annotated[int | None, typer.Option("--width")] = None,
    height: Annotated[int | None, typer.Option("--height")] = None,
    frames: Annotated[int | None, typer.Option("--frames")] = None,
    fps: Annotated[int, typer.Option("--fps")] = DEFAULT_FPS,
    steps: Annotated[int | None, typer.Option("--steps")] = None,
    guidance: Annotated[float | None, typer.Option("--guidance")] = None,
    quant: Annotated[str | None, typer.Option("--quant", case_sensitive=False)] = None,
    cache: Annotated[str | None, typer.Option("--cache", case_sensitive=False)] = None,
    compile: Annotated[str | None, typer.Option("--compile", case_sensitive=False)] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = ...,
    result_jsonl: Annotated[Path, typer.Option("--result-jsonl")] = ...,
    save_video: Annotated[bool | None, typer.Option("--save-video/--no-save-video")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    options = RunOptions(
        preset=_validate_optional_choice(preset, PRESET_CHOICES, "preset"),
        model=_validate_choice(model, MODEL_CHOICES, "model"),
        backend=_validate_choice(backend, BACKEND_CHOICES, "backend"),
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
        quant=_validate_optional_choice(quant, QUANT_CHOICES, "quant"),
        cache=_validate_optional_choice(cache, CACHE_CHOICES, "cache"),
        compile=_validate_optional_choice(compile, COMPILE_CHOICES, "compile"),
        output_dir=output_dir,
        result_jsonl=result_jsonl,
        save_video=save_video,
        dry_run=dry_run,
    )
    raise typer.Exit(run_command(options))


@app.command()
def report(
    input: Annotated[Path, typer.Option("--input")] = ...,
    output: Annotated[Path, typer.Option("--output")] = ...,
) -> None:
    records = read_jsonl(input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(records), encoding="utf-8")


@models_app.command("list")
def models_list(
    model: Annotated[str, typer.Option("--model", case_sensitive=False)] = ...,
    model_dir: Annotated[list[Path] | None, typer.Option("--model-dir")] = None,
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
) -> None:
    model = _validate_choice(model, MODEL_CHOICES, "model")
    roots = model_dirs_from_sources(
        model=model,
        cli_dirs=model_dir or [],
        env_file=env_file,
    )
    candidates = discover_models(roots, model=model)
    if not candidates:
        typer.echo("No model candidates found.")
        return

    for index, candidate in enumerate(candidates, start=1):
        data = candidate_to_dict(candidate)
        markers = ",".join(data["markers"])
        typer.echo(
            f"{index}. {data['id']} "
            f"family={data['model_family_guess']} markers={markers} path={data['path']}"
        )


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
        )
        records = Profiler(backend).run(config)
        append_jsonl(options.result_jsonl, records)
    return 0


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
        raise SystemExit(str(exc)) from exc
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
            typer.echo("Run profile from the CLI with `fastgen-profile run --help`.")
            return 0
        if choice == "2":
            typer.echo("List models from the CLI with `fastgen-profile models list --help`.")
            return 0
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
    return PresetRun(
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


def _preset_runs(preset: str, options: RunOptions) -> list[PresetRun]:
    guidance = options.guidance if options.guidance is not None else DEFAULT_GUIDANCE
    quant = options.quant or "none"
    cache = options.cache or "none"
    save_video = options.save_video

    if preset == "smoke":
        return [
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
        ]

    if preset == "small-baseline":
        return [
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
        ]

    if preset == "quality-threshold":
        return [
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
        ]

    if preset == "stress":
        if options.model != "wan2.2":
            raise typer.BadParameter("The stress preset is currently limited to --model wan2.2.")
        return [
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
        ]

    if preset == "cache-experiment":
        return [
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
        ]

    if preset == "compile-experiment":
        return [
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
        ]

    raise typer.BadParameter(f"Unknown preset: {preset}")


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

    typer.echo("Discovered model directories:")
    for path in found:
        typer.echo(f"- {path}")
    if source in {"ollama", "all"} and any(".ollama" in str(path) for path in found):
        typer.echo("Note: Ollama directories are registered for discovery only; direct Ollama blob loading is not implemented.")

    replacement_preview = [path.expanduser().resolve() for path in found]
    typer.echo(f"FASTGEN_MODEL_DIRS={':'.join(str(path) for path in replacement_preview)}")
    if dry_run:
        typer.echo("Dry run: .env was not modified.")
        return 0
    if require_confirmation:
        confirm = input(f"Save to {env_file}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            typer.echo("Import cancelled.")
            return 0

    replacement = replace_model_dirs_in_env(env_file, found)
    typer.echo(f"Updated {env_file}: {len(replacement)} model directories registered.")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
