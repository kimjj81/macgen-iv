# macgen-iv

Mac and Apple Silicon focused image/video generation profiling and optimization tooling.

The initial target is a reproducible profiler and benchmark harness for MLX-based video generation pipelines. It profiles model execution phase by phase, records benchmark runs as JSONL, and keeps optimization decisions tied to measured bottlenecks.

## Current Scope

- Python + MLX first.
- Reproducible benchmark inputs: prompt, seed, resolution, frames, steps, backend, and model.
- Phase-level and step-level timing.
- Backend adapters for target model families such as Wan2.2 and LTX2.3.
- Markdown report generation from structured profiler results.

## Out Of Scope Initially

- Custom inference engine work.
- Rust or Swift implementations.
- MLX upstream changes.
- Performance improvement claims without benchmark evidence.
- Custom Metal kernels before a specific kernel bottleneck is proven.

## Project Layout

```text
AGENTS.md
docs/
src/fastgen_profiler/
tests/
pyproject.toml
```

See `docs/architecture.md` for the layer model and `docs/profiling.md` for the timing schema.

## Quick Start

```bash
uv sync --extra dev
uv run fastgen-profile
uv run fastgen-profile run
uv run fastgen-profile models
```

Use `uv run` when the virtual environment is not activated:

```bash
uv run fastgen-profile run \
  --preset smoke \
  --model wan2.2 \
  --backend stub \
  --prompt "a cinematic mountain flythrough" \
  --seed 7 \
  --output-dir artifacts/videos \
  --result-jsonl artifacts/results.jsonl
uv run fastgen-profile report --input artifacts/results.jsonl --output artifacts/report.md
```

Run the full preset suite for one model and generate comparison output:

```bash
uv run fastgen-profile profile \
  --model wan2.2 \
  --backend stub \
  --prompt "a cinematic mountain flythrough" \
  --seed 7
```

By default this writes `artifacts/profiles/YYYYMMDD_HHmmSST{locale}_wan2.2.jsonl` and a matching Markdown report beside it. The console summary and report compare preset variants by total time, phase breakdown, average denoise step time, peak memory, skipped/failed runs, and the recommended next bottleneck to inspect.

Or activate the virtual environment first:

```bash
source .venv/bin/activate
fastgen-profile
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run fastgen-profile --help
uv run fastgen-profile profile --help
```

To include optional MLX dependencies:

```bash
uv sync --extra dev --extra mlx
```

Running `fastgen-profile` without arguments in an interactive terminal opens a menu for common tasks, including importing local model directories into `.env`.
`fastgen-profile run` and `fastgen-profile models` also open interactive prompts when required values are omitted in a terminal.

Available run presets:

- `smoke`
- `small-baseline`
- `quality-threshold`
- `stress`
- `cache-experiment`
- `compile-experiment`

If `--preset` is omitted and the full manual shape is not provided, the CLI prompts for one of these presets in an interactive terminal.

## Local Models

Register local model roots with CLI options or `.env`:

```bash
FASTGEN_MODEL_DIRS=/Users/me/DrawThings/Models:/Users/me/.cache/huggingface/hub
FASTGEN_MODEL_DIR_WAN22=/Volumes/models/wan
FASTGEN_MODEL_DIR_LTX23=/Volumes/models/ltx
```

List discovered candidates:

```bash
uv run fastgen-profile models list --model-dir /path/to/models
```

`models list` prints all discovered Wan2.2 and LTX2.3 generation model candidates by default. Pass `--model wan2.2` or `--model ltx2.3` only when you want to filter the list.

Import known local model roots, scan them for Wan2.2/LTX2.3 generation model directories, and write only those model directories into `.env`:

```bash
uv run fastgen-profile models import --source all
uv run fastgen-profile models import --source huggingface --dry-run
```

Supported import sources are `drawthings`, `comfyui`, `huggingface`, `lmstudio`, `ollama`, and `all`. Importing registers directories only; it does not copy, move, convert, or download model files.
Import replaces the existing `FASTGEN_MODEL_DIRS` value with Wan2.2/LTX2.3 generation model candidate directories found in the current scan. LM Studio/Ollama LLM-only and GGUF-only directories are skipped. If no generation model candidates are found, the command exits with an error and does not change `.env`.
Draw Things discovery checks both `~/Library/Containers/Draw Things/Data` and `~/Library/Containers/com.liuliu.draw-things/Data`, including their `Documents/Models` subdirectories.

Convert local checkpoints to the MLX directory layout expected by benchmarks:

```bash
uv run fastgen-profile models convert \
  --model wan2.2 \
  --source /path/to/Wan2.2-TI2V-5B \
  --output-dir /path/to/Wan2.2-TI2V-5B-MLX \
  --quantize \
  --bits 4 \
  --register

uv run fastgen-profile models convert \
  --model ltx2.3 \
  --source /path/to/ltx_2.3_22b_distilled_1.1_q6p.safetensors \
  --output-dir /path/to/LTX-2.3-MLX \
  --variant distilled \
  --register
```

`--register` adds the converted output directory to `.env` so later benchmark commands can discover it. Use `--dry-run` to print the conversion command without running it.

Use a specific model:

```bash
uv run fastgen-profile run \
  --preset smoke \
  --model wan2.2 \
  --backend mlx \
  --model-id owner/wan-local \
  --prompt "a cinematic mountain flythrough" \
  --seed 7 \
  --output-dir artifacts/videos \
  --result-jsonl artifacts/results.jsonl
```

`--model-path /exact/model/dir` bypasses discovery. `--model-dir` can be repeated and is combined with `.env` directories.
