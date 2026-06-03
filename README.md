# macgen-iv

Mac and Apple Silicon focused image/video generation profiling and benchmark tooling.

## Mission

`macgen-iv` exists to make local image and video generation performance measurable before it is optimized. The project is a reproducible Python + MLX profiler for model pipelines such as Wan2.2 and LTX2.3; it is not a new inference engine.

The profiler keeps the execution path narrow:

```text
CLI -> profiler -> backend adapter -> model pipeline -> metrics recorder -> report generator
```

The core goals are:

- benchmark the same prompt, seed, model, backend, shape, and quality settings repeatably
- record phase-level and denoise-step timings with MLX synchronization where needed
- keep model-specific loading and execution inside backend adapters
- write benchmark data to JSONL so runs can be appended, compared, and audited
- generate Markdown summaries from structured benchmark records
- make optimization work start from measured bottlenecks, not guesses

The initial implementation is Python + MLX. Rust, Swift, custom inference engines, MLX upstream changes, and custom Metal kernels are out of scope until profiling data proves a specific need.

## How To Use

Install the development environment:

```bash
uv sync --extra dev
```

Inspect the CLI:

```bash
uv run fastgen-profile --help
uv run fastgen-profile run --help
uv run fastgen-profile profile --help
uv run fastgen-profile models --help
```

The installed command in this checkout is `fastgen-profile`; `fastgen-profiler` is also registered as a compatibility alias.

Run a safe smoke benchmark with the deterministic stub backend:

```bash
uv run fastgen-profile run \
  --preset smoke \
  --model wan2.2 \
  --backend stub \
  --prompt "a cinematic mountain flythrough" \
  --seed 7 \
  --output-dir artifacts/videos \
  --result-jsonl artifacts/results.jsonl
```

Generate a Markdown report from a JSONL result file:

```bash
uv run fastgen-profile report \
  --input artifacts/results.jsonl \
  --output artifacts/report.md
```

Run the full preset suite for one model:

```bash
uv run fastgen-profile profile \
  --model wan2.2 \
  --backend stub \
  --prompt "a cinematic mountain flythrough" \
  --seed 7
```

By default, `profile` writes a JSONL file under `artifacts/profiles/` and a matching `.md` report beside it.

To use real MLX backends, install the optional MLX dependencies and point the CLI at local model files:

```bash
uv sync --extra dev --extra mlx
```

Register or discover local generation model directories:

```bash
uv run fastgen-profile models list --model-dir /path/to/models
uv run fastgen-profile models import --source all
uv run fastgen-profile models import --source huggingface --dry-run
```

Model discovery uses `.env` and CLI-provided directories. Supported `.env` keys include:

```bash
FASTGEN_MODEL_DIRS=/Users/me/DrawThings/Models:/Users/me/.cache/huggingface/hub
FASTGEN_MODEL_DIR_WAN22=/Volumes/models/wan
FASTGEN_MODEL_DIR_LTX23=/Volumes/models/ltx
```

Use `--model-path /exact/model/dir` to bypass discovery, or repeat `--model-dir` to add search roots. Model import registers directories only; it does not copy, move, convert, or download model files.

Current caveat: the LTX2.3 MLX adapter can request a missing text encoder through its guarded auto-download helper. That path runs a memory guard before network access, but it is a code-level exception to the repository goal of no automatic model downloads by default.

Convert local checkpoints to the MLX directory layout used by the benchmark adapters:

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

Run a smoke benchmark against a local MLX model:

```bash
uv run fastgen-profile run \
  --preset smoke \
  --model wan2.2 \
  --backend mlx \
  --model-path /path/to/Wan2.2-TI2V-5B-MLX \
  --prompt "a cinematic mountain flythrough" \
  --seed 7 \
  --output-dir artifacts/videos \
  --result-jsonl artifacts/results.jsonl
```

When `fastgen-profile` is run without arguments in an interactive terminal, it opens a small menu for common tasks. Non-interactive runs require the needed options explicitly.

## About Benchmark

Benchmark records are JSONL. Each line is one measurement record, grouped by `run_id`; the Markdown report is a readable view of that structured data, not the source of truth.

Every run records stable inputs and metadata, including:

- model, backend, model path or id, and source root
- prompt and negative prompt hashes
- seed, width, height, frames, FPS, steps, guidance, quant, cache, and compile settings
- phase name, denoise step index where relevant, elapsed seconds, memory fields, output path, and error
- profile id, profile name, preset, variant label, and machine metadata

The measured phases are:

- `model_load`
- `prompt_prepare`
- `text_encoder`
- `latent_init`
- `denoise_total`
- `denoise_step`
- `vae_decode`
- `video_encode`
- `file_write`
- `total`

Built-in presets:

- `smoke`: 384x384, 16 frames, 8 steps, no video by default
- `small-baseline`: 512x512, 24 frames, 16 steps, guidance and quant variants
- `quality-threshold`: 720x480, 48 frames, step-count variants, saves video by default
- `stress`: 1280x720, 81 frames, Wan2.2 only for now
- `cache-experiment`: compares `none`, `prompt`, `feature`, and `all`
- `compile-experiment`: compares compile `off` and `on`

`profile` runs the full preset suite for the selected model. For LTX2.3, the stress preset is recorded as skipped until that backend path is stable enough for the stress shape.

Benchmark comparison is only meaningful when prompt, seed, model, backend, resolution, frames, steps, and relevant quality settings are controlled. Change one dimension at a time when possible.

Real MLX/Metal runs are safety guarded. Run shapes are capped, JSONL read/write sizes are bounded, memory guard failures are recorded as benchmark errors, and heavy model execution is intended to be opt-in with local model files. The stub backend is the default way to verify CLI, JSONL, and report behavior without weights.

## Development

Run tests:

```bash
uv run pytest -q
```

Useful docs:

- `docs/architecture.md`
- `docs/profiling.md`
- `docs/benchmark-matrix.md`
- `docs/memory-safety.md`
- `docs/model-targets.md`
- `docs/optimization-roadmap.md`
