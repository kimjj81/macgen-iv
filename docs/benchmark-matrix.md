# Benchmark Matrix

Benchmarks should compare one changed dimension at a time when possible. Runs are only comparable when prompt, seed, resolution, frames, steps, backend, and model inputs are controlled.

## Recommended Dimensions

### Model

- `wan2.2`
- `ltx2.3`

### Resolution

Use explicit width and height values. Avoid labels such as "small" unless they are expanded into concrete dimensions in the run record.

### Frames

Record exact frame count.

### Steps

Record exact denoising step count.

### Precision / Quantization

Record precision and quantization settings as explicit configuration fields.

### Guidance

Record guidance scale and any model-specific guidance mode.

### Cache

Compare cache off and cache on only after the baseline run is reproducible.

### Compile

Compare compile off and compile on only after a baseline profiler exists. `mx.compile` experiments should target pure, stable hot paths identified by profiling.

## Initial Measurement Options

Use these options as the first reproducible measurement set. Each run should keep prompt and seed fixed unless the option explicitly changes them.

The CLI exposes these as `fastgen-profile run --preset PRESET_NAME`. If `--preset` is omitted and the full manual run shape is not provided, the CLI prompts for a preset in an interactive terminal.

### Smoke Test

Purpose: verify that the profiler, JSONL append path, and report path work before expensive model runs.

- resolution: `384x384`
- frames: `16`
- steps: `8`
- cache: `none`
- compile: `off`
- save video: `off`

Example:

```bash
fastgen-profile run \
  --preset smoke \
  --model wan2.2 \
  --backend stub \
  --prompt "baseline prompt" \
  --seed 1 \
  --output-dir artifacts/videos \
  --result-jsonl artifacts/results.jsonl
```

### Small Baseline

Purpose: establish the first comparable baseline before cache or compile experiments.

- resolution: `512x512`
- frames: `24`
- steps: `16`
- compare guidance: `1.0` and the model default
- compare quant: `q8p` and `none`
- cache: `none`
- compile: `off`
- CLI preset: `small-baseline`

Run one dimension at a time. For example, compare guidance with quant fixed, then compare quant with guidance fixed.

### Quality Threshold Test

Purpose: find the lowest step count that still meets visual quality expectations for the same scenario.

- resolution: `720x480`
- frames: `48`
- steps: `16`, `24`, `32`, `40`
- seed: same value for every run
- prompt: same text for every run
- save video: `on`
- cache: `none`
- compile: `off`
- CLI preset: `quality-threshold`

### Stress Test

Purpose: measure high-resolution/high-frame behavior only after smaller baselines are stable.

- resolution: `1280x720`
- frames: `81`
- steps: `24`, `40`
- model: `wan2.2` first
- defer `ltx2.3` until backend stability is proven separately
- cache: `none`
- compile: `off`
- CLI preset: `stress`

### Cache Experiment

Purpose: isolate cache behavior after a baseline run is reproducible.

- prompt: same text for every run
- seed: same value for every run
- compare cache: `none`, `prompt`, `feature`, `all`
- compile: `off`
- keep model, resolution, frames, steps, guidance, and quant fixed
- CLI preset: `cache-experiment`

### Compile Experiment

Purpose: compare compile behavior only after baseline results identify stable hot paths.

- prerequisite: baseline complete
- compare compile: `off`, `on`
- keep model, shape, frame count, step count, prompt, seed, guidance, quant, and cache fixed
- do not introduce `mx.compile` before a baseline profiler and bottleneck data exist
- CLI preset: `compile-experiment`

## JSONL Record Requirements

Every benchmark result should include:

- model
- backend
- prompt or prompt identifier
- seed
- resolution
- frames
- steps
- precision or quantization
- guidance
- cache state
- compile state
- phase timings
- step timings where available
- output artifact paths
