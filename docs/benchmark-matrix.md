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
