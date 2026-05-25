# Architecture

The profiler is organized as a narrow, measurable execution path:

```text
CLI -> profiler -> backend adapter -> model pipeline -> metrics recorder -> report generator
```

## Layers

### CLI

The CLI owns user input, benchmark configuration, output paths, and report format selection. It should not contain model-specific logic or optimization behavior.

### Profiler

The profiler coordinates one benchmark run. It provides the timing boundaries, calls the selected backend adapter, synchronizes MLX execution where needed, and emits structured metrics.

### Backend Adapter

Backend adapters isolate model-specific behavior. Each adapter translates a common benchmark request into the calls required by a model pipeline.

Initial adapters live under `src/fastgen_profiler/backends/`:

- `wan22.py`
- `ltx23.py`

The profiler core must not hard-code one model pipeline.

### Model Pipeline

The model pipeline is the MLX implementation or wrapper that actually loads weights, prepares prompts, runs denoising, decodes frames, and writes outputs. Exact loading logic can be replaced without changing the profiler contract.

### Metrics Recorder

The metrics recorder stores run metadata, phase timings, step timings, synchronization notes, quality settings, and output artifact references. Benchmark records are written as JSONL so runs can be appended and compared.

### Report Generator

Report generators convert structured metrics into human-readable summaries. Markdown is the first report format, but JSONL remains the source of truth for benchmark data.
