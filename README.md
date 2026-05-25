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

## Development

```bash
python -m pip install -e ".[dev]"
pytest
fastgen-profiler --help
```
