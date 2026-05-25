# AGENTS.md

Repository instructions for `macgen-iv`.

## Scope

This repository builds a Mac and Apple Silicon focused image/video generation profiling and optimization engine.

The first target is a reproducible Python + MLX profiler and benchmark tool for video generation models. It is not a custom inference engine.

Architecture and workflow details live in:

- `docs/architecture.md`
- `docs/profiling.md`
- `docs/benchmark-matrix.md`
- `docs/optimization-roadmap.md`
- `docs/model-targets.md`

## Engineering Rules

- Use Python + MLX for the initial implementation.
- Do not start with Rust or Swift.
- Do not modify MLX upstream.
- Keep profiler core logic independent from any single model pipeline.
- Use backend adapters for model-specific loading and execution.
- Every optimization must start from profiler data.
- Do not claim performance improvements without benchmark data.
- Generated benchmark results must be written to JSONL.
- Phase timings must include explicit MLX synchronization where needed.
- Quality and speed tradeoffs must be recorded, not guessed.
- `mx.compile` must only be introduced after a baseline profiler exists.
- Attention, cache, and sampler optimization comes after bottleneck confirmation.
- Custom Metal kernels are out of scope until profiling proves a specific kernel bottleneck.

## Verification

- Run the narrowest useful tests before claiming completion.
- Record commands and verification results in handoffs.
- If verification is partial or skipped, state why.
