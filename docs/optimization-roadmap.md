# Optimization Roadmap

Optimization work must follow measured profiler evidence.

1. Establish reproducible baseline.
2. Identify bottleneck.
3. Reduce obvious overhead.
4. Try `mx.compile` on pure, stable hot paths.
5. Add prompt/text conditioning cache.
6. Add denoise step profiling.
7. Add sampler comparison.
8. Add attention/cache optimization.
9. Add video chunking/VAE decode optimization.
10. Consider custom Metal kernel only if profiling proves it.

## Rules

- Do not claim speedups without benchmark data.
- Record quality and speed tradeoffs in benchmark output.
- Keep optimization changes isolated enough to compare before and after runs.
- Prefer backend-specific optimization behind adapter interfaces.
- Keep the profiler schema stable as optimization experiments are added.
