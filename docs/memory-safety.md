# Memory Safety

MLX and Metal runs can allocate enough unified memory to destabilize the
machine. The profiler must fail closed before starting heavy work.

## Required Guard Order

Before a process imports `mlx.core` for a model run, it must:

1. Read system memory telemetry.
2. Verify free memory, pressure, and swap thresholds.
3. Check the shape-driven allocation budget.
4. Preflight local model structure and large model, VAE, text-encoder, and
   upscaler files before loading them.
5. Configure MLX resource limits with `configure_mlx_resource_limits()`.

System telemetry helpers may return unknown values when OS commands fail, but
must not leak raw telemetry exceptions past the guard layer. On macOS, unknown
free-memory, memory-pressure, or swap-file telemetry must fail closed before
MLX/Metal work starts and during MLX/Metal runtime checks; high free memory is
not enough when pressure or swap state cannot be verified.
`configure_mlx_resource_limits()` must also enforce the pre-run system gate
itself, so direct calls cannot probe or import MLX while free memory, pressure,
or swap state is already unsafe.
MLX availability probes must run in child processes without capturing
unbounded stdout/stderr into the parent process.
The test-only `FASTGEN_TEST_SKIP_MLX_IMPORT_PROBE=1` switch must only be
honored inside a pytest process; normal CLI/script execution must not be able
to bypass the child-process MLX availability probe with environment variables.
Model structure must not be guessed from package defaults before MLX is
initialized. Wan2.2 and LTX2.3 local adapters must reject missing config files
before importing `mlx.core` or `mlx_video` model modules.

`mlx_cleanup()` must not initialize MLX in a fresh process. It may only clear
MLX caches when `mlx.core` is already loaded in the current process.
Cleanup must return a status record even when free-memory telemetry fails; an
abort path must not be converted into an unhandled cleanup exception.
Synchronization helpers must follow the same rule: calling `synchronize()` or
`synchronize_mlx()` must not import `mlx.core` or configure MLX limits in a
process that has not already loaded MLX for real model work.

Allocator limits must preserve the configured system reserve. If total-memory
telemetry is unavailable on macOS, or if the derived safe MLX cap is below the
minimum usable allocator limit, the run must fail closed instead of starting
MLX with its default allocator behavior.
Cache and wired allocator limits must never exceed the configured MLX memory
limit, even when default values are used.
Allocator and reserve environment overrides must be finite positive numbers;
`nan`, `inf`, non-positive, and byte-unrepresentable values must fail closed.

During a run, the watchdog must check both system memory pressure and MLX
allocator counters when MLX is loaded. If active plus cached MLX memory nears
the configured allocator limit, the run must abort and clean up before system
memory pressure becomes critical.
If runtime telemetry cannot be captured at all, the watchdog must also abort
and clean up; unknown memory state during MLX/Metal work is unsafe.
Large host allocation preflights must follow the same macOS telemetry rule:
unknown pressure or swap telemetry, or critical pressure, must abort before
large file reads, tensor materialization, NumPy conversion, or video encoding.
Video postprocess and frame export preflights must reserve more than the input
frame buffer because lower-level encoders and image writers may allocate
contiguous, converted, or temporary buffers outside MLX allocator counters.

## Direct Scripts

Direct benchmark scripts must be safe by default.

- Real model execution must require an explicit opt-in environment variable.
- Default invocation must not import MLX, NumPy, model adapters, or model files.
- Heavy runs should execute model work in child processes so Metal state does
  not accumulate in the parent process.
- Child results must be written atomically, and stale child result files must
  not be reused after a failed child process.
- Parent processes must not capture unbounded child stdout/stderr or read
  unbounded child result files into memory. Child logs should spool to disk,
  and child result files must have a small maximum size.
- Parent processes may run system-only memory checks and cooldowns between
  child processes, but must not import MLX for that recovery path.
- Direct benchmark scripts must not configure MLX allocator limits before the
  backend adapter has completed local model config and asset preflight.
- Safety-related numeric environment variables, including shape, timeout, log
  tail, and child-result limits, must be parsed as positive integers. Invalid
  or non-positive values must fail closed instead of weakening a guard.
- Child-mode environment variables must also be validated: the step count must
  be positive, and the child result path must remain inside the configured
  output directory.

## CLI Suite Runs

CLI MLX runs are one heavy model execution per process by default. A profile
suite must stop after the first completed MLX run and resume later from a fresh
process or a child-process orchestrator, because Python-level cleanup cannot
prove all Metal state has been released.
CLI pre-run checks must remain system-only: they may check memory telemetry,
shape budgets, prompt budgets, cleanup, and cooldown, but must not configure
MLX allocator limits before the backend has completed local model config and
asset preflight.

For `scripts/steps_benchmark.py`, real execution requires:

```bash
FASTGEN_STEPS_ALLOW_HEAVY=1
```

The historical sweep must also be explicit, including a second opt-in for
multiple heavy child processes:

```bash
FASTGEN_STEPS_ALLOW_MULTIPLE_HEAVY=1
FASTGEN_STEPS_VALUES=24,32,36,40,44
```

Heavy step values run in child processes. If a child reports a runtime abort,
guard block, non-skipped error, malformed result, or no result record, the
parent must stop the remaining heavy steps instead of launching another
MLX/Metal process after a known unsafe signal.

## Downloads

Automatic model, tokenizer, or text-encoder downloads must be disabled by
default. Missing local model assets should raise a clear error unless the
caller explicitly enables download behavior.
Video postprocess helpers must follow the same rule as model phases: frame
shape and host allocation checks run first, then MLX runtime guard/limits, then
any `mlx_video` postprocess import.

## Verification

Run the focused memory-safety tests after changing guard, adapter, or direct
benchmark behavior:

```bash
uv run pytest -q tests/test_mlx_guard.py tests/test_steps_benchmark.py
```

Run the full suite before handoff:

```bash
uv run pytest -q
```

Verify direct benchmark defaults remain safe:

```bash
FASTGEN_STEPS_OUTPUT_BASE=/tmp/macgen-safe-default .venv/bin/python scripts/steps_benchmark.py
```

Expected result: a skip record explaining that
`FASTGEN_STEPS_ALLOW_HEAVY=1` is required. The default run must not attempt a
real MLX model load.

When heavy execution is explicitly enabled, any guard block, abort, dependency
preflight failure, or child-process error must return a non-zero exit code so
automation cannot treat an unsafe or incomplete benchmark as successful.
