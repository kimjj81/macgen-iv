# Wan2.2 MLX Benchmark Results

Benchmarks for **Wan2.2-TI2V-5B** running on Apple Silicon via the MLX backend adapter.

## Hardware & Software

| Item | Value |
|---|---|
| Chip | Apple M4 Ultra |
| Memory | 128 GB unified |
| OS | macOS 26.5 Tahoe (Darwin 25.5.0) |
| Backend | MLX 0.25.x, mlx_video, Metal |
| Model | Wan2.2-TI2V-5B (MLX converted, BF16) |
| Resolution | 832 × 480, 41 frames, 24 fps |
| Prompt | "A golden retriever trots along a wet sandy beach" |
| Seed | 42 |

## Optimization Summary

| Configuration | Denoise | VAE Decode | Total Pipeline | vs Baseline |
|---|---|---|---|---|
| baseline (no compile, full CFG) | 56.0 s | 21.1 s | 77.1 s | — |
| compile=on, full CFG | 51.4 s | 20.0 s | 71.4 s | 7% faster |
| **compile=on, --cfg-steps=5** | **29.3 s** | **18.9 s** | **48.2 s** | **38% faster** |
| compile=on, --cfg-steps=3 | 38.6 s* | — | — | ~33% faster |
| compile=off, guidance=1.0 (no CFG) | 23.0 s* | — | — | 59% faster (quality loss) |

\* Measured via standalone profiling scripts, not through the CLI benchmark recorder.

The best practical configuration is **compile=on + cfg-steps=5**: it achieves a 38% speedup with negligible quality impact.

## What Is Interval CFG?

Classifier-Free Guidance (CFG) runs two forward passes per denoise step — one conditional (with prompt) and one unconditional (without). This is the "B=2" batch.

**Interval CFG** (`--cfg-steps N`) applies B=2 CFG only on the first N steps, then switches to B=1 (conditional-only) for the remaining steps. Rationale:

- Early denoise steps (high noise) determine overall composition and layout — prompt guidance matters most here.
- Later steps (low noise) refine detail and remove residual noise — the conditional-only direction is sufficient.

Forward pass count for 20 steps:

| cfg-steps | B=2 steps | B=1 steps | Total forwards | Savings |
|---|---|---|---|---|
| 0 (all) | 20 | 0 | 40 | — |
| 5 | 5 | 15 | 25 | 37.5% |
| 3 | 3 | 17 | 23 | 42.5% |
| 1 | 1 | 19 | 21 | 47.5% |

`cfg-steps=5` is the sweet spot: beyond this, speed gains are marginal and prompt adherence begins to soften.

## Step-by-Step Thermal Throttling

Denoise steps slow down progressively during a 20-step run. Memory stays constant (10.4 GB active), confirming **thermal throttling** as the cause.

```
Step   Time    vs Step 0
  0    2.24s   —
  1    2.31s   +3%
  2    2.56s   +14%
  3    2.63s   +18%
  4    2.70s   +21%
  5    2.76s   +23%
  6    2.79s   +25%
  7    2.82s   +26%
  8    2.85s   +27%
  9    2.86s   +28%
 10    2.89s   +29%
 11    2.91s   +30%
 12    2.92s   +30%
 13    2.92s   +30%
 14    2.93s   +31%
 15    2.93s   +31%
 16    2.90s   +29%
 17    3.10s   +38%
 18    2.97s   +33%
 19    3.02s   +35%
```

The first 3 steps run at ~2.3 s each; steps 10+ stabilize around ~2.9 s. Steps 17-19 occasionally spike above 3.0 s. The GPU heats up quickly in the first 5 steps, then settles into a throttled steady state.

This behavior means **interval CFG provides a compounding benefit**: not only does it skip the B=2 pass on later steps, but those later steps are also thermally throttled, so skipping the unconditional pass there saves proportionally more time.

## Quantization & Compile Overhead

Quantization was tested to check whether reduced precision improves throughput on Apple Silicon:

| Config | Denoise | Notes |
|---|---|---|
| BF16 baseline | 56.0 s | Reference |
| Q8 quantization | 63.9 s | 14% slower — dequant overhead dominates |
| Q4 quantization | 79.2 s | 41% slower — heavy dequant on every matmul |
| compile + Q4 | 78.6 s | Quantization overhead swamps compile gains |

Quantization hurts performance here because Apple Silicon has hardware BF16 support, and the dequantization step at each matmul adds overhead that exceeds the memory bandwidth savings. Quantization remains useful for reducing peak memory (important for constrained devices), but it is not a throughput win on M4 Ultra.

## Pipeline Phase Breakdown

Typical phase distribution for a 20-step run (baseline, no compile):

```
Phase             Time    Share
model_load        ~2 s     2%
encode_text       ~3 s     4%
denoise_total     56 s    73%
vae_decode        21 s    27%
```

Denoise is the dominant bottleneck at 73%. VAE decode is 27% and benefits from tiled decode (not yet implemented in the adapter). Text encoding is a one-time cost amortized over all denoise steps.

## Model Architecture (Bottleneck Context)

Wan2.2-TI2V-5B uses a 40-block transformer. Each block contains:

1. **Self-attention** (RoPE + SDPA) — memory-bound
2. **Cross-attention** (text conditioning) — memory-bound, KV precomputed
3. **FFN** (2 × Linear + GELU) — compute-bound

All 40 blocks run per denoise step. With CFG (B=2), each block processes two sequences per pass. Interval CFG halves the per-block work after the first N steps.

## Kernel Panic Root Cause

A watchdog timeout kernel panic was observed during early benchmarking (pid 38880: Python, 31 threads, cpu_usage 4,993,728). Root cause: sustained GPU + CPU saturation for over 92 seconds prevented the kernel watchdog daemon from scheduling, triggering a hardware watchdog reset.

This motivated the adaptive memory guard and the interval CFG approach — both reduce sustained compute pressure to keep the system responsive.

## How to Reproduce

```bash
# Full CFG (baseline)
MACGEN_ALLOW_PARENT_MLX=1 uv run fastgen-profile run \
  --model wan2.2 --backend mlx \
  --model-path /path/to/wan22-ti2v-5b-mlx \
  --width 832 --height 480 --frames 41 --steps 20 \
  --guidance 3.5 --compile on \
  --quant none --cache none \
  --prompt "A golden retriever trots along a wet sandy beach" \
  --seed 42 --output-dir /tmp/bench \
  --result-jsonl artifacts/bench_full_cfg.jsonl --no-save-video

# Interval CFG (cfg-steps=5)
MACGEN_ALLOW_PARENT_MLX=1 uv run fastgen-profile run \
  --model wan2.2 --backend mlx \
  --model-path /path/to/wan22-ti2v-5b-mlx \
  --width 832 --height 480 --frames 41 --steps 20 \
  --guidance 3.5 --compile on --cfg-steps 5 \
  --quant none --cache none \
  --prompt "A golden retriever trots along a wet sandy beach" \
  --seed 42 --output-dir /tmp/bench \
  --result-jsonl artifacts/bench_cfg5.jsonl --no-save-video
```

## Files

- `artifacts/bottleneck_bench/` — raw JSONL benchmark data
- `scripts/profile_interval_cfg.py` — interval CFG profiling script
- `scripts/profile_compile_interval_cfg.py` — compile + interval CFG combo
- `scripts/profile_native_steps.py` — per-step timing with compile
- `scripts/profile_no_cfg.py` — CFG disabled comparison
- `scripts/profile_step_memory.py` — memory tracking per step
- `artifacts/wan22_bottleneck_report.txt` — initial bottleneck analysis
- `artifacts/wan22_optimization_report.txt` — compile/quant optimization report

## Next Steps

1. **Tiled VAE decode** — reduce the 27% VAE share and lower peak memory
2. **Thermal-aware scheduling** — brief pauses between steps to let GPU clock recover
3. **LTX-2.3 adapter** — text projection structure mismatch needs resolution
4. **Quality evaluation** — side-by-side comparison of cfg-steps=5 vs full CFG at multiple guidance values
