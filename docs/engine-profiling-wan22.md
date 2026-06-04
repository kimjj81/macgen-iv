# Wan2.2 Engine-Level Profiling Report

## Methodology

Instrumented every transformer block (30 blocks) across 3 denoising steps at
832×480, 41 frames, B=2 CFG, no compile, bf16 weights. Each sub-component
(Q/K/V projections, RoPE, SDPA, FFN fc1/fc2, cross-attention) timed
individually with `mx.eval()` synchronization.

## Per-Step Total: ~2.4s

### Block-Level Breakdown (per block average)

| Component | Time/block | % of block |
|---|---|---|
| self_attn | 0.0326s | 42.5% |
| ffn | 0.0350s | 45.5% |
| cross_attn | 0.0088s | 11.4% |
| modulation | 0.0005s | 0.6% |

All 30 blocks are uniform — no outlier blocks. Cost scales linearly.

### Self-Attention Sub-Components (30 blocks × 3 steps = 90 samples)

| Operation | Total (90 calls) | Per call | % of self_attn |
|---|---|---|---|
| RoPE apply | 0.750s | 0.0083s | 23.5% |
| SDPA (Metal) | 0.986s | 0.0110s | 30.9% |
| Q projection | 0.530s | 0.0059s | 16.6% |
| K projection | 0.286s | 0.0032s | 9.0% |
| V proj + reshape | 0.296s | 0.0033s | 9.3% |
| O projection | 0.294s | 0.0033s | 9.2% |
| QK norm | 0.049s | 0.0005s | 1.5% |

### FFN Sub-Components

| Operation | Total (90 calls) | Per call | % of FFN |
|---|---|---|---|
| fc1 (1536→8960) | 1.375s | 0.0153s | 43.4% |
| fc2 (8960→1536) | 1.480s | 0.0164s | 46.8% |
| GELU activation | 0.310s | 0.0034s | 9.8% |

### Cross-Attention Sub-Components (K/V pre-cached)

| Operation | Total (90 calls) | Per call | % of cross |
|---|---|---|---|
| Q projection | 0.454s | 0.0050s | 52.7% |
| O projection | 0.280s | 0.0031s | 32.6% |
| SDPA | 0.127s | 0.0014s | 14.7% |

## Where Does Time Actually Go?

```
Per step (~2.4s), all 30 blocks combined:

FFN matmuls (fc1+fc2):    43.9%  1.05s   ← biggest matmul: 1536↔8960
Self-attn SDPA:           13.7%  0.33s   ← Metal kernel, already optimized
RoPE apply:               10.4%  0.25s   ← float32 memory-bound ops
Self-attn linear (Q+K+V): 12.0%  0.29s   ← bf16 GEMM
Self-attn linear (O):      4.1%  0.10s
Cross-attn linear (Q+O):   9.9%  0.24s
Cross-attn SDPA:            1.8%  0.04s
Other (norm, reshape):      4.2%  0.10s
```

## Optimization Opportunities (engine-level, parameter-independent)

### 1. FFN Kernel Fusion — High Impact
**Current**: fc1 → GELU → fc2 are 3 separate Metal dispatches with materialized intermediates.
- fc1 output: [B, seq_len, 8960] bf16 = ~290MB materialized per block
- GELU output: same size
- 30 blocks × 3 dispatches = 90 kernel launches + 2 intermediate tensors per block

**Possible fix**: Fused `fc1 + GELU + fc2` Metal kernel. Eliminates intermediate
tensor allocation and 1 kernel launch per block. Expected saving: 10-15% of FFN time.

### 2. QKV Projection Fusion — Medium Impact
**Current**: Q, K, V are 3 separate linear projections in self-attention.
- 3 separate GEMM calls, each with weight loading overhead
- Q and K projections could be batched into a single GEMM (concatenated weights)

**Possible fix**: Fused QKV projection — single [dim, 3×dim] GEMM instead of 3× [dim, dim].
Reduces kernel launch overhead and may improve GEMM occupancy.
Expected saving: 5-10% of self-attention time.

### 3. RoPE Optimization — Medium Impact
**Current**: RoPE runs in float32 for precision. Per step, called 4× (Q and K, each B=2).
- reshape, stack, multiply, concatenate — all memory-bound float32 ops
- precomputed cos/sin helps but the rotation arithmetic is still expensive

**Possible fix a)**: Fused RoPE Metal kernel (single kernel for Q, single for K) instead of
6 separate MLX ops per call. Expected saving: 30-50% of RoPE time = 3-5% total.

**Possible fix b)**: Compute RoPE in bf16 instead of float32. The precomputed frequencies
are already float32; the rotation could use lower precision with minimal quality impact.
Needs quality validation.

### 4. Memory Layout — Low-Medium Impact
**Current**: Multiple reshape/transpose operations between operations.
- Q/K/V: [B,S,N,D] → transpose → [B,N,S,D] for SDPA, then transpose back
- These create copies and hurt cache locality

**Possible fix**: Pre-arrange weight matrices to produce [B,N,S,D] output directly
from linear projection, avoiding transpose. Only viable with custom kernels.

### 5. mx.compile Graph Optimization — Already Done
Already applied. Gives ~7% improvement from operator fusion and memory planning.

## What Won't Help

- **SDPA optimization**: Already using `mx.fast.scaled_dot_product_attention` — the
  Metal FlashAttention kernel. This is as fast as it gets without custom kernels.
- **Cross-attention KV cache**: Already pre-computed before the denoise loop.
- **Quantization (int8/int4)**: Would reduce memory bandwidth but hurts quality.
  Worth exploring separately for speed-only benchmarks.

## Priority Order

1. **FFN fusion** — largest single win, cleanest to implement
2. **QKV fusion** — second largest, well-understood optimization
3. **RoPE fused kernel** — significant because it's called 120× per step (30 blocks × 4)
4. **Memory layout** — requires custom Metal kernels, higher risk

## Reproduce

```bash
# Block-level breakdown
MACGEN_ALLOW_PARENT_MLX=1 uv run python scripts/deep_profile_wan22.py

# Sub-component breakdown
MACGEN_ALLOW_PARENT_MLX=1 uv run python scripts/deep_profile_subcomponents.py
```
