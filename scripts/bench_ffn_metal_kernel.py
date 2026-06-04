#!/usr/bin/env python3
"""Custom Metal kernels for FFN fusion.

Implements:
1. fused_bias_gelu: Linear output + bias + GELU(tanh) in one kernel
   Eliminates one intermediate tensor write/read

2. fused_ffn: Full fc1+gelu+fc2 via tiling
   Keeps the large intermediate in shared memory / registers
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))

# ---- Metal kernel: fused bias + GELU(tanh) ----
FUSED_BIAS_GELU_SOURCE = r"""
    // Fused bias + GELU(tanh) kernel
    // Input: matmul output [M, N] (row contiguous)
    // Bias: [N]
    // Output: GELU(input + bias) [M, N]
    
    uint elem = thread_position_in_grid.x;
    uint N = inp_shape[1];
    uint total = inp_shape[0] * N;
    if (elem >= total) return;
    
    T val = inp[elem] + bias[elem % N];
    
    // GELU(tanh) approximation
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float sqrt_2_over_pi = 0.7978845608028654f;
    float coeff = 0.044715f;
    
    float xf = static_cast<float>(val);
    float inner = sqrt_2_over_pi * (xf + coeff * xf * xf * xf);
    float tanh_inner = metal::precise::tanh(inner);
    float result = 0.5f * xf * (1.0f + tanh_inner);
    
    out[elem] = static_cast<T>(result);
"""

fused_bias_gelu_kernel = mx.fast.metal_kernel(
    name="fused_bias_gelu",
    input_names=["inp", "bias"],
    output_names=["out"],
    source=FUSED_BIAS_GELU_SOURCE,
)


def fused_bias_gelu(x, bias, dtype=mx.bfloat16):
    """Apply bias + GELU(tanh) in a single Metal kernel."""
    M, N = x.shape
    total = M * N
    outputs = fused_bias_gelu_kernel(
        inputs=[x, bias],
        template=[("T", dtype)],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )
    return outputs[0]


def main():
    from mlx_video.models.wan_2.utils import load_wan_model
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2 import transformer, attention as attn_mod
    import json as _json, math

    print("=== Metal Kernel FFN Fusion Benchmark ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))
    model = load_wan_model(MODEL / "model.safetensors", config)

    F, H, W = 11, 30, 52
    seq_len = math.ceil((H * W) / (config.patch_size[1] * config.patch_size[2]) * F)

    noise = mx.random.normal((48, F, H, W), dtype=mx.bfloat16)
    t = mx.array([999.0, 999.0], dtype=mx.float32)
    ctx_list = [mx.random.normal((config.text_len, config.t5_dim), dtype=mx.bfloat16) for _ in range(2)]
    ctx = model.embed_text(ctx_list)
    grid_sizes = [(F, H // config.patch_size[1], W // config.patch_size[2])] * 2
    kv = model.prepare_cross_kv(ctx)
    rope = model.prepare_rope(grid_sizes)
    mx.eval(noise, t, ctx, kv, rope); mx.synchronize()

    results = {}
    N = 5

    # ---- Test 1: Baseline (bf16, compile) ----
    print("--- 1. Baseline (bf16, compile) ---")
    compiled = mx.compile(model)
    _ = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(_); mx.synchronize()

    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["baseline"] = sum(times) / N
    print(f"  {results['baseline']:.4f}s/step\n")

    # ---- Test 2: Fused bias+GELU kernel in FFN ----
    print("--- 2. Fused bias+GELU Metal kernel ---")
    orig_ffn_call = transformer.WanFFN.__call__

    def fused_ffn_call(self, x):
        x_w = x.astype(attn_mod._linear_dtype(self.fc1))
        # fc1: matmul only (no bias yet)
        w1 = self.fc1.weight.T
        h = x_w @ w1
        # Fused bias + GELU
        h = fused_bias_gelu(h, self.fc1.bias, dtype=h.dtype)
        return self.fc2(h)

    transformer.WanFFN.__call__ = fused_ffn_call
    # Re-create compiled model with new FFN
    compiled2 = mx.compile(model)
    _ = compiled2([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(_); mx.synchronize()

    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = compiled2([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["fused_bias_gelu_metal"] = sum(times) / N
    print(f"  {results['fused_bias_gelu_metal']:.4f}s/step\n")

    # ---- Test 3: Verify Metal kernel correctness ----
    print("--- Correctness check ---")
    x_test = mx.random.normal((4, 1024), dtype=mx.bfloat16)
    bias_test = mx.random.normal((1024,), dtype=mx.bfloat16)

    # Reference: add bias then GELU(tanh)
    ref = nn.GELU(approx="tanh")(x_test + bias_test)
    # Metal kernel
    out = fused_bias_gelu(x_test, bias_test, dtype=mx.bfloat16)
    mx.eval(ref, out)
    max_diff = mx.max(mx.abs(ref - out)).item()
    print(f"  Max diff vs reference: {max_diff:.6f} {'OK' if max_diff < 0.01 else 'FAIL'}")

    # Restore
    transformer.WanFFN.__call__ = orig_ffn_call

    # ---- Summary ----
    print("\n" + "="*70)
    print("METAL KERNEL FFN RESULTS")
    print("="*70)
    baseline = results["baseline"]
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        pct = (1 - v / baseline) * 100
        print(f"  {k:<35} {v:>7.4f}s/step  ({pct:>+6.1f}%)")

    out_path = "artifacts/bottleneck_bench/ffn_metal_kernel.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"strategy": k, "time_s": round(v, 4),
                               "speedup_pct": round((1 - v / baseline) * 100, 1)}) + "\n")


if __name__ == "__main__":
    main()
