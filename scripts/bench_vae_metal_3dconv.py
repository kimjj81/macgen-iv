#!/usr/bin/env python3
"""Custom Metal 3D Convolution kernel for Wan2.2 VAE decoder.

Uses mx.fast.metal_kernel with template types for bf16 support.
Each thread computes one output pixel by iterating over the 3x3x3 kernel.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


def make_conv3d_kernel(B, T_pad, H_pad, W_pad, Cin, Cout,
                       T_out, H_out, W_out, kd=3, kh=3, kw=3):
    """Create a specialized Metal 3D conv kernel with hardcoded constants."""
    source = f"""
    constexpr uint KB = {B}, KT_in = {T_pad}, KH_in = {H_pad}, KW_in = {W_pad};
    constexpr uint KC_in = {Cin}, KC_out = {Cout};
    constexpr uint KT_out = {T_out}, KH_out = {H_out}, KW_out = {W_out};
    constexpr uint KKD = {kd}, KKH = {kh}, KKW = {kw};
    
    uint total = KB * KT_out * KH_out * KW_out * KC_out;
    uint out_idx = thread_position_in_grid.x;
    
    if (out_idx >= total) return;
    
    uint co = out_idx % KC_out;
    uint rem = out_idx / KC_out;
    uint wo = rem % KW_out;
    rem = rem / KW_out;
    uint ho = rem % KH_out;
    rem = rem / KH_out;
    uint to = rem % KT_out;
    uint b = rem / KT_out;
    
    float accum = float(bias[co]);
    
    for (uint d = 0; d < KKD; d++) {{
        uint ti = to + d;
        if (ti >= KT_in) continue;
        
        for (uint r = 0; r < KKH; r++) {{
            uint hi = ho + r;
            if (hi >= KH_in) continue;
            
            for (uint s = 0; s < KKW; s++) {{
                uint wi = wo + s;
                if (wi >= KW_in) continue;
                
                uint w_base = ((co * KKD + d) * KKH + r) * KKW + s;
                
                for (uint ci = 0; ci < KC_in; ci++) {{
                    uint in_off = ((b * KT_in + ti) * KH_in + hi) * KW_in + wi;
                    in_off = in_off * KC_in + ci;
                    uint wt_off = w_base * KC_in + ci;
                    accum += float(input[in_off]) * float(weight[wt_off]);
                }}
            }}
        }}
    }}
    
    uint out_off = (((b * KT_out + to) * KH_out + ho) * KW_out + wo) * KC_out + co;
    output[out_off] = T(accum);
"""
    return mx.fast.metal_kernel(
        name=f"conv3d_{B}_{T_pad}_{H_pad}_{W_pad}_{Cin}_{Cout}",
        input_names=["input", "weight", "bias"],
        output_names=["output"],
        source=source,
        ensure_row_contiguous=True,
    )


def main():
    from mlx_video.models.wan_2.vae22 import CausalConv3d
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.config import WanModelConfig
    import json as _json

    print("=== Custom Metal 3D Conv Kernel Benchmark ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))

    # 3x3x3 conv with padding=1 → causal pad=2, spatial pad=1
    kd, kh, kw = 3, 3, 3
    pt, ph, pw = 2, 1, 1

    tests = [
        ("small [1,21,60,104,C1024]", 1, 21, 60, 104, 1024, 1024),
        ("medium [1,41,120,208,C512]", 1, 41, 120, 208, 512, 512),
        ("large [1,41,240,416,C256]", 1, 41, 240, 416, 512, 256),
    ]

    N = 3
    all_results = {}

    for label, B, T, H, W, Cin, Cout in tests:
        print(f"\n--- {label} ---")

        orig = CausalConv3d(Cin, Cout, 3, padding=1)
        x = mx.random.normal((B, T, H, W, Cin), dtype=mx.bfloat16)
        mx.eval(x, orig.weight, orig.bias)
        mx.synchronize()

        # Original per-frame timing
        times = []
        for _ in range(N):
            mx.synchronize(); t0 = time.perf_counter()
            y_orig = orig(x)
            mx.eval(y_orig); mx.synchronize()
            times.append(time.perf_counter() - t0)
        orig_time = sum(times) / N
        all_results[f"{label}_original"] = orig_time
        print(f"  Original: {orig_time*1000:.0f}ms")

        # Prepare padded input manually
        # Causal temporal padding: prepend 2 zero frames
        pad_t = mx.zeros((B, pt, H, W, Cin), dtype=mx.bfloat16)
        x_padded = mx.concatenate([pad_t, x], axis=1)
        # Spatial padding: 1 on each side of H and W
        x_padded = mx.pad(x_padded, [
            (0, 0), (0, 0),
            (ph, ph),
            (pw, pw),
            (0, 0),
        ], mode="constant")
        T_pad = T + pt
        H_pad = H + 2 * ph
        W_pad = W + 2 * pw
        T_out = T  # T + 2 - 3 + 1 = T
        H_out = H
        W_out = W
        mx.eval(x_padded); mx.synchronize()

        # Create and test Metal kernel
        try:
            kernel = make_conv3d_kernel(
                B, T_pad, H_pad, W_pad, Cin, Cout,
                T_out, H_out, W_out, kd, kh, kw
            )

            total_threads = B * T_out * H_out * W_out * Cout
            grid = (total_threads, 1, 1)
            tg = (256, 1, 1)

            # Warmup
            y_metal = kernel(
                inputs=[x_padded, orig.weight, orig.bias],
                template=[("T", mx.bfloat16)],
                grid=grid,
                threadgroup=tg,
                output_shapes=[(B, T_out, H_out, W_out, Cout)],
                output_dtypes=[mx.bfloat16],
            )
            mx.eval(y_metal); mx.synchronize()

            # Correctness
            diff = mx.max(mx.abs(y_orig - y_metal)).item()
            mean_diff = mx.mean(mx.abs(y_orig - y_metal)).item()
            print(f"  Diff: max={diff:.4f}, mean={mean_diff:.6f} {'OK' if diff < 2.0 else 'HIGH'}")

            # Timing
            times = []
            for _ in range(N):
                mx.synchronize(); t0 = time.perf_counter()
                y_metal = kernel(
                    inputs=[x_padded, orig.weight, orig.bias],
                    template=[("T", mx.bfloat16)],
                    grid=grid,
                    threadgroup=tg,
                    output_shapes=[(B, T_out, H_out, W_out, Cout)],
                    output_dtypes=[mx.bfloat16],
                )
                mx.eval(y_metal); mx.synchronize()
                times.append(time.perf_counter() - t0)
            metal_time = sum(times) / N
            speedup = orig_time / metal_time
            all_results[f"{label}_metal_direct"] = metal_time
            print(f"  Metal (direct): {metal_time*1000:.0f}ms ({speedup:.2f}x)")

        except Exception as e:
            print(f"  Metal kernel FAILED: {e}")
            all_results[f"{label}_metal_direct"] = -1

        # Also test: batched per-frame approach using Metal for GEMM-like ops
        # Skip for now - focus on direct conv3d

    # Summary
    print(f"\n{'='*70}")
    print("METAL 3D CONV KERNEL RESULTS")
    print(f"{'='*70}")
    for k, v in sorted(all_results.items(), key=lambda x: x[1]):
        if v >= 0:
            print(f"  {k:<50} {v*1000:>6.0f}ms")
        else:
            print(f"  {k:<50} FAILED")

    out_path = "artifacts/bottleneck_bench/vae_metal_3dconv.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in all_results.items():
            f.write(json.dumps({"test": k, "time_ms": round(v*1000, 1) if v >= 0 else -1}) + "\n")


if __name__ == "__main__":
    main()
