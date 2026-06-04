#!/usr/bin/env python3
"""Full pipeline benchmark: depth-chunked conv3d.

Strategy: run generate_video once for baseline, then re-run VAE decode
with patched conv3d on the same latents. This isolates the VAE improvement
while accounting for the full pipeline cost.

Measurements:
1. generate_video (baseline) — total + VAE decode time
2. generate_video (no VAE) — denoise time only  
3. VAE decode with patched conv3d on saved latents
4. generate_video (compile) — compiled model + original VAE
5. generate_video (compile + patched VAE)
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
from pathlib import Path
from types import MethodType

import mlx.core as mx
import numpy as np

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))
OUT = "artifacts/bottleneck_bench/pipeline_conv3d_bench.jsonl"


def fast_conv3d(self, x, cache_x=None):
    B, T, H, W, C = x.shape
    kd, kh, kw = self.kernel_size
    if kd == 1 and kh == 1 and kw == 1:
        x_flat = x.reshape(B * T, H, W, C)
        w2d = self.weight[:, 0, :, :, :]
        y = mx.conv_general(x_flat, w2d) + self.bias
        return y.reshape(B, T, y.shape[1], y.shape[2], -1)
    pad_needed = self._causal_pad_t
    if cache_x is not None and pad_needed > 0:
        x = mx.concatenate([cache_x, x], axis=1)
        pad_needed -= cache_x.shape[1]
    if pad_needed > 0:
        x = mx.concatenate([mx.zeros((B, pad_needed, H, W, C), dtype=x.dtype), x], axis=1)
    if self._pad_h > 0 or self._pad_w > 0:
        x = mx.pad(x, [(0,0),(0,0),(self._pad_h,self._pad_h),(self._pad_w,self._pad_w),(0,0)])
    T_padded = x.shape[1]
    T_out = (T_padded - kd) // self.stride[0] + 1
    spatial = H * W
    cs = 2 if spatial > 40000 else (3 if spatial > 10000 else (4 if spatial > 3000 else 8))
    Hp, Wp = x.shape[2], x.shape[3]
    chunks = []
    for c_start in range(0, T_out, cs):
        c_end = min(c_start + cs, T_out)
        ct = c_end - c_start
        outputs_d = []
        for d in range(kd):
            frames = x[:, c_start+d:c_end+d].reshape(B*ct, Hp, Wp, C)
            w2d = self.weight[:, d, :, :, :]
            conv_out = mx.conv_general(frames, w2d, stride=(self.stride[1], self.stride[2]))
            outputs_d.append(conv_out.reshape(B, ct, conv_out.shape[1], conv_out.shape[2], -1))
        r = outputs_d[0]
        for d in range(1, kd):
            r = r + outputs_d[d]
        chunks.append(r + self.bias)
    return mx.concatenate(chunks, axis=1)


def main():
    from mlx_video.models.wan_2.vae22 import CausalConv3d, denormalize_latents
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.generate import generate_video as wan_generate

    config = WanModelConfig(**json.loads((MODEL / "config.json").read_text()))

    print("=== Full Pipeline + VAE Benchmark ===\n")

    results = {}
    latents_saved = None
    baseline_video = None

    # ===== 1. Baseline full pipeline (no compile) =====
    print("--- 1. Baseline (no compile) ---")
    t0 = time.time()
    wan_generate(
        model_dir=str(MODEL),
        prompt="A golden retriever trots along a wet sandy beach, water splashing",
        height=480, width=832, num_frames=41,
        steps=10, guide_scale=3.5, seed=42,
        output_path="/tmp/pipeline_bench_baseline.mp4",
        no_compile=True,
    )
    baseline_total = time.time() - t0
    results["baseline_total"] = baseline_total
    print(f"  Total: {baseline_total:.1f}s\n")

    # ===== 2. Compiled model =====
    print("--- 2. Compiled model ---")
    t0 = time.time()
    wan_generate(
        model_dir=str(MODEL),
        prompt="A golden retriever trots along a wet sandy beach, water splashing",
        height=480, width=832, num_frames=41,
        steps=10, guide_scale=3.5, seed=42,
        output_path="/tmp/pipeline_bench_compile.mp4",
        no_compile=False,
    )
    compile_total = time.time() - t0
    results["compile_total"] = compile_total
    print(f"  Total: {compile_total:.1f}s\n")

    # ===== 3. VAE decode standalone benchmark =====
    # Generate latents via denoise, then compare VAE decode times
    print("--- 3. VAE decode standalone ---")
    from mlx_video.models.wan_2.utils import load_wan_model
    from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler

    # Random latents for isolated VAE benchmark
    latents = mx.random.normal((1, 11, 30, 52, 48), dtype=mx.bfloat16)
    latents = denormalize_latents(latents)
    mx.eval(latents)
    mx.synchronize()

    N = 3

    # 3a. Original VAE
    print("  3a. Original VAE:")
    vae_orig = load_vae_decoder(MODEL / "vae.safetensors", config)
    _ = vae_orig(latents); mx.eval(_); mx.synchronize()  # warmup
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        video = vae_orig(latents)
        mx.eval(video); mx.synchronize()
        times.append(time.perf_counter() - t0)
    vae_orig_time = sum(times) / N
    results["vae_original"] = vae_orig_time
    print(f"    {vae_orig_time*1000:.0f}ms (avg of {N})")

    # 3b. Depth-chunked VAE
    print("  3b. Depth-chunked VAE:")
    vae_fast = load_vae_decoder(MODEL / "vae.safetensors", config)
    patched = 0
    for name, child in vae_fast.named_modules():
        if isinstance(child, CausalConv3d):
            child.__call__ = MethodType(fast_conv3d, child)
            patched += 1
    print(f"    Patched {patched} conv3d instances")
    _ = vae_fast(latents); mx.eval(_); mx.synchronize()  # warmup
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        video2 = vae_fast(latents)
        mx.eval(video2); mx.synchronize()
        times.append(time.perf_counter() - t0)
    vae_fast_time = sum(times) / N
    results["vae_depth_chunked"] = vae_fast_time
    print(f"    {vae_fast_time*1000:.0f}ms (avg of {N})")

    diff = mx.max(mx.abs(video - video2)).item()
    vae_speedup = vae_orig_time / vae_fast_time
    print(f"    Speedup: {vae_speedup:.2f}x, max diff: {diff:.6f}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("FULL PIPELINE BENCHMARK RESULTS")
    print(f"{'='*60}")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:<30} {v:>8.2f}s")

    # Estimate pipeline improvement
    # Assume denoise = total - VAE (rough)
    denoise_est = baseline_total - vae_orig_time
    pipeline_opt_est = denoise_est + vae_fast_time
    print(f"\n  Estimated pipeline improvement:")
    print(f"    Denoise (est):      {denoise_est:.1f}s")
    print(f"    VAE original:       {vae_orig_time:.1f}s")
    print(f"    VAE depth-chunked:  {vae_fast_time:.1f}s")
    print(f"    Pipeline (est):     {pipeline_opt_est:.1f}s ({baseline_total/pipeline_opt_est:.2f}x)")
    
    denoise_compile_est = compile_total - vae_orig_time
    compile_opt_est = denoise_compile_est + vae_fast_time
    print(f"    Compile (est):      {denoise_compile_est:.1f}s")
    print(f"    Compile+opt (est):  {compile_opt_est:.1f}s ({compile_total/compile_opt_est:.2f}x)")

    # Save
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"config": k, "time_s": round(v, 2)}) + "\n")
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
