#!/usr/bin/env python3
"""VAE full decode benchmark: adaptive conv3d via direct weight replacement.

Instead of monkey-patching the class, we directly replace the forward method
on each CausalConv3d instance in the decoder.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from types import MethodType

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


def adaptive_conv3d_forward(self, x, cache_x=None):
    """Replacement forward: batched for small spatial, per-frame for large."""
    B, T, H, W, C = x.shape
    kd, kh, kw = self.kernel_size

    if kd == 1 and kh == 1 and kw == 1:
        x_flat = x.reshape(B * T, H, W, C)
        w2d = self.weight[:, 0, :, :, :]
        y = mx.conv_general(x_flat, w2d) + self.bias
        return y.reshape(B, T, y.shape[1], y.shape[2], -1)

    # Causal temporal padding
    pad_needed = self._causal_pad_t
    if cache_x is not None and pad_needed > 0:
        x = mx.concatenate([cache_x, x], axis=1)
        pad_needed -= cache_x.shape[1]
    if pad_needed > 0:
        pad_t = mx.zeros((B, pad_needed, H, W, C), dtype=x.dtype)
        x = mx.concatenate([pad_t, x], axis=1)

    # Spatial padding
    if self._pad_h > 0 or self._pad_w > 0:
        x = mx.pad(x, [
            (0, 0), (0, 0),
            (self._pad_h, self._pad_h),
            (self._pad_w, self._pad_w),
            (0, 0),
        ])

    T_padded = x.shape[1]
    T_out = (T_padded - kd) // self.stride[0] + 1

    spatial_size = x.shape[2] * x.shape[3]
    # Batch for small spatial, per-frame for large
    if spatial_size <= 25000:
        return _batched_conv(self, x, B, T_out, kd)
    else:
        return _per_frame_conv(self, x, B, T_out, kd)


def _batched_conv(self, x, B, T_out, kd):
    chunk_size = 8
    output_frames = []
    Hp, Wp, C = x.shape[2], x.shape[3], x.shape[4]

    for cs in range(0, T_out, chunk_size):
        ce = min(cs + chunk_size, T_out)
        ct = ce - cs
        windows = []
        for t in range(cs, ce):
            ts = t * self.stride[0]
            for d in range(kd):
                windows.append(x[:, ts + d])

        batched = mx.stack(windows, axis=0).reshape(ct * kd * B, Hp, Wp, C)
        chunk_result = None
        for d in range(kd):
            idx = []
            for i in range(d, ct * kd, kd):
                for b in range(B):
                    idx.append(i * B + b)
            d_batch = batched[idx]
            w2d = self.weight[:, d, :, :, :]
            conv_out = mx.conv_general(d_batch, w2d, stride=(self.stride[1], self.stride[2]))
            conv_out = conv_out.reshape(ct, B, conv_out.shape[1], conv_out.shape[2], -1)
            chunk_result = conv_out if chunk_result is None else chunk_result + conv_out
        output_frames.append(chunk_result)

    output = mx.concatenate(output_frames, axis=0)
    return output.transpose(1, 0, 2, 3, 4) + self.bias


def _per_frame_conv(self, x, B, T_out, kd):
    outputs = []
    for t in range(T_out):
        ts = t * self.stride[0]
        accum = None
        for d in range(kd):
            frame = x[:, ts + d]
            w2d = self.weight[:, d, :, :, :]
            conv_out = mx.conv_general(frame, w2d, stride=(self.stride[1], self.stride[2]))
            accum = conv_out if accum is None else accum + conv_out
        outputs.append(accum + self.bias)
    return mx.stack(outputs, axis=1)


def patch_conv3d_instances(module):
    """Replace __call__ on all CausalConv3d instances with adaptive version."""
    from mlx_video.models.wan_2.vae22 import CausalConv3d
    patched = 0
    for name, child in module.named_modules():
        if isinstance(child, CausalConv3d):
            child.__call__ = MethodType(adaptive_conv3d_forward, child)
            patched += 1
    return patched


def main():
    from mlx_video.models.wan_2.vae22 import denormalize_latents
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.config import WanModelConfig
    import json as _json

    print("=== VAE Adaptive Conv3d Full Decode ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))

    N = 3
    results = {}

    latents = mx.random.normal((1, 11, 30, 52, 48), dtype=mx.bfloat16)
    latents = denormalize_latents(latents)
    mx.eval(latents); mx.synchronize()

    # 1. Baseline
    print("--- Baseline ---")
    vae = load_vae_decoder(MODEL / "vae.safetensors", config)
    _ = vae(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r_orig = vae(latents)
        mx.eval(r_orig); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["baseline"] = sum(times) / N
    print(f"  {results['baseline']*1000:.0f}ms")

    # 2. Adaptive
    print("\n--- Adaptive Conv3d ---")
    vae2 = load_vae_decoder(MODEL / "vae.safetensors", config)
    patched = patch_conv3d_instances(vae2)
    print(f"  Patched {patched} CausalConv3d instances")

    _ = vae2(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r_fast = vae2(latents)
        mx.eval(r_fast); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["adaptive"] = sum(times) / N
    print(f"  {results['adaptive']*1000:.0f}ms")

    # Correctness
    diff = mx.max(mx.abs(r_orig - r_fast)).item()
    print(f"  Max diff: {diff:.6f} {'OK' if diff < 0.01 else 'FAIL'}")

    # Summary
    print(f"\n{'='*70}")
    print("VAE ADAPTIVE CONV3D RESULTS")
    print(f"{'='*70}")
    baseline = results["baseline"]
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        pct = (1 - v / baseline) * 100
        print(f"  {k:<25} {v*1000:>6.0f}ms  ({pct:>+6.1f}%)")

    out_path = "artifacts/bottleneck_bench/vae_adaptive.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"strategy": k, "time_ms": round(v*1000, 1),
                               "vs_baseline_pct": round((1-v/baseline)*100, 1)}) + "\n")


if __name__ == "__main__":
    main()
