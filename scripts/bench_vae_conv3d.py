#!/usr/bin/env python3
"""VAE CausalConv3d optimization benchmark.

Current implementation decomposes 3D conv into per-frame 2D convs:
  for t in range(T_out):  # 41 iterations
      for d in range(kd):  # 3 iterations
          conv2d(frame, weight)

This is extremely slow because of:
1. 123 individual conv2d calls (41 × 3)
2. No batching = poor GPU utilization
3. Python loop overhead

Optimization: batch all frames into a single conv2d call.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


class FastCausalConv3d(nn.Module):
    """Optimized CausalConv3d using batched 2D conv instead of per-frame loop."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self._causal_pad_t = 2 * padding[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        # Same weight layout as original: [O, D, H, W, I]
        self.weight = mx.zeros(
            (out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x, cache_x=None):
        B, T, H, W, C = x.shape
        kd, kh, kw = self.kernel_size

        # 1x1x1 shortcut
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
        H_padded, W_padded = x.shape[2], x.shape[3]
        T_out = (T_padded - kd) // self.stride[0] + 1

        # OPTIMIZATION: Batch all frames into single conv2d
        # Collect input windows: for each output frame t, gather kd input frames
        # Stack all into [B * T_out * kd, H, W, C] for batched conv2d
        windows = []
        for t in range(T_out):
            t_start = t * self.stride[0]
            for d in range(kd):
                windows.append(x[:, t_start + d])  # [B, H_padded, W_padded, C]

        # Stack: [B * T_out * kd, H_padded, W_padded, C]
        # Each window[i] is [B, H, W, C], stacking gives [T_out*kd, B, H, W, C]
        batched = mx.stack(windows, axis=0)  # [T_out*kd, B, H_padded, W_padded, C]
        BT = T_out * kd * B
        batched = batched.reshape(BT, H_padded, W_padded, C)

        # Single batched conv2d with weight for kernel position d
        # Need to expand weight for each d position
        # weight shape: [O, kd, kh, kw, I]
        # For each d, use weight[:, d] → [O, kh, kw, I]
        # We need to repeat the weight kd times for T_out frames
        # Actually, the weight for each window depends on its d index

        # Split by kernel depth position, conv each, then sum
        results = []
        for d in range(kd):
            # All windows at depth position d: indices d, d+kd, d+2*kd, ...
            idx = list(range(d, T_out * kd, kd))  # T_out indices for this d
            d_batch = batched[idx]  # [T_out * B, H_padded, W_padded, C]
            w2d = self.weight[:, d, :, :, :]  # [O, kh, kw, I]
            conv_out = mx.conv_general(
                d_batch, w2d,
                stride=(self.stride[1], self.stride[2])
            )  # [T_out * B, H_out, W_out, O]
            results.append(conv_out.reshape(T_out, B, conv_out.shape[1], conv_out.shape[2], -1))

        # Sum over depth dimension: each result[t] contributes to output frame t
        output = results[0]
        for d in range(1, kd):
            output = output + results[d]

        # output: [T_out, B, H_out, W_out, O] → [B, T_out, H_out, W_out, O]
        output = output.transpose(1, 0, 2, 3, 4)
        output = output + self.bias

        return output


def benchmark_conv3d():
    """Benchmark original vs optimized CausalConv3d."""
    from mlx_video.models.wan_2.vae22 import CausalConv3d as OrigCausalConv3d

    # Test with upsample_3 ResidualBlock dimensions
    # Conv: in=512, out=256, kernel=3x3x3, padding=1
    in_ch, out_ch = 512, 256
    kd, kh, kw = 3, 3, 3

    orig_conv = OrigCausalConv3d(in_ch, out_ch, 3, padding=1)
    fast_conv = FastCausalConv3d(in_ch, out_ch, 3, padding=1)
    # Copy weights
    fast_conv.weight = orig_conv.weight
    fast_conv.bias = orig_conv.bias

    # Input: [1, 41, 240, 416, 512] — upsample_3 ResBlock_0 size
    x = mx.random.normal((1, 41, 240, 416, in_ch), dtype=mx.bfloat16)
    mx.eval(x, orig_conv.weight, fast_conv.weight)
    mx.synchronize()

    print("=== CausalConv3d Optimization Benchmark ===\n")
    print(f"Input: {x.shape} ({x.nbytes/1e6:.0f}MB)")

    # Warmup
    _ = orig_conv(x); mx.eval(_); mx.synchronize()
    _ = fast_conv(x); mx.eval(_); mx.synchronize()

    N = 3
    results = {}

    # Original
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        y_orig = orig_conv(x)
        mx.eval(y_orig); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["original_per_frame_loop"] = sum(times) / N
    print(f"Original (per-frame loop): {results['original_per_frame_loop']*1000:.0f}ms")

    # Optimized
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        y_fast = fast_conv(x)
        mx.eval(y_fast); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["batched_conv2d"] = sum(times) / N
    print(f"Optimized (batched conv2d): {results['batched_conv2d']*1000:.0f}ms")

    # Correctness check
    max_diff = mx.max(mx.abs(y_orig - y_fast)).item()
    print(f"\nMax diff: {max_diff:.6f} {'OK' if max_diff < 0.01 else 'FAIL'}")

    speedup = results["original_per_frame_loop"] / results["batched_conv2d"]
    print(f"Speedup: {speedup:.2f}x")

    # Also test smaller size: upsample_1 (60×104)
    print("\n--- Smaller test: [1, 21, 60, 104, 1024] ---")
    in_ch2, out_ch2 = 1024, 1024
    orig_conv2 = OrigCausalConv3d(in_ch2, out_ch2, 3, padding=1)
    fast_conv2 = FastCausalConv3d(in_ch2, out_ch2, 3, padding=1)
    fast_conv2.weight = orig_conv2.weight
    fast_conv2.bias = orig_conv2.bias

    x2 = mx.random.normal((1, 21, 60, 104, in_ch2), dtype=mx.bfloat16)
    mx.eval(x2, fast_conv2.weight); mx.synchronize()

    _ = orig_conv2(x2); mx.eval(_); mx.synchronize()
    _ = fast_conv2(x2); mx.eval(_); mx.synchronize()

    for label, conv_fn in [("original", orig_conv2), ("batched", fast_conv2)]:
        times = []
        for _ in range(N):
            mx.synchronize(); t0 = time.perf_counter()
            y = conv_fn(x2)
            mx.eval(y); mx.synchronize()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / N
        results[f"small_{label}"] = avg
        print(f"  {label}: {avg*1000:.0f}ms")

    speedup2 = results["small_original"] / results["small_batched"]
    print(f"  Speedup: {speedup2:.2f}x")

    # Summary
    print(f"\n{'='*70}")
    print("CAUSAL CONV3D OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:<40} {v*1000:>6.0f}ms")

    out_path = "artifacts/bottleneck_bench/vae_conv3d_opt.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"test": k, "time_ms": round(v*1000, 1)}) + "\n")


if __name__ == "__main__":
    benchmark_conv3d()
