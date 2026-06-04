#!/usr/bin/env python3
"""VAE CausalConv3d optimization v2: chunked batched conv2d.

Key insight: batching ALL frames causes OOM at large resolutions.
Solution: process temporal chunks of 4-8 frames at a time.
Each chunk is a single batched conv2d, much faster than per-frame.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


class ChunkedCausalConv3d(nn.Module):
    """Optimized CausalConv3d: chunked temporal batching."""

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
        self.weight = mx.zeros(
            (out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x, cache_x=None, chunk_size=8):
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

        # Process in temporal chunks
        output_frames = []
        for chunk_start in range(0, T_out, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T_out)
            ct = chunk_end - chunk_start  # frames in this chunk

            # Gather input windows for this chunk
            # For each output frame, need kd input frames
            windows = []
            for t in range(chunk_start, chunk_end):
                t_start = t * self.stride[0]
                for d in range(kd):
                    windows.append(x[:, t_start + d])

            # [ct * kd, B, H, W, C] → [ct * kd * B, H, W, C]
            batched = mx.stack(windows, axis=0).reshape(ct * kd * B, x.shape[2], x.shape[3], C)

            # Conv per depth position, sum
            chunk_outputs = []
            for d in range(kd):
                # indices for depth d: d, d+kd, d+2*kd, ...
                idx = list(range(d, ct * kd, kd))
                d_batch = batched[idx * B if B > 1 else idx]
                # Expand idx for batch
                if B > 1:
                    expanded = []
                    for i in idx:
                        expanded.extend(range(i * B, (i + 1) * B))
                    d_batch = batched[expanded]

                w2d = self.weight[:, d, :, :, :]
                conv_out = mx.conv_general(d_batch, w2d, stride=(self.stride[1], self.stride[2]))
                chunk_outputs.append(conv_out.reshape(ct, B, conv_out.shape[1], conv_out.shape[2], -1))

            chunk_result = chunk_outputs[0]
            for d in range(1, kd):
                chunk_result = chunk_result + chunk_outputs[d]

            # [ct, B, H_out, W_out, O] → collect
            output_frames.append(chunk_result)

        # [chunks, ct, B, H, W, O] → [T_out, B, H, W, O] → [B, T_out, H, W, O]
        output = mx.concatenate(output_frames, axis=0)  # [T_out, B, H, W, O]
        output = output.transpose(1, 0, 2, 3, 4) + self.bias
        return output


def main():
    from mlx_video.models.wan_2.vae22 import CausalConv3d as OrigConv

    print("=== Chunked CausalConv3d Benchmark ===\n")

    # Test sizes matching VAE decode stages
    tests = [
        ("ups3_small [1,41,240,416,512→256]", 512, 256, (1, 41, 240, 416)),
        ("ups2 [1,41,120,208,1024→512]", 1024, 512, (1, 41, 120, 208)),
        ("ups1 [1,21,60,104,1024→1024]", 1024, 1024, (1, 21, 60, 104)),
        ("ups0 [1,11,30,52,1024→1024]", 1024, 1024, (1, 11, 30, 52)),
    ]

    N = 3
    all_results = {}

    for label, in_ch, out_ch, shape in tests:
        print(f"\n--- {label} ---")
        B, T, H, W = shape

        orig = OrigConv(in_ch, out_ch, 3, padding=1)
        fast = ChunkedCausalConv3d(in_ch, out_ch, 3, padding=1)
        fast.weight = orig.weight
        fast.bias = orig.bias

        x = mx.random.normal(shape + (in_ch,), dtype=mx.bfloat16)
        mx.eval(x, fast.weight); mx.synchronize()

        # Warmup
        _ = orig(x); mx.eval(_); mx.synchronize()
        _ = fast(x, chunk_size=8); mx.eval(_); mx.synchronize()

        # Original timing
        times = []
        for _ in range(N):
            mx.synchronize(); t0 = time.perf_counter()
            y_orig = orig(x)
            mx.eval(y_orig); mx.synchronize()
            times.append(time.perf_counter() - t0)
        orig_time = sum(times) / N
        all_results[f"{label}_original"] = orig_time
        print(f"  Original: {orig_time*1000:.0f}ms")

        # Try different chunk sizes
        for cs in [4, 8, 16]:
            times = []
            for _ in range(N):
                mx.synchronize(); t0 = time.perf_counter()
                y_fast = fast(x, chunk_size=cs)
                mx.eval(y_fast); mx.synchronize()
                times.append(time.perf_counter() - t0)
            fast_time = sum(times) / N
            speedup = orig_time / fast_time
            key = f"{label}_chunk{cs}"
            all_results[key] = fast_time
            print(f"  Chunked(c={cs}): {fast_time*1000:.0f}ms ({speedup:.2f}x)")

        # Correctness
        diff = mx.max(mx.abs(y_orig - y_fast)).item()
        print(f"  Max diff: {diff:.6f} {'OK' if diff < 0.01 else 'FAIL'}")

    # Summary
    print(f"\n{'='*70}")
    print("CHUNKED CONV3D RESULTS")
    print(f"{'='*70}")
    for k, v in sorted(all_results.items(), key=lambda x: x[1]):
        print(f"  {k:<55} {v*1000:>6.0f}ms")

    out_path = "artifacts/bottleneck_bench/vae_conv3d_chunked.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in all_results.items():
            f.write(json.dumps({"test": k, "time_ms": round(v*1000, 1)}) + "\n")


if __name__ == "__main__":
    main()
