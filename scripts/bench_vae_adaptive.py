#!/usr/bin/env python3
"""VAE full decode benchmark: adaptive chunked conv3d + mx.eval optimization.

Strategy:
- For small resolutions (≤120×208): use batched conv3d with chunk_size=8
- For large resolutions (≥240×416): use original per-frame conv3d
- Keep mx.eval() barriers for memory management
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import math
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


class AdaptiveCausalConv3d(nn.Module):
    """CausalConv3d with adaptive batching: batched for small, per-frame for large."""

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

    def __call__(self, x, cache_x=None):
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

        # Adaptive: batch for small spatial dims, per-frame for large
        # Threshold: H*W <= 120*208 = 24960 → batched
        spatial_size = x.shape[2] * x.shape[3]
        use_batch = spatial_size <= 25000

        if use_batch:
            return self._batched_forward(x, B, T_out, kd)
        else:
            return self._per_frame_forward(x, B, T_out, kd)

    def _batched_forward(self, x, B, T_out, kd):
        chunk_size = 8
        output_frames = []
        H_p, W_p = x.shape[2], x.shape[3]
        C = x.shape[4]

        for chunk_start in range(0, T_out, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T_out)
            ct = chunk_end - chunk_start

            windows = []
            for t in range(chunk_start, chunk_end):
                t_start = t * self.stride[0]
                for d in range(kd):
                    windows.append(x[:, t_start + d])

            batched = mx.stack(windows, axis=0).reshape(
                ct * kd * B, H_p, W_p, C
            )

            chunk_result = None
            for d in range(kd):
                idx = []
                for i in range(d, ct * kd, kd):
                    for b in range(B):
                        idx.append(i * B + b)
                d_batch = batched[idx]
                w2d = self.weight[:, d, :, :, :]
                conv_out = mx.conv_general(
                    d_batch, w2d,
                    stride=(self.stride[1], self.stride[2])
                )
                conv_out = conv_out.reshape(ct, B, conv_out.shape[1], conv_out.shape[2], -1)
                chunk_result = conv_out if chunk_result is None else chunk_result + conv_out

            output_frames.append(chunk_result)

        output = mx.concatenate(output_frames, axis=0)
        output = output.transpose(1, 0, 2, 3, 4) + self.bias
        return output

    def _per_frame_forward(self, x, B, T_out, kd):
        # Original per-frame approach
        outputs = []
        for t in range(T_out):
            t_start = t * self.stride[0]
            accum = None
            for d in range(kd):
                frame = x[:, t_start + d]
                w2d = self.weight[:, d, :, :, :]
                conv_out = mx.conv_general(
                    frame, w2d,
                    stride=(self.stride[1], self.stride[2])
                )
                accum = conv_out if accum is None else accum + conv_out
            outputs.append(accum + self.bias)
        return mx.stack(outputs, axis=1)


def main():
    from mlx_video.models.wan_2.vae22 import (
        CausalConv3d, ResidualBlock, ResidualBlockLayers,
        Up_ResidualBlock, Decoder3d, Wan22VAEDecoder,
        denormalize_latents, Head22
    )
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.config import WanModelConfig
    import json as _json

    print("=== VAE Full Decode: Adaptive Conv3d ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))
    vae = load_vae_decoder(MODEL / "vae.safetensors", config)

    latents = mx.random.normal((1, 11, 30, 52, 48), dtype=mx.bfloat16)
    latents = denormalize_latents(latents)
    mx.eval(latents); mx.synchronize()

    N = 3
    results = {}

    # 1. Baseline
    print("--- Baseline ---")
    _ = vae(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = vae(latents)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["baseline"] = sum(times) / N
    print(f"  {results['baseline']*1000:.0f}ms")

    # 2. Monkey-patch CausalConv3d with adaptive version
    print("\n--- Adaptive Conv3d (batched for small, per-frame for large) ---")

    # Replace CausalConv3d globally
    old_init = CausalConv3d.__init__
    old_call = CausalConv3d.__call__

    def new_init(self, in_ch, out_ch, ks, stride=1, padding=0):
        AdaptiveCausalConv3d.__init__(self, in_ch, out_ch, ks, stride, padding)

    def new_call(self, x, cache_x=None):
        return AdaptiveCausalConv3d.__call__(self, x, cache_x)

    CausalConv3d.__init__ = new_init
    CausalConv3d.__call__ = new_call

    vae2 = load_vae_decoder(MODEL / "vae.safetensors", config)

    _ = vae2(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = vae2(latents)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["adaptive_conv3d"] = sum(times) / N
    print(f"  {results['adaptive_conv3d']*1000:.0f}ms")

    # Restore
    CausalConv3d.__init__ = old_init
    CausalConv3d.__call__ = old_call

    # Check correctness
    diff = mx.max(mx.abs(r - vae(latents))).item()
    print(f"  Max diff: {diff:.6f}")

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
