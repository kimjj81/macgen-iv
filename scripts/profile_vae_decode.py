#!/usr/bin/env python3
"""Wan2.2 VAE decode profiling: identify bottlenecks in VAE decode pipeline.

Current VAE decode takes 35-50s (40%+ of total pipeline).
This script instruments each decoder layer to find where time goes.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


def main():
    from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder, denormalize_latents
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.config import WanModelConfig
    import json as _json

    print("=== VAE Decode Profiling ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))
    vae = load_vae_decoder(MODEL / "vae.safetensors", config)

    # Simulate denoised latents: [1, 11, 30, 52, 48] (channels-last)
    latents = mx.random.normal((1, 11, 30, 52, 48), dtype=mx.bfloat16)
    latents = denormalize_latents(latents)
    mx.eval(latents); mx.synchronize()

    print(f"Latent shape: {latents.shape}")
    print(f"Expected output: [1, 41, 480, 832, 3]\n")

    # Profile each decoder component
    decoder = vae.decoder
    timings = {}

    # Step 1: conv2 (1x1x1 conv)
    mx.synchronize(); t0 = time.perf_counter()
    x = vae.conv2(latents)
    mx.eval(x); mx.synchronize()
    timings["conv2"] = time.perf_counter() - t0
    print(f"conv2: {timings['conv2']*1000:.0f}ms  shape: {x.shape}")

    # Step 2: decoder.conv1
    mx.synchronize(); t0 = time.perf_counter()
    x = decoder.conv1(x)
    mx.eval(x); mx.synchronize()
    timings["dec_conv1"] = time.perf_counter() - t0
    print(f"decoder.conv1: {timings['dec_conv1']*1000:.0f}ms  shape: {x.shape}")

    # Step 3: middle blocks
    for i, layer in enumerate(decoder.middle):
        mx.synchronize(); t0 = time.perf_counter()
        x = layer(x)
        mx.eval(x); mx.synchronize()
        name = f"middle_{i}"
        if isinstance(layer, nn.Module):
            name = f"middle_{type(layer).__name__}"
        timings[name] = time.perf_counter() - t0
        print(f"  {name}: {timings[name]*1000:.0f}ms  shape: {x.shape}")

    # Step 4: upsample blocks
    for i, layer in enumerate(decoder.upsamples):
        mx.synchronize(); t0 = time.perf_counter()
        x = layer(x, first_chunk=True)
        mx.eval(x); mx.synchronize()
        name = f"upsample_{i}"
        timings[name] = time.perf_counter() - t0
        print(f"  {name}: {timings[name]*1000:.0f}ms  shape: {x.shape}")

        # Profile sub-layers of this upsample block
        if hasattr(layer, 'res_blocks'):
            # Re-run with per-layer timing
            # Reset x to before this block
            pass

    # Step 5: head
    mx.synchronize(); t0 = time.perf_counter()
    x = decoder.head(x)
    mx.eval(x); mx.synchronize()
    timings["head"] = time.perf_counter() - t0
    print(f"  head: {timings['head']*1000:.0f}ms  shape: {x.shape}")

    total = sum(timings.values())
    print(f"\nTotal: {total*1000:.0f}ms")

    # Now profile detailed upsample internals
    print("\n=== Detailed Upsample Block Profiling ===\n")

    # Re-run with sub-layer timing for each upsample block
    x = vae.conv2(latents)
    x = decoder.conv1(x)
    for layer in decoder.middle:
        x = layer(x)
    mx.eval(x); mx.synchronize()

    for i, layer in enumerate(decoder.upsamples):
        print(f"\nUpsample block {i}:")
        sub_timings = {}

        # Get sub-layers
        if hasattr(layer, 'res_blocks'):
            for j, rb in enumerate(layer.res_blocks):
                mx.synchronize(); t0 = time.perf_counter()
                x = rb(x, first_chunk=True)
                mx.eval(x); mx.synchronize()
                sub_timings[f"block_{i}_res_{j}"] = time.perf_counter() - t0
                print(f"  res_block[{j}]: {sub_timings[f'block_{i}_res_{j}']*1000:.0f}ms  shape: {x.shape}")

        if hasattr(layer, 'up') and layer.up is not None:
            mx.synchronize(); t0 = time.perf_counter()
            x = layer.up(x, first_chunk=True)
            mx.eval(x); mx.synchronize()
            sub_timings[f"block_{i}_upsample"] = time.perf_counter() - t0
            print(f"  upsample: {sub_timings[f'block_{i}_upsample']*1000:.0f}ms  shape: {x.shape}")

        if hasattr(layer, 'conv'):
            mx.synchronize(); t0 = time.perf_counter()
            x = layer.conv(x, first_chunk=True)
            mx.eval(x); mx.synchronize()
            sub_timings[f"block_{i}_conv"] = time.perf_counter() - t0
            print(f"  conv: {sub_timings[f'block_{i}_conv']*1000:.0f}ms  shape: {x.shape}")

        timings.update(sub_timings)

    # Also profile with mx.compile
    print("\n=== Compiled VAE Decode ===\n")

    @mx.compile
    def compiled_decode(z):
        return vae(z)

    # Warmup
    _ = compiled_decode(latents)
    mx.eval(_); mx.synchronize()

    N = 3
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = compiled_decode(latents)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)

    compiled_avg = sum(times) / N
    timings["compiled_full"] = compiled_avg
    print(f"Compiled VAE decode: {compiled_avg*1000:.0f}ms (avg of {N})")
    print(f"Uncompiled total: {total*1000:.0f}ms")
    print(f"Compile speedup: {(1-compiled_avg/total)*100:.1f}%")

    # ---- Summary ----
    print("\n" + "="*70)
    print("VAE DECODE PROFILE SUMMARY")
    print("="*70)
    sorted_t = sorted(timings.items(), key=lambda x: -x[1])
    for name, t in sorted_t:
        pct = 100 * t / total
        bar = "#" * int(pct / 2)
        print(f"  {name:<30} {t*1000:>6.0f}ms  ({pct:>5.1f}%) {bar}")

    # Save
    out_path = "artifacts/bottleneck_bench/vae_profile.jsonl"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in timings.items():
            f.write(json.dumps({"component": k, "time_ms": round(v*1000, 1),
                               "pct_of_total": round(100*v/total, 1)}) + "\n")


if __name__ == "__main__":
    main()
