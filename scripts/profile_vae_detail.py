#!/usr/bin/env python3
"""Wan2.2 VAE decode detailed profiling: per-operation inside upsample blocks.

Focus on upsample_2 and upsample_3 which consume 70% of VAE time.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


def main():
    from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder, denormalize_latents
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.wan_2.config import WanModelConfig
    import json as _json

    print("=== VAE Detailed Upsample Profiling ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))
    vae = load_vae_decoder(MODEL / "vae.safetensors", config)
    decoder = vae.decoder

    latents = mx.random.normal((1, 11, 30, 52, 48), dtype=mx.bfloat16)
    latents = denormalize_latents(latents)
    mx.eval(latents); mx.synchronize()

    # Run up to each upsample block and profile internals
    x = vae.conv2(latents)
    x = decoder.conv1(x)
    for layer in decoder.middle:
        x = layer(x)
    mx.eval(x); mx.synchronize()

    timings = {}

    for i, layer in enumerate(decoder.upsamples):
        print(f"\n--- Upsample block {i} ---")
        print(f"  Input shape: {x.shape}")

        # Profile each sub-module
        x_main = x
        for j, module in enumerate(layer.upsamples):
            mod_type = type(module).__name__
            mx.synchronize(); t0 = time.perf_counter()
            if mod_type == "Resample":
                x_main = module(x_main, first_chunk=True)
            else:
                x_main = module(x_main)
            mx.eval(x_main); mx.synchronize()
            dt = time.perf_counter() - t0
            key = f"ups{i}_{mod_type}_{j}"
            timings[key] = dt
            print(f"  {key}: {dt*1000:.0f}ms  shape: {x_main.shape}")

        # Profile shortcut
        if layer.avg_shortcut is not None:
            mx.synchronize(); t0 = time.perf_counter()
            x_short = layer.avg_shortcut(x, first_chunk=True)
            mx.eval(x_short); mx.synchronize()
            dt = time.perf_counter() - t0
            timings[f"ups{i}_shortcut"] = dt
            print(f"  ups{i}_shortcut: {dt*1000:.0f}ms  shape: {x_short.shape}")
            x_main = x_main + x_short

        x = x_main
        mx.eval(x); mx.synchronize()

    # head
    mx.synchronize(); t0 = time.perf_counter()
    x = decoder.head(x)
    mx.eval(x); mx.synchronize()
    timings["head"] = time.perf_counter() - t0
    print(f"\nhead: {timings['head']*1000:.0f}ms  shape: {x.shape}")

    # ---- Summary ----
    total = sum(timings.values())
    print(f"\n{'='*70}")
    print("VAE UPSAMPLE DETAIL")
    print(f"{'='*70}")
    sorted_t = sorted(timings.items(), key=lambda x: -x[1])
    for name, t in sorted_t:
        pct = 100 * t / total
        bar = "#" * int(pct / 2)
        print(f"  {name:<35} {t*1000:>6.0f}ms  ({pct:>5.1f}%) {bar}")

    out_path = "artifacts/bottleneck_bench/vae_upsample_detail.jsonl"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in timings.items():
            f.write(json.dumps({"component": k, "time_ms": round(v*1000, 1),
                               "pct": round(100*v/total, 1)}) + "\n")


if __name__ == "__main__":
    main()
