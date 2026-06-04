#!/usr/bin/env python3
"""VAE decode optimization: test removing mx.eval() barriers.

The VAE decoder has multiple mx.eval(x) calls to limit graph size.
This may hurt performance by preventing MLX from overlapping ops.

Test: monkey-patch decoder to remove all mx.eval() and compare.
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
    from mlx_video.models.wan_2 import vae22
    import json as _json, types

    print("=== VAE Decode: mx.eval() Removal Test ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))
    vae = load_vae_decoder(MODEL / "vae.safetensors", config)

    latents = mx.random.normal((1, 11, 30, 52, 48), dtype=mx.bfloat16)
    latents = denormalize_latents(latents)
    mx.eval(latents); mx.synchronize()

    N = 3
    results = {}

    # 1. Baseline (original with mx.eval)
    print("--- Baseline (original) ---")
    _ = vae(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = vae(latents)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["baseline"] = sum(times) / N
    print(f"  {results['baseline']*1000:.0f}ms")

    # 2. Patch: remove mx.eval() from Decoder3d.__call__
    print("\n--- No mx.eval() in Decoder3d.__call__ ---")

    # Save original
    orig_decoder_call = vae22.Decoder3d.__call__

    def decoder_no_eval(self, x, first_chunk=False):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        # NO mx.eval(x) here
        for i, layer in enumerate(self.upsamples):
            x = layer(x, first_chunk)
            # NO mx.eval(x) here
        x = self.head(x)
        return x

    vae22.Decoder3d.__call__ = decoder_no_eval
    # Reload vae to pick up changes
    vae2 = load_vae_decoder(MODEL / "vae.safetensors", config)

    _ = vae2(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = vae2(latents)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["no_eval_decoder"] = sum(times) / N
    print(f"  {results['no_eval_decoder']*1000:.0f}ms")

    # 3. Patch: remove mx.eval() from Up_ResidualBlock too
    print("\n--- No mx.eval() anywhere ---")

    orig_up_call = vae22.Up_ResidualBlock.__call__

    def up_no_eval(self, x, first_chunk=False):
        x_main = x
        for module in self.upsamples:
            if type(module).__name__ == "Resample":
                x_main = module(x_main, first_chunk)
            else:
                x_main = module(x_main)
            # NO mx.eval(x_main)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            # NO mx.eval(x_shortcut)
            return x_main + x_shortcut
        return x_main

    vae22.Up_ResidualBlock.__call__ = up_no_eval

    # Also remove from ResidualBlockLayers
    orig_rbl_call = vae22.ResidualBlockLayers.__call__

    def rbl_no_eval(self, x, feat_cache=None, feat_idx=None):
        x = self.layer_0(x)
        x = mx.silu(x)
        if feat_cache is not None:
            x = self._conv_with_cache(self.layer_2, x, feat_cache, feat_idx)
        else:
            x = self.layer_2(x)
        # NO mx.eval(x)
        x = self.layer_3(x)
        x = mx.silu(x)
        if feat_cache is not None:
            x = self._conv_with_cache(self.layer_6, x, feat_cache, feat_idx)
        else:
            x = self.layer_6(x)
        return x

    vae22.ResidualBlockLayers.__call__ = rbl_no_eval

    vae3 = load_vae_decoder(MODEL / "vae.safetensors", config)

    _ = vae3(latents); mx.eval(_); mx.synchronize()
    times = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        r = vae3(latents)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)
    results["no_eval_all"] = sum(times) / N
    print(f"  {results['no_eval_all']*1000:.0f}ms")

    # Restore
    vae22.Decoder3d.__call__ = orig_decoder_call
    vae22.Up_ResidualBlock.__call__ = orig_up_call
    vae22.ResidualBlockLayers.__call__ = orig_rbl_call

    # Summary
    print(f"\n{'='*70}")
    print("MX.EVAL REMOVAL RESULTS")
    print(f"{'='*70}")
    baseline = results["baseline"]
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        pct = (1 - v / baseline) * 100
        print(f"  {k:<25} {v*1000:>6.0f}ms  ({pct:>+6.1f}%)")

    out_path = "artifacts/bottleneck_bench/vae_eval_removal.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"strategy": k, "time_ms": round(v*1000, 1),
                               "vs_baseline_pct": round((1-v/baseline)*100, 1)}) + "\n")


if __name__ == "__main__":
    main()
