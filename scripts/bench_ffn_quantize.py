#!/usr/bin/env python3
"""FFN optimization: test quantization for the FFN layers.

MLX supports per-channel int8/int4 quantization via nn.quantize.
This reduces weight memory bandwidth by 2-4x, which can speed up
memory-bound or bandwidth-limited GEMM operations.

Tests:
1. Baseline (bf16, compile)
2. FFN-only int8 quantization (bf16 inputs/outputs, int8 weights)
3. Full model int8 quantization
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))
RESULTS = "artifacts/bottleneck_bench/ffn_quantize.jsonl"


def bench_steps(model, noise, t, ctx, seq_len, kv, rope, n=5, label=""):
    compiled = mx.compile(model)
    # Warmup + trace
    _ = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(_); mx.synchronize()

    times = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        r = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(r); mx.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"  {label}: {avg:.4f}s/step ({n} runs)")
    return avg


def main():
    from mlx_video.models.wan_2.utils import load_wan_model
    from mlx_video.models.wan_2.config import WanModelConfig
    import json as _json, math

    print("=== FFN Quantization Benchmark ===\n")

    config = WanModelConfig(**_json.loads((MODEL / "config.json").read_text()))
    F, H, W = 11, 30, 52
    seq_len = math.ceil((H * W) / (config.patch_size[1] * config.patch_size[2]) * F)

    # Load fresh model for each test to avoid quantization leaking
    results = {}

    # ---- 1. Baseline bf16 ----
    print("--- 1. Baseline (bf16, compile) ---")
    model = load_wan_model(MODEL / "model.safetensors", config)
    noise = mx.random.normal((48, F, H, W), dtype=mx.bfloat16)
    t = mx.array([999.0, 999.0], dtype=mx.float32)
    ctx_list = [mx.random.normal((config.text_len, config.t5_dim), dtype=mx.bfloat16) for _ in range(2)]
    ctx = model.embed_text(ctx_list)
    grid_sizes = [(F, H // config.patch_size[1], W // config.patch_size[2])] * 2
    kv = model.prepare_cross_kv(ctx)
    rope = model.prepare_rope(grid_sizes)
    mx.eval(noise, t, ctx, kv, rope); mx.synchronize()

    results["bf16"] = bench_steps(model, noise, t, ctx, seq_len, kv, rope, label="bf16")

    # ---- 2. FFN-only int8 quantization ----
    print("\n--- 2. FFN int8 quantization (rest bf16) ---")

    # Quantize only the FFN layers
    def ffn_quantize_predicate(path, module):
        # Only quantize Linear layers inside FFN blocks
        return isinstance(module, nn.Linear) and "ffn" in path

    nn.quantize(model, group_size=64, bits=4, class_predicate=ffn_quantize_predicate)
    mx.eval(model.parameters()); mx.synchronize()

    # Count quantized params
    def _flat_params(d):
        for v in (d.values() if isinstance(d, dict) else d):
            if isinstance(v, dict):
                yield from _flat_params(v)
            elif isinstance(v, list):
                for item in v:
                    if hasattr(item, 'size'):
                        yield item
            elif hasattr(v, 'size'):
                yield v

    total_params = sum(p.size for p in _flat_params(model.parameters()))
    quant_params = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            quant_params += mod.weight.size
    print(f"  Quantized {quant_params/1e6:.1f}M / {total_params/1e6:.1f}M params ({100*quant_params/total_params:.0f}%)")

    results["ffn_int4_g64"] = bench_steps(model, noise, t, ctx, seq_len, kv, rope, label="ffn_int4_g64")

    # ---- 3. Full model int4 quantization ----
    print("\n--- 3. Full model int4 quantization ---")
    del model
    mx.clear_cache()

    model = load_wan_model(MODEL / "model.safetensors", config)
    ctx = model.embed_text(ctx_list)
    kv = model.prepare_cross_kv(ctx)
    rope = model.prepare_rope(grid_sizes)
    mx.eval(ctx, kv, rope); mx.synchronize()

    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters()); mx.synchronize()

    results["full_int4_g64"] = bench_steps(model, noise, t, ctx, seq_len, kv, rope, label="full_int4_g64")

    # ---- 4. FFN int8 (group_size=128) ----
    print("\n--- 4. FFN int8 quantization (group_size=128) ---")
    del model
    mx.clear_cache()

    model = load_wan_model(MODEL / "model.safetensors", config)
    ctx = model.embed_text(ctx_list)
    kv = model.prepare_cross_kv(ctx)
    rope = model.prepare_rope(grid_sizes)
    mx.eval(ctx, kv, rope); mx.synchronize()

    nn.quantize(model, group_size=128, bits=8, class_predicate=ffn_quantize_predicate)
    mx.eval(model.parameters()); mx.synchronize()

    results["ffn_int8_g128"] = bench_steps(model, noise, t, ctx, seq_len, kv, rope, label="ffn_int8_g128")

    # ---- Summary ----
    print("\n" + "="*70)
    print("QUANTIZATION RESULTS")
    print("="*70)
    baseline = results["bf16"]
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        pct = (1 - v / baseline) * 100
        print(f"  {k:<25} {v:>7.4f}s/step  ({pct:>+6.1f}%)")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"strategy": k, "time_s": round(v, 4),
                               "speedup_pct": round((1 - v / baseline) * 100, 1)}) + "\n")


if __name__ == "__main__":
    main()
