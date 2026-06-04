#!/usr/bin/env python3
"""Wan2.2 FFN optimization benchmark.

Tests strategies for speeding up the FFN (43.9% of denoise time):
1. Baseline (no compile)
2. Model-level compile
3. Chunked FFN (better cache locality) + compile
4. Per-FFN compiled forward

Reports denoise time for each approach.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import gc
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL_PATH = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))
RESULTS_PATH = "artifacts/bottleneck_bench/ffn_optimization.jsonl"


def load_model():
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.wan_2 import WanModel
    from mlx_video.models.wan_2.utils import load_wan_model

    config_path = MODEL_PATH / "config.json"
    import json as _json
    config = WanModelConfig(**_json.loads(config_path.read_text()))
    model = load_wan_model(MODEL_PATH / "model.safetensors", config)
    return model, config


def make_inputs(model, config):
    """Create dummy inputs matching 832x480 41f."""
    # Latent: 832x480, 41f → (48, 11, 30, 52)
    F, H, W = 11, 30, 52
    # Actual seq_len after patchify (patch_size [1,2,2])
    import math
    seq_len = math.ceil((H * W) / (config.patch_size[1] * config.patch_size[2]) * F)

    noise = mx.random.normal((48, F, H, W), dtype=mx.bfloat16)
    t = mx.array([999.0, 999.0], dtype=mx.float32)

    # Dummy text context — text_embedding_0 expects t5_dim (4096) input
    text_len = config.text_len
    t5_dim = config.t5_dim  # 4096 — actual T5 encoder output dim
    context_list = [mx.random.normal((text_len, t5_dim), dtype=mx.bfloat16) for _ in range(2)]
    context_embedded = model.embed_text(context_list)

    # Precompute cross_kv and rope — use grid_sizes after patchify
    # patch_size=[1,2,2] → grid = (F, H//2, W//2) = (11, 15, 26)
    grid_sizes = [(F, H // config.patch_size[1], W // config.patch_size[2])] * 2
    cross_kv = model.prepare_cross_kv(context_embedded)
    rope_cos_sin = model.prepare_rope(grid_sizes)

    mx.eval(noise, t, context_embedded, cross_kv, rope_cos_sin)
    mx.synchronize()

    return noise, t, context_embedded, seq_len, cross_kv, rope_cos_sin


def bench_forward(label, forward_fn, noise, t, ctx, seq_len, kv, rope, n=5):
    """Benchmark a forward function."""
    # Warmup
    r = forward_fn([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(r)
    mx.synchronize()

    times = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        r = forward_fn([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(r)
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    std = (sum((x - avg)**2 for x in times) / len(times)) ** 0.5
    print(f"  {label}: {avg:.4f}s ± {std:.4f}s ({n} runs)")
    return avg


def main():
    print("=== FFN Optimization Benchmark ===\n")

    # Load model
    print("Loading model...")
    model, config = load_model()
    noise, t, ctx, seq_len, kv, rope = make_inputs(model, config)
    print(f"  seq_len={seq_len}, dim={model.dim}, blocks={len(model.blocks)}")
    print(f"  FFN dim: {model.blocks[0].ffn.fc1.weight.shape}\n")

    results = {}

    # ---- Strategy 1: Baseline (no compile) ----
    print("--- 1. Baseline (no compile) ---")
    results["baseline"] = bench_forward("baseline", model, noise, t, ctx, seq_len, kv, rope)

    # ---- Strategy 2: Model-level compile ----
    print("\n--- 2. Model-level mx.compile ---")
    compiled = mx.compile(model)
    # Trace
    _ = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(_); mx.synchronize()
    results["model_compile"] = bench_forward("model_compile", compiled, noise, t, ctx, seq_len, kv, rope)

    # ---- Strategy 3: Chunked FFN + compile ----
    print("\n--- 3. Chunked FFN + compile ---")
    from mlx_video.models.wan_2 import transformer, attention as attn_mod

    orig_ffn = transformer.WanFFN.__call__

    for chunk_size in [512, 1024, 2048]:
        transformer.WanFFN.__call__ = orig_ffn  # reset

        def make_chunked(cs):
            def chunked_call(self, x):
                b, s, d = x.shape
                if s <= cs:
                    x_w = x.astype(attn_mod._linear_dtype(self.fc1))
                    return self.fc2(self.act(self.fc1(x_w)))
                outputs = []
                for i in range(0, s, cs):
                    c = x[:, i:i+cs, :]
                    x_w = c.astype(attn_mod._linear_dtype(self.fc1))
                    outputs.append(self.fc2(self.act(self.fc1(x_w))))
                return mx.concatenate(outputs, axis=1)
            return chunked_call

        transformer.WanFFN.__call__ = make_chunked(chunk_size)
        compiled_c = mx.compile(model)
        _ = compiled_c([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(_); mx.synchronize()

        key = f"chunked_c{chunk_size}_compile"
        results[key] = bench_forward(key, compiled_c, noise, t, ctx, seq_len, kv, rope)

    # ---- Strategy 4: Per-FFN compiled ----
    print("\n--- 4. Per-FFN mx.compile ---")
    transformer.WanFFN.__call__ = orig_ffn  # reset

    # First compilation on first block, reuse pattern
    _compiled_ffn_cache = {}

    def per_ffn_compiled_call(self, x):
        fid = id(self)
        if fid not in _compiled_ffn_cache:
            fc1 = self.fc1
            fc2 = self.fc2
            act = self.act

            @mx.compile
            def _fwd(x):
                x_w = x.astype(attn_mod._linear_dtype(fc1))
                return fc2(act(fc1(x_w)))

            _compiled_ffn_cache[fid] = _fwd
        return _compiled_ffn_cache[fid](x)

    transformer.WanFFN.__call__ = per_ffn_compiled_call

    # No model-level compile — each FFN is individually compiled
    _ = model([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(_); mx.synchronize()

    results["per_ffn_compile"] = bench_forward("per_ffn_compile", model, noise, t, ctx, seq_len, kv, rope)

    # ---- Cleanup ----
    transformer.WanFFN.__call__ = orig_ffn
    _compiled_ffn_cache.clear()

    # ---- Summary ----
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    baseline = results["baseline"]
    print(f"{'Strategy':<30} | {'Time/step':>10} | {'vs baseline':>12}")
    print("-" * 60)
    best_k = min(results, key=results.get)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        pct = (1 - v / baseline) * 100
        marker = " <-- BEST" if k == best_k else ""
        print(f"{k:<30} | {v:>9.4f}s | {pct:>+10.1f}%{marker}")

    # Save
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({
                "strategy": k,
                "time_per_step_s": round(v, 4),
                "speedup_pct": round((1 - v / baseline) * 100, 1),
            }) + "\n")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
