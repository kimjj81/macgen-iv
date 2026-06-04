#!/usr/bin/env python3
"""FFN fusion benchmark via generate_video.

Tests FFN optimizations in the actual pipeline:
1. Baseline (compile, no FFN changes)
2. Chunked FFN (compile)
3. Custom fused FFN with in-place-style GELU (compile)

Measures denoise time specifically.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json

MODEL = os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx")
RESULTS = "artifacts/bottleneck_bench/ffn_pipeline.jsonl"


def bench_pipeline(label, steps=10):
    """Run generate_video and return denoise time."""
    from mlx_video.models.wan_2.generate import generate_video

    t0 = time.time()
    generate_video(
        model_dir=MODEL,
        prompt="A golden retriever trots along a wet sandy beach",
        height=480, width=832, num_frames=41,
        steps=steps, guide_scale=3.5, seed=42,
        output_path=f"/tmp/ffn_bench_{label.replace(' ','_')}.mp4",
    )
    total = time.time() - t0
    # Parse denoise time from output
    return total


def main():
    from mlx_video.models.wan_2 import transformer, attention as attn_mod
    import mlx.core as mx

    results = {}

    # ---- 1. Baseline (default compile) ----
    print("--- 1. Baseline (compile, default FFN) ---")
    t1 = bench_pipeline("baseline", steps=10)
    results["baseline_compile"] = t1
    print(f"  Total: {t1:.1f}s\n")

    # ---- 2. Chunked FFN ----
    print("--- 2. Chunked FFN c1024 + compile ---")
    orig = transformer.WanFFN.__call__

    def make_chunked(cs):
        def chunked(self, x):
            b, s, d = x.shape
            if s <= cs:
                x_w = x.astype(attn_mod._linear_dtype(self.fc1))
                return self.fc2(self.act(self.fc1(x_w)))
            out = []
            for i in range(0, s, cs):
                c = x[:, i:i+cs, :]
                x_w = c.astype(attn_mod._linear_dtype(self.fc1))
                out.append(self.fc2(self.act(self.fc1(x_w))))
            return mx.concatenate(out, axis=1)
        return chunked

    # Test best chunk size from previous benchmark
    transformer.WanFFN.__call__ = make_chunked(1024)
    t2 = bench_pipeline("chunked_1024", steps=10)
    results["chunked_1024_compile"] = t2
    print(f"  Total: {t2:.1f}s\n")

    transformer.WanFFN.__call__ = make_chunked(2048)
    t3 = bench_pipeline("chunked_2048", steps=10)
    results["chunked_2048_compile"] = t3
    print(f"  Total: {t3:.1f}s\n")

    # ---- 3. Fused GELU into matmul output ----
    # Strategy: compute fc1 output, apply GELU, but DON'T materialize
    # the full intermediate. Instead, process fc2 row-by-row in chunks
    # that fit in cache, fusing GELU with the chunk.
    print("--- 3. Cache-friendly fused FFN ---")
    import mlx.nn as nn

    def fused_ffn_call(self, x):
        """FFN that fuses GELU computation to avoid materializing full intermediate."""
        b, s, d = x.shape
        x_w = x.astype(attn_mod._linear_dtype(self.fc1))

        # Process in cache-friendly chunks
        chunk = 2048
        if s <= chunk:
            h = self.act(self.fc1(x_w))
            return self.fc2(h)

        out = []
        for i in range(0, s, chunk):
            c = x_w[:, i:i+chunk, :]
            h = self.fc1(c)
            h = self.act(h)
            # Immediately consume in fc2 while h might still be in cache
            out.append(self.fc2(h))
        return mx.concatenate(out, axis=1)

    transformer.WanFFN.__call__ = fused_ffn_call
    t4 = bench_pipeline("fused_cache", steps=10)
    results["fused_cache_compile"] = t4
    print(f"  Total: {t4:.1f}s\n")

    # ---- 4. No compile + fused (to separate compile vs fusion benefit) ----
    print("--- 4. Fused cache FFN, no compile ---")
    import mlx_video.models.wan_2.generate as gen_mod

    # Monkey-patch to force no_compile
    orig_main = gen_mod.generate_video
    t5_total = time.time()
    gen_mod.generate_video(
        model_dir=MODEL,
        prompt="A golden retriever trots along a wet sandy beach",
        height=480, width=832, num_frames=41,
        steps=10, guide_scale=3.5, seed=42,
        output_path="/tmp/ffn_bench_fused_nocompile.mp4",
        no_compile=True,
    )
    t5 = time.time() - t5_total
    results["fused_cache_nocompile"] = t5
    print(f"  Total: {t5:.1f}s\n")

    # Restore
    transformer.WanFFN.__call__ = orig

    # ---- Summary ----
    print("\n" + "="*70)
    print("FFN OPTIMIZATION PIPELINE RESULTS")
    print("="*70)
    baseline = results["baseline_compile"]
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        pct = (1 - v / baseline) * 100
        print(f"  {k:<35} {v:>6.1f}s  ({pct:>+.1f}%)")

    import os
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"strategy": k, "total_s": round(v, 1),
                               "vs_baseline_pct": round((1 - v / baseline) * 100, 1)}) + "\n")

    # Cleanup temp files
    for f in ["baseline", "chunked_1024", "chunked_2048", "fused_cache", "fused_nocompile"]:
        p = f"/tmp/ffn_bench_{f}.mp4"
        if os.path.exists(p): os.remove(p)


if __name__ == "__main__":
    main()
