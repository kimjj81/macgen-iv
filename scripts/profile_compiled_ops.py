#!/usr/bin/env python3
"""Profile individual ops inside a compiled denoise step.

Uses mx.compile but with per-op timing via mx.synchronize().
Identifies where time actually goes after compile optimization.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import mlx.core as mx
from pathlib import Path

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx"))


def main():
    from mlx_video.models.wan_2.utils import load_wan_model
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2 import transformer, attention as attn_mod, rope as rope_mod
    import json, math

    config = WanModelConfig(**json.loads((MODEL / "config.json").read_text()))
    model = load_wan_model(MODEL / "model.safetensors", config)

    F, H, W = 11, 30, 52
    seq_len = math.ceil((H * W) / (config.patch_size[1] * config.patch_size[2]) * F)
    t5_dim = config.t5_dim
    text_len = config.text_len

    noise = mx.random.normal((48, F, H, W), dtype=mx.bfloat16)
    t = mx.array([999.0, 999.0], dtype=mx.float32)
    ctx_list = [mx.random.normal((text_len, t5_dim), dtype=mx.bfloat16) for _ in range(2)]
    ctx = model.embed_text(ctx_list)

    grid_sizes = [(F, H // config.patch_size[1], W // config.patch_size[2])] * 2
    kv = model.prepare_cross_kv(ctx)
    rope = model.prepare_rope(grid_sizes)

    mx.eval(noise, t, ctx, kv, rope)
    mx.synchronize()

    # Profile a single block's ops with synchronization
    block = model.blocks[0]

    # Patchify
    x_patches, gs = model._patchify(noise)
    x = mx.broadcast_to(x_patches, (2,) + x_patches.shape[1:])
    if x.shape[1] < seq_len:
        x = mx.concatenate([x, mx.zeros((2, seq_len - x.shape[1], model.dim), dtype=x.dtype)], axis=1)

    sinusoid = t[..., None].astype(mx.float32) * model._inv_freq
    sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)
    e = model.time_embedding_1(model.time_embedding_act(model.time_embedding_0(sin_emb)))
    e0 = model.time_projection(model.time_projection_act(e)).reshape(2, 1, 6, model.dim)

    mx.eval(x, e0)
    mx.synchronize()

    # Compile model forward
    compiled = mx.compile(model)
    _ = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
    mx.eval(_); mx.synchronize()

    # Now profile compiled model over 5 steps
    print("=== Compiled Denoise Step Profiling ===\n")
    N = 5
    times = []
    for _ in range(N):
        mx.synchronize()
        t0 = time.perf_counter()
        result = compiled([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(result)
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"Compiled model forward: {avg:.4f}s (avg of {N})")

    # Now compare without compile
    times_nc = []
    for _ in range(N):
        mx.synchronize()
        t0 = time.perf_counter()
        result = model([noise, noise], t, ctx, seq_len, cross_kv_caches=kv, rope_cos_sin=rope)
        mx.eval(result)
        mx.synchronize()
        times_nc.append(time.perf_counter() - t0)

    avg_nc = sum(times_nc) / len(times_nc)
    print(f"No compile model forward: {avg_nc:.4f}s (avg of {N})")
    print(f"Compile speedup: {(1 - avg/avg_nc)*100:.1f}%")

    # Profile what compile actually optimizes
    # Test individual operations that might be bottlenecks
    print("\n=== Individual Op Benchmarks (B=2, seq_len=4290, dim=3072) ===\n")

    # 1. FFN fc1
    fc1 = block.ffn.fc1
    x_test = x.astype(attn_mod._linear_dtype(fc1))

    @mx.compile
    def bench_fc1(x):
        return fc1(x)

    times_fc1 = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        h = bench_fc1(x_test); mx.eval(h); mx.synchronize()
        times_fc1.append(time.perf_counter() - t0)
    print(f"FFN fc1 (compiled): {sum(times_fc1)/N:.4f}s  shape: {x_test.shape} -> (2,4290,{fc1.weight.shape[0]})")

    # 2. FFN fc1+gelu+fc2
    fc2 = block.ffn.fc2
    act = block.ffn.act

    @mx.compile
    def bench_ffn_full(x):
        return fc2(act(fc1(x)))

    times_ffn = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        h = bench_ffn_full(x_test); mx.eval(h); mx.synchronize()
        times_ffn.append(time.perf_counter() - t0)
    print(f"FFN full (compiled): {sum(times_ffn)/N:.4f}s")

    # 3. SDPA (self-attn style)
    n_heads = block.self_attn.num_heads
    head_dim = block.self_attn.head_dim
    q = mx.random.normal((2, n_heads, seq_len, head_dim), dtype=mx.bfloat16)
    k = mx.random.normal((2, n_heads, seq_len, head_dim), dtype=mx.bfloat16)
    v = mx.random.normal((2, n_heads, seq_len, head_dim), dtype=mx.bfloat16)

    @mx.compile
    def bench_sdpa(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/head_dim**0.5)

    times_sdpa = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        h = bench_sdpa(q, k, v); mx.eval(h); mx.synchronize()
        times_sdpa.append(time.perf_counter() - t0)
    print(f"SDPA (compiled): {sum(times_sdpa)/N:.4f}s  shape: q={q.shape}")

    # 4. RoPE apply
    q_rope = mx.random.normal((2, seq_len, n_heads, head_dim), dtype=mx.float32)
    cos_f, sin_f = rope

    @mx.compile
    def bench_rope(q, cos_f, sin_f):
        half_d = q.shape[-1] // 2
        q_s = q[:, :seq_len].reshape(2, seq_len, n_heads, half_d, 2)
        out_real = q_s[..., 0] * cos_f - q_s[..., 1] * sin_f
        out_imag = q_s[..., 0] * sin_f + q_s[..., 1] * cos_f
        return mx.stack([out_real, out_imag], axis=-1).reshape(2, seq_len, n_heads, head_dim)

    times_rope = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        h = bench_rope(q_rope, cos_f, sin_f); mx.eval(h); mx.synchronize()
        times_rope.append(time.perf_counter() - t0)
    print(f"RoPE apply (compiled): {sum(times_rope)/N:.4f}s")

    # 5. Cross-attn SDPA (shorter K/V)
    ctx_len = ctx.shape[1]
    ck = mx.random.normal((2, n_heads, ctx_len, head_dim), dtype=mx.bfloat16)
    cv = mx.random.normal((2, n_heads, ctx_len, head_dim), dtype=mx.bfloat16)
    cq = mx.random.normal((2, n_heads, seq_len, head_dim), dtype=mx.bfloat16)

    @mx.compile
    def bench_cross_sdpa(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/head_dim**0.5)

    times_cross = []
    for _ in range(N):
        mx.synchronize(); t0 = time.perf_counter()
        h = bench_cross_sdpa(cq, ck, cv); mx.eval(h); mx.synchronize()
        times_cross.append(time.perf_counter() - t0)
    print(f"Cross-attn SDPA (compiled): {sum(times_cross)/N:.4f}s  K/V len: {ctx_len}")

    # Summary
    ffn_per_block = sum(times_ffn)/N
    sdpa_per_block = sum(times_sdpa)/N
    cross_per_block = sum(times_cross)/N
    total_per_block = ffn_per_block + sdpa_per_block + cross_per_block
    all_blocks = total_per_block * 30

    print(f"\n=== Per-Block Estimate (×30 blocks) ===")
    print(f"FFN:  {ffn_per_block*30:.4f}s ({100*ffn_per_block*30/all_blocks:.0f}%)")
    print(f"SDPA: {sdpa_per_block*30:.4f}s ({100*sdpa_per_block*30/all_blocks:.0f}%)")
    print(f"Cross:{cross_per_block*30:.4f}s ({100*cross_per_block*30/all_blocks:.0f}%)")
    print(f"Estimated total: {all_blocks:.4f}s (actual compiled: {avg:.4f}s)")


if __name__ == "__main__":
    main()
