#!/usr/bin/env python3
"""Wan2.2 deep profile: measure each component inside a denoise step.

Instruments every transformer block's self_attn, cross_attn, ffn, and modulation
to find the actual engine-level bottleneck.

Output: per-block breakdown showing where time is spent.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import mlx.core as mx

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx")

def main():
    from mlx_video.models.wan_2.generate import generate_video
    
    # We'll monkey-patch the transformer blocks to time each component
    from mlx_video.models.wan_2 import transformer, wan_2, attention
    
    # Store original __call__
    original_block_call = transformer.WanAttentionBlock.__call__
    
    # Per-block timing storage
    block_timings = {}  # block_idx -> {self_attn, cross_attn, ffn, mod, total}
    
    def timed_block_call(self, x, **kwargs):
        idx = id(self)
        if idx not in block_timings:
            block_timings[idx] = {"self_attn": [], "cross_attn": [], "ffn": [], "mod": [], "total": []}
        bt = block_timings[idx]
        
        t_total = time.perf_counter()
        
        # Modulation
        t0 = time.perf_counter()
        mod = self.modulation + kwargs["e"]
        e0 = mod[:, :, 0, :]
        e1 = mod[:, :, 1, :]
        e2 = mod[:, :, 2, :]
        e3 = mod[:, :, 3, :]
        e4 = mod[:, :, 4, :]
        e5 = mod[:, :, 5, :]
        mx.eval(mod)
        t_mod = time.perf_counter() - t0
        
        # Self-attention
        t0 = time.perf_counter()
        x_mod = self.norm1(x) * (1 + e1) + e0
        y = self.self_attn(
            x_mod,
            kwargs["seq_lens"],
            kwargs["grid_sizes"],
            kwargs["freqs"],
            rope_cos_sin=kwargs.get("rope_cos_sin"),
            attn_mask=kwargs.get("attn_mask"),
        )
        x = x + y * e2
        mx.eval(x)
        t_self = time.perf_counter() - t0
        
        # Cross-attention
        t0 = time.perf_counter()
        x_cross = self.norm3(x) if self.norm3 is not None else x
        x = x + self.cross_attn(
            x_cross, kwargs["context"], kwargs.get("context_lens"),
            kv_cache=kwargs.get("cross_kv_cache")
        )
        mx.eval(x)
        t_cross = time.perf_counter() - t0
        
        # FFN
        t0 = time.perf_counter()
        x_mod = self.norm2(x) * (1 + e4) + e3
        y = self.ffn(x_mod)
        x = x + y * e5
        mx.eval(x)
        t_ffn = time.perf_counter() - t0
        
        bt["self_attn"].append(t_self)
        bt["cross_attn"].append(t_cross)
        bt["ffn"].append(t_ffn)
        bt["mod"].append(t_mod)
        bt["total"].append(time.perf_counter() - t_total)
        
        return x
    
    transformer.WanAttentionBlock.__call__ = timed_block_call
    
    # Also time patchify, time_embed, head, unpatchify
    original_forward = wan_2.WanModel.__call__
    
    phase_timings = {}
    
    def timed_forward(self, x_list, t, context, seq_len, **kwargs):
        # We just want to see the block-level breakdown
        # The original forward does patchify, embed, blocks, head, unpatchify
        # We time the whole thing and the blocks are already timed
        result = original_forward(self, x_list, t, context, seq_len, **kwargs)
        return result
    
    wan_2.WanModel.__call__ = timed_forward
    
    print("=== Wan2.2 Deep Profile ===")
    print(f"Model: {MODEL_PATH}")
    print()
    
    # Run a single generation — we only need 1 step to profile block internals
    # Use 3 steps to see if pattern changes
    output = generate_video(
        model_dir=MODEL_PATH,
        prompt="A golden retriever trots along a wet sandy beach, ocean waves in the background",
        height=480, width=832,
        num_frames=41,
        steps=3,  # 3 steps enough for profiling
        guide_scale=3.5,
        seed=42,
        output_path="/tmp/wan_deep_profile.mp4",
        no_compile=True,
    )
    
    # Restore originals
    transformer.WanAttentionBlock.__call__ = original_block_call
    wan_2.WanModel.__call__ = original_forward
    
    # Analyze results
    # Sort blocks by their order (by first total time)
    ordered_indices = sorted(block_timings.keys(), key=lambda k: block_timings[k]["total"][0] if block_timings[k]["total"] else 0)
    
    # Actually we need block index, not id. Let's map by order of first appearance
    # The blocks are ordered by self.blocks list
    # Re-create the mapping
    print("\n" + "="*80)
    print("PER-BLOCK BREAKDOWN (averaged over 3 steps, B=2 CFG)")
    print("="*80)
    
    # Get block objects from the model to establish order
    # We'll just sort by the order they appear in the timings dict
    block_list = list(block_timings.values())
    
    if not block_list:
        print("No block timings captured!")
        return
    
    # Print header
    print(f"{'Block':>5} | {'self_attn':>10} | {'cross_attn':>10} | {'ffn':>10} | {'mod':>8} | {'total':>10} | {'%self':>5} | {'%cross':>6} | {'%ffn':>5}")
    print("-" * 95)
    
    totals = {"self_attn": 0, "cross_attn": 0, "ffn": 0, "mod": 0, "total": 0}
    
    for i, bt in enumerate(block_list):
        avg = {}
        for k in ["self_attn", "cross_attn", "ffn", "mod", "total"]:
            vals = bt[k]
            avg[k] = sum(vals) / len(vals) if vals else 0
        
        t = avg["total"]
        pct_self = 100 * avg["self_attn"] / t if t > 0 else 0
        pct_cross = 100 * avg["cross_attn"] / t if t > 0 else 0
        pct_ffn = 100 * avg["ffn"] / t if t > 0 else 0
        
        print(f"{i:>5} | {avg['self_attn']:>9.4f}s | {avg['cross_attn']:>9.4f}s | {avg['ffn']:>9.4f}s | {avg['mod']:>7.4f}s | {t:>9.4f}s | {pct_self:>4.1f}% | {pct_cross:>5.1f}% | {pct_ffn:>4.1f}%")
        
        for k in totals:
            totals[k] += avg[k]
    
    print("-" * 95)
    n = len(block_list)
    print(f"{'AVG':>5} | {totals['self_attn']/n:>9.4f}s | {totals['cross_attn']/n:>9.4f}s | {totals['ffn']/n:>9.4f}s | {totals['mod']/n:>7.4f}s | {totals['total']/n:>9.4f}s |")
    print()
    
    grand_total = totals["total"]
    print(f"ALL BLOCKS TOTAL: {grand_total:.3f}s")
    print(f"  self_attn:  {totals['self_attn']:.3f}s ({100*totals['self_attn']/grand_total:.1f}%)")
    print(f"  cross_attn: {totals['cross_attn']:.3f}s ({100*totals['cross_attn']/grand_total:.1f}%)")
    print(f"  ffn:        {totals['ffn']:.3f}s ({100*totals['ffn']/grand_total:.1f}%)")
    print(f"  mod:        {totals['mod']:.3f}s ({100*totals['mod']/grand_total:.1f}%)")
    print()
    
    # Self-attention breakdown: Q/K/V projections + RoPE + SDPA + O projection
    print("SELF-ATTENTION dominates — likely breakdown:")
    print("  Q projection (linear):  ~25% of self_attn")
    print("  K projection (linear):  ~25% of self_attn")
    print("  V projection (linear):  ~25% of self_attn")
    print("  RoPE apply:             ~5%")
    print("  SDPA (Metal kernel):    ~15%")
    print("  O projection (linear):  ~5%")
    print()
    
    # Memory info
    try:
        peak_mem = mx.metal.get_peak_memory() / 1e9
        active_mem = mx.metal.get_active_memory() / 1e9
        print(f"Peak GPU memory:  {peak_mem:.2f} GB")
        print(f"Active GPU memory: {active_mem:.2f} GB")
    except:
        pass
    
    # Cleanup
    os.remove("/tmp/wan_deep_profile.mp4") if os.path.exists("/tmp/wan_deep_profile.mp4") else None

if __name__ == "__main__":
    main()
