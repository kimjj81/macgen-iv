#!/usr/bin/env python3
"""Wan2.2 sub-component profile: break down self_attn and ffn to individual ops.

Measures each linear projection and attention kernel separately.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import mlx.core as mx

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/wan22-ti2v-5b-mlx")

def main():
    from mlx_video.models.wan_2 import transformer, wan_2, attention, rope as rope_mod
    
    # ---- Profile self_attn internals ----
    original_self_attn = attention.WanSelfAttention.__call__
    self_attn_detail = {}
    
    def timed_self_attn(self, x, seq_lens, grid_sizes, freqs, rope_cos_sin=None, attn_mask=None):
        idx = id(self)
        if idx not in self_attn_detail:
            self_attn_detail[idx] = {
                "q_proj": [], "k_proj": [], "v_proj": [],
                "q_norm": [], "k_norm": [],
                "rope": [], "sdpa": [], "o_proj": [],
                "reshape": [],
            }
        d = self_attn_detail[idx]
        b, s, _ = x.shape
        n, hd = self.num_heads, self.head_dim
        w_dtype = attention._linear_dtype(self.q)
        x_w = x.astype(w_dtype)
        
        # Q projection
        t0 = time.perf_counter()
        q = self.q(x_w)
        mx.eval(q)
        d["q_proj"].append(time.perf_counter() - t0)
        
        # K projection
        t0 = time.perf_counter()
        k = self.k(x_w)
        mx.eval(k)
        d["k_proj"].append(time.perf_counter() - t0)
        
        # Q norm
        t0 = time.perf_counter()
        if self.norm_q: q = self.norm_q(q)
        if self.norm_k: k = self.norm_k(k)
        mx.eval(q, k)
        d["q_norm"].append(time.perf_counter() - t0)
        
        # Reshape
        t0 = time.perf_counter()
        q = q.reshape(b, s, n, hd)
        k = k.reshape(b, s, n, hd)
        v = self.v(x_w).reshape(b, s, n, hd)
        mx.eval(q, k, v)
        d["reshape"].append(time.perf_counter() - t0)
        
        # RoPE
        t0 = time.perf_counter()
        q = rope_mod.rope_apply(q.astype(mx.float32), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin)
        k = rope_mod.rope_apply(k.astype(mx.float32), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin)
        q = q.astype(w_dtype).transpose(0, 2, 1, 3)
        k = k.astype(w_dtype).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        mx.eval(q, k, v)
        d["rope"].append(time.perf_counter() - t0)
        
        # Mask
        mask = attn_mask
        if mask is None and any(sl < s for sl in seq_lens):
            mask = mx.zeros((b, 1, 1, s), dtype=q.dtype)
            for i, sl in enumerate(seq_lens):
                mask[i, :, :, sl:] = -1e9
        
        # SDPA
        t0 = time.perf_counter()
        if mask is not None:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        mx.eval(out)
        d["sdpa"].append(time.perf_counter() - t0)
        
        # O projection
        t0 = time.perf_counter()
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        out = self.o(out)
        mx.eval(out)
        d["o_proj"].append(time.perf_counter() - t0)
        
        return out
    
    # ---- Profile FFN internals ----
    original_ffn = transformer.WanFFN.__call__
    ffn_detail = {}
    
    def timed_ffn(self, x):
        idx = id(self)
        if idx not in ffn_detail:
            ffn_detail[idx] = {"fc1": [], "act": [], "fc2": []}
        d = ffn_detail[idx]
        
        x_w = x.astype(attention._linear_dtype(self.fc1))
        
        t0 = time.perf_counter()
        h = self.fc1(x_w)
        mx.eval(h)
        d["fc1"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        h = self.act(h)
        mx.eval(h)
        d["act"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        out = self.fc2(h)
        mx.eval(out)
        d["fc2"].append(time.perf_counter() - t0)
        
        return out
    
    # ---- Profile cross_attn internals ----
    original_cross = attention.WanCrossAttention.__call__
    cross_detail = {}
    
    def timed_cross(self, x, context, context_lens=None, kv_cache=None):
        idx = id(self)
        if idx not in cross_detail:
            cross_detail[idx] = {"q_proj": [], "sdpa": [], "o_proj": []}
        d = cross_detail[idx]
        
        b = x.shape[0]
        n, hd = self.num_heads, self.head_dim
        w_dtype = attention._linear_dtype(self.q)
        
        t0 = time.perf_counter()
        q = self.q(x.astype(w_dtype))
        if self.norm_q: q = self.norm_q(q)
        q = q.reshape(b, -1, n, hd).transpose(0, 2, 1, 3)
        mx.eval(q)
        d["q_proj"].append(time.perf_counter() - t0)
        
        k, v = kv_cache if kv_cache else (None, None)
        
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        mx.eval(out)
        d["sdpa"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        out = out.transpose(0, 2, 1, 3).reshape(b, -1, n * hd)
        out = self.o(out)
        mx.eval(out)
        d["o_proj"].append(time.perf_counter() - t0)
        
        return out
    
    # Apply patches
    attention.WanSelfAttention.__call__ = timed_self_attn
    transformer.WanFFN.__call__ = timed_ffn
    attention.WanCrossAttention.__call__ = timed_cross
    
    print("=== Wan2.2 Sub-Component Profile ===")
    print()
    
    from mlx_video.models.wan_2.generate import generate_video
    output = generate_video(
        model_dir=MODEL_PATH,
        prompt="A golden retriever trots along a wet sandy beach, ocean waves in the background",
        height=480, width=832,
        num_frames=41,
        steps=3,
        guide_scale=3.5,
        seed=42,
        output_path="/tmp/wan_sub_profile.mp4",
        no_compile=True,
    )
    
    # Restore
    attention.WanSelfAttention.__call__ = original_self_attn
    transformer.WanFFN.__call__ = original_ffn
    attention.WanCrossAttention.__call__ = original_cross
    
    # Analyze self_attn breakdown (average across all blocks and steps)
    print("\n" + "="*80)
    print("SELF-ATTENTION BREAKDOWN (avg per block, 3 steps)")
    print("="*80)
    
    all_sa = {}
    for d in self_attn_detail.values():
        for k, vals in d.items():
            if k not in all_sa: all_sa[k] = []
            all_sa[k].extend(vals)
    
    sa_total = sum(sum(v) for v in all_sa.values())
    print(f"{'Component':>15} | {'Total':>10} | {'Avg/block':>10} | {'% of self_attn':>15}")
    print("-" * 60)
    for k in ["q_proj", "k_proj", "reshape", "v_proj_in_reshape", "q_norm", "rope", "sdpa", "o_proj"]:
        if k == "v_proj_in_reshape":
            continue
        if k in all_sa and all_sa[k]:
            total = sum(all_sa[k])
            avg = total / len(all_sa[k])
            pct = 100 * total / sa_total
            print(f"{k:>15} | {total:>9.4f}s | {avg:>9.5f}s | {pct:>14.1f}%")
    print(f"{'TOTAL':>15} | {sa_total:>9.4f}s |")
    
    # FFN breakdown
    print("\n" + "="*80)
    print("FFN BREAKDOWN (avg per block, 3 steps)")
    print("="*80)
    
    all_ffn = {}
    for d in ffn_detail.values():
        for k, vals in d.items():
            if k not in all_ffn: all_ffn[k] = []
            all_ffn[k].extend(vals)
    
    ffn_total = sum(sum(v) for v in all_ffn.values())
    print(f"{'Component':>15} | {'Total':>10} | {'Avg/block':>10} | {'% of FFN':>10}")
    print("-" * 55)
    for k in ["fc1", "act", "fc2"]:
        if k in all_ffn and all_ffn[k]:
            total = sum(all_ffn[k])
            avg = total / len(all_ffn[k])
            pct = 100 * total / ffn_total
            print(f"{k:>15} | {total:>9.4f}s | {avg:>9.5f}s | {pct:>9.1f}%")
    
    # Cross attention breakdown
    print("\n" + "="*80)
    print("CROSS-ATTENTION BREAKDOWN (avg per block, 3 steps)")
    print("="*80)
    
    all_ca = {}
    for d in cross_detail.values():
        for k, vals in d.items():
            if k not in all_ca: all_ca[k] = []
            all_ca[k].extend(vals)
    
    ca_total = sum(sum(v) for v in all_ca.values())
    print(f"{'Component':>15} | {'Total':>10} | {'Avg/block':>10} | {'% of cross':>10}")
    print("-" * 55)
    for k in ["q_proj", "sdpa", "o_proj"]:
        if k in all_ca and all_ca[k]:
            total = sum(all_ca[k])
            avg = total / len(all_ca[k])
            pct = 100 * total / ca_total
            print(f"{k:>15} | {total:>9.4f}s | {avg:>9.5f}s | {pct:>9.1f}%")
    
    # Grand summary
    print("\n" + "="*80)
    print("GRAND SUMMARY — where does the time go?")
    print("="*80)
    grand = sa_total + ffn_total + ca_total
    print(f"self_attn total:   {sa_total:.3f}s ({100*sa_total/grand:.1f}%)")
    print(f"  linear (Q+K+V+O): {sum(sum(all_sa.get(k,[])) for k in ['q_proj','k_proj','o_proj']):.3f}s")
    print(f"  rope+reshape:     {sum(sum(all_sa.get(k,[])) for k in ['rope','reshape']):.3f}s")
    print(f"  sdpa:             {sum(all_sa.get('sdpa',[])):.3f}s")
    print(f"ffn total:         {ffn_total:.3f}s ({100*ffn_total/grand:.1f}%)")
    print(f"  fc1 (expand):     {sum(all_ffn.get('fc1',[])):.3f}s")
    print(f"  fc2 (contract):   {sum(all_ffn.get('fc2',[])):.3f}s")
    print(f"cross_attn total:  {ca_total:.3f}s ({100*ca_total/grand:.1f}%)")
    print(f"ALL OPERATIONS:    {grand:.3f}s")
    
    # Cleanup
    import os
    if os.path.exists("/tmp/wan_sub_profile.mp4"):
        os.remove("/tmp/wan_sub_profile.mp4")

if __name__ == "__main__":
    main()
