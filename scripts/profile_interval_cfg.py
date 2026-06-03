#!/usr/bin/env python3
"""Test interval CFG: apply CFG only on first N steps, rest B=1."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
from pathlib import Path

def run_with_interval_cfg(cfg_steps: int, total_steps: int = 20, guidance: float = 3.5):
    """Run denoise with CFG only on first cfg_steps steps."""
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    model_path = Path.home() / ".cache/huggingface/hub/wan22-ti2v-5b-mlx"
    
    from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline
    
    # Use CFG mode for encoding (need both cond and uncond)
    pipeline = Wan22MLXPipeline(
        model_path=model_path,
        width=832, height=480, frames=41, fps=24,
        steps=total_steps, guidance=guidance, quant="none", cache="none", compile="off",
        seed=42,
    )
    
    pipeline.load_model()
    prepared = pipeline.prepare_prompt(prompt="A golden retriever trots along a wet sandy beach.", negative_prompt="airbrushed,plastic,CGI")
    context = pipeline.encode_text(prepared)
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    mx = pipeline.mx
    model = pipeline.model
    scheduler = pipeline.scheduler
    model._compiled = mx.compile(model)
    _call = model._compiled
    context_cfg = pipeline.context_cfg
    # Extract conditional-only context from the B=2 CFG context
    context_cond = context_cfg[0:1]  # First half is conditional
    
    # Prepare B=1 cross_kv for the non-CFG steps
    cross_kv_b1 = pipeline.model.prepare_cross_kv(context_cond)
    mx.eval(cross_kv_b1)
    
    cross_kv_b2 = pipeline.cross_kv
    timestep_list = scheduler.timesteps.tolist()
    seq_len = pipeline.seq_len
    
    # Warmup (1 step B=2)
    preds = _call(
        [latents, latents], t=mx.array([timestep_list[0], timestep_list[0]]),
        context=context_cfg, seq_len=seq_len,
        cross_kv_caches=cross_kv_b2, rope_cos_sin=pipeline.rope_cos_sin,
    )
    mx.eval(preds)
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    total_start = time.perf_counter()
    
    for i in range(total_steps):
        timestep_val = timestep_list[i]
        t0 = time.perf_counter()
        
        if i < cfg_steps:
            # B=2 CFG pass
            t_batch = mx.array([timestep_val, timestep_val])
            preds = _call(
                [latents, latents], t=t_batch,
                context=context_cfg, seq_len=seq_len,
                cross_kv_caches=cross_kv_b2, rope_cos_sin=pipeline.rope_cos_sin,
            )
            mx.eval(preds)
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
        else:
            # B=1 pass (conditional only)
            preds = _call(
                [latents], t=mx.array([timestep_val]),
                context=context_cond, seq_len=seq_len,
                cross_kv_caches=cross_kv_b1, rope_cos_sin=pipeline.rope_cos_sin,
            )
            mx.eval(preds)
            noise_pred = preds[0]
        
        latents = scheduler.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)
        
        t1 = time.perf_counter()
    
    total = time.perf_counter() - total_start
    return total

def main():
    configs = [
        (20, "All steps CFG (B=2)"),
        (10, "First 10 CFG, rest B=1"),
        (5,  "First 5 CFG, rest B=1"),
        (3,  "First 3 CFG, rest B=1"),
        (1,  "First 1 CFG (warmup), rest B=1"),
        (0,  "No CFG (B=1)"),
    ]
    
    print(f"{'Config':30s} | {'Time':>7s} | {'vs Full':>7s}")
    print("-" * 55)
    
    results = {}
    for cfg_steps, label in configs:
        t = run_with_interval_cfg(cfg_steps)
        results[cfg_steps] = t
        baseline = results.get(20, t)
        ratio = t / baseline if baseline != t else 1.0
        print(f"{label:30s} | {t:5.1f}s | {ratio:5.2f}x")

if __name__ == "__main__":
    main()
