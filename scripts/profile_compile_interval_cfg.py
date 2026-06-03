#!/usr/bin/env python3
"""Compare compile + interval CFG combos."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
from pathlib import Path

def run_benchmark(steps=20, cfg_steps=5, guidance=3.5, use_compile=True):
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    model_path = Path.home() / ".cache/huggingface/hub/wan22-ti2v-5b-mlx"
    from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline
    
    pipeline = Wan22MLXPipeline(
        model_path=model_path,
        width=832, height=480, frames=41, fps=24,
        steps=steps, guidance=guidance, quant="none", cache="none", compile="off",
        seed=42,
    )
    pipeline.load_model()
    prepared = pipeline.prepare_prompt(prompt="A golden retriever trots along a wet sandy beach.", negative_prompt="airbrushed,plastic,CGI")
    context = pipeline.encode_text(prepared)
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    mx = pipeline.mx
    model = pipeline.model
    scheduler = pipeline.scheduler
    
    if use_compile:
        model._compiled = mx.compile(model)
    _call = getattr(model, "_compiled", model)
    
    context_cfg = pipeline.context_cfg
    context_cond = context_cfg[0:1]
    cross_kv_b2 = pipeline.cross_kv
    cross_kv_b1 = model.prepare_cross_kv(context_cond)
    mx.eval(cross_kv_b1)
    timestep_list = scheduler.timesteps.tolist()
    seq_len = pipeline.seq_len
    
    # Warmup (B=2)
    preds = _call([latents, latents], t=mx.array([timestep_list[0], timestep_list[0]]),
        context=context_cfg, seq_len=seq_len, cross_kv_caches=cross_kv_b2, rope_cos_sin=pipeline.rope_cos_sin)
    mx.eval(preds)
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    # Also warmup B=1
    preds = _call([latents], t=mx.array([timestep_list[0]]),
        context=context_cond, seq_len=seq_len, cross_kv_caches=cross_kv_b1, rope_cos_sin=pipeline.rope_cos_sin)
    mx.eval(preds)
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    total_start = time.perf_counter()
    for i in range(steps):
        timestep_val = timestep_list[i]
        if i < cfg_steps:
            preds = _call([latents, latents], t=mx.array([timestep_val, timestep_val]),
                context=context_cfg, seq_len=seq_len, cross_kv_caches=cross_kv_b2, rope_cos_sin=pipeline.rope_cos_sin)
            mx.eval(preds)
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
        else:
            preds = _call([latents], t=mx.array([timestep_val]),
                context=context_cond, seq_len=seq_len, cross_kv_caches=cross_kv_b1, rope_cos_sin=pipeline.rope_cos_sin)
            mx.eval(preds)
            noise_pred = preds[0]
        latents = scheduler.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)
    
    return time.perf_counter() - total_start

def main():
    configs = [
        (True,  20, "compile + full CFG"),
        (True,   5, "compile + 5-step CFG"),
        (True,   3, "compile + 3-step CFG"),
        (False, 20, "no compile + full CFG"),
        (False,  5, "no compile + 5-step CFG"),
    ]
    
    print(f"{'Config':30s} | {'Time':>7s} | {'vs Baseline':>10s}")
    print("-" * 55)
    
    baseline = None
    for use_compile, cfg_steps, label in configs:
        t = run_benchmark(cfg_steps=cfg_steps, use_compile=use_compile)
        if baseline is None:
            baseline = t
        print(f"{label:30s} | {t:5.1f}s | {t/baseline:8.2f}x")

if __name__ == "__main__":
    main()
