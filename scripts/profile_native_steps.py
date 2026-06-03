#!/usr/bin/env python3
"""Profile step-by-step timing using mlx_video generate internals directly."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
from pathlib import Path

def main():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    model_path = Path.home() / ".cache/huggingface/hub/wan22-ti2v-5b-mlx"
    
    # Use the adapter to load, then profile the native loop
    from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline
    
    pipeline = Wan22MLXPipeline(
        model_path=model_path,
        width=832, height=480, frames=41, fps=24,
        steps=20, guidance=3.5, quant="none", cache="none", compile="off",
        seed=42,
    )
    
    print("Loading model...")
    pipeline.load_model()
    
    print("Encoding text...")
    prepared = pipeline.prepare_prompt(prompt="A golden retriever trots along a wet sandy beach.", negative_prompt="airbrushed,plastic,CGI")
    context = pipeline.encode_text(prepared)
    
    print("Initializing latents...")
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    mx = pipeline.mx
    model = pipeline.model
    scheduler = pipeline.scheduler
    
    # Compile model forward
    model._compiled = mx.compile(model)
    
    _call = model._compiled
    context_batch = pipeline.context_cfg
    timestep_list = scheduler.timesteps.tolist()
    seq_len = pipeline.seq_len
    
    # Warmup compile
    print("Warmup compile (1 step)...")
    t0 = time.perf_counter()
    preds = _call(
        [latents, latents], t=mx.array([timestep_list[0], timestep_list[0]]),
        context=context_batch, seq_len=seq_len,
        cross_kv_caches=pipeline.cross_kv, rope_cos_sin=pipeline.rope_cos_sin,
    )
    mx.eval(preds)
    t1 = time.perf_counter()
    print(f"Warmup: {t1-t0:.1f}s (includes compile trace)")
    
    # Re-init latents for clean test
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    print(f"\n{'Step':>4s} | {'Model':>7s} | {'Sched':>7s} | {'ActiveGB':>8s}")
    print("-" * 40)
    
    for i in range(20):
        timestep_val = timestep_list[i]
        gs = 3.5
        t_batch = mx.array([timestep_val, timestep_val])
        
        t0 = time.perf_counter()
        preds = _call(
            [latents, latents], t=t_batch,
            context=context_batch, seq_len=seq_len,
            cross_kv_caches=pipeline.cross_kv, rope_cos_sin=pipeline.rope_cos_sin,
        )
        mx.eval(preds)
        t1 = time.perf_counter()
        
        noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
        noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
        
        t2 = time.perf_counter()
        latents = scheduler.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)
        t3 = time.perf_counter()
        
        active = mx.get_active_memory() / 1e9
        print(f"{i:4d} | {t1-t0:6.2f}s | {t3-t2:6.3f}s | {active:7.1f}GB")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
