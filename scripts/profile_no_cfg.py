#!/usr/bin/env python3
"""Test denoise with CFG disabled (B=1 forward pass)."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
from pathlib import Path

def main():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    model_path = Path.home() / ".cache/huggingface/hub/wan22-ti2v-5b-mlx"
    
    from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline
    
    # guidance=1.0 disables CFG → B=1 forward
    pipeline = Wan22MLXPipeline(
        model_path=model_path,
        width=832, height=480, frames=41, fps=24,
        steps=20, guidance=1.0, quant="none", cache="none", compile="off",
        seed=42,
    )
    
    print("Loading model...")
    pipeline.load_model()
    
    print("Encoding text (CFG disabled)...")
    prepared = pipeline.prepare_prompt(prompt="A golden retriever trots along a wet sandy beach.", negative_prompt="airbrushed,plastic,CGI")
    context = pipeline.encode_text(prepared)
    
    print("Initializing latents...")
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    mx = pipeline.mx
    model = pipeline.model
    scheduler = pipeline.scheduler
    
    # Compile
    model._compiled = mx.compile(model)
    _call = model._compiled
    context_cond = pipeline.context_cond
    timestep_list = scheduler.timesteps.tolist()
    seq_len = pipeline.seq_len
    
    # Warmup
    print("Warmup...")
    preds = _call(
        [latents], t=mx.array([timestep_list[0]]),
        context=context_cond, seq_len=seq_len,
        cross_kv_caches=pipeline.cross_kv, rope_cos_sin=pipeline.rope_cos_sin,
    )
    mx.eval(preds)
    
    # Re-init
    latents = pipeline.init_latents(seed=42, width=832, height=480, frames=41)
    
    total_start = time.perf_counter()
    
    for i in range(20):
        timestep_val = timestep_list[i]
        t0 = time.perf_counter()
        
        preds = _call(
            [latents], t=mx.array([timestep_val]),
            context=context_cond, seq_len=seq_len,
            cross_kv_caches=pipeline.cross_kv, rope_cos_sin=pipeline.rope_cos_sin,
        )
        mx.eval(preds)
        noise_pred = preds[0]
        
        latents = scheduler.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)
        
        t1 = time.perf_counter()
        print(f"Step {i:2d}: {t1-t0:.2f}s")
    
    total = time.perf_counter() - total_start
    print(f"\nTotal denoise (no CFG): {total:.1f}s")
    print(f"Avg per step: {total/20:.2f}s")
    print(f"\nCompare with CFG=3.5 baseline: ~56s (2.8s/step)")
    print(f"Speedup: {56.0/total:.1f}x")

if __name__ == "__main__":
    main()
