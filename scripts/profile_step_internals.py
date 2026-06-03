#!/usr/bin/env python3
"""Profile individual operations inside a Wan2.2 denoise step."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import sys
import time
import numpy as np

def main():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    # Load model via mlx_video
    from mlx_video.models.wan_2 import generate as wan_gen
    from pathlib import Path
    import json
    
    model_path = Path.home() / ".cache/huggingface/hub/wan22-ti2v-5b-mlx"
    
    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)
    
    print(f"Loading model from {model_path}...")
    t0 = time.perf_counter()
    
    # Use wan22 generate_video internals
    # We need to do this manually to profile each step
    
    from mlx_video.models.wan_2.wan_model import WanModel
    from mlx_video.models.wan_2.pipeline import VideoPipeline
    
    # Load pipeline
    pipeline = wan_gen.load_pipeline(str(model_path))
    t1 = time.perf_counter()
    print(f"Pipeline loaded in {t1-t0:.1f}s")
    
    # Prepare inputs
    prompt = "A golden retriever trots along a wet sandy beach."
    neg_prompt = "airbrushed,plastic,CGI"
    
    # Encode text
    t2 = time.perf_counter()
    context = pipeline.encode_text(prompt, neg_prompt)
    mx.eval(context)
    t3 = time.perf_counter()
    print(f"Text encoded in {t3-t2:.1f}s")
    
    # Init latents
    H, W, F = 480, 832, 41
    latents = pipeline.init_latents(F, H, W, seed=42)
    
    # Profile a few steps
    from mlx_video.models.wan_2.scheduler import FlowMatchScheduler
    scheduler = FlowMatchScheduler(
        shift=5.0,
        num_train_timesteps=1000,
        num_inference_steps=20,
    )
    scheduler.set_timesteps(20)
    
    for step_idx in range(5):
        timestep = scheduler.timesteps[step_idx]
        t = mx.array([timestep] * 2)
        
        # Profile model forward
        step_start = time.perf_counter()
        
        # Split the forward into parts if possible
        noise_pred = pipeline.model(
            [latents] * 2,
            t=t,
            context=context,
        )
        mx.eval(noise_pred)
        
        step_end = time.perf_counter()
        
        # Scheduler step
        latents = scheduler.step(noise_pred, timestep, latents)
        mx.eval(latents)
        
        sched_end = time.perf_counter()
        
        print(f"Step {step_idx}: model={step_end-step_start:.2f}s  scheduler={sched_end-step_end:.2f}s  total={sched_end-step_start:.2f}s")
    
    print("Done.")

if __name__ == "__main__":
    main()
