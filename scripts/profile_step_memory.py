#!/usr/bin/env python3
"""Profile step-by-step timing with memory tracking."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import subprocess

def get_memory_info():
    """Get system memory info."""
    result = subprocess.run(["sysctl", "vm.page_pageable_internal_count", "vm.page_free_count", 
                            "vm.page_external_count", "vm.page_size"], 
                           capture_output=True, text=True)
    return result.stdout.strip()

def get_gpu_memory():
    """Get GPU memory via system_profiler or ioreg."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        free = 0
        active = 0
        for line in lines:
            if "Pages free" in line:
                free = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages active" in line:
                active = int(line.split(":")[1].strip().rstrip("."))
        return free * 4096 / 1e9, active * 4096 / 1e9  # GB
    except:
        return 0, 0

def main():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    
    from fastgen_profiler.backends.wan22_mlx_adapter import Wan22MLXPipeline
    from pathlib import Path
    
    model_path = Path.home() / ".cache/huggingface/hub/wan22-ti2v-5b-mlx"
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
    latents = pipeline.init_latents()
    
    print(f"\n{'Step':>4s} | {'Time':>7s} | {'FreeGB':>7s} | {'ActiveGB':>8s} | {'GPUAlloc':>8s}")
    print("-" * 50)
    
    for step in range(20):
        free, active = get_gpu_memory()
        gpu_alloc = mx.get_active_memory() / 1e9
        
        t0 = time.perf_counter()
        latents = pipeline.denoise_step(latents, step_index=step, steps=20, guidance=3.5, cache="none")
        t1 = time.perf_counter()
        
        print(f"{step:4d} | {t1-t0:6.2f}s | {free:6.1f}GB | {active:7.1f}GB | {gpu_alloc:7.1f}GB")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
