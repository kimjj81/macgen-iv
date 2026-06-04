#!/usr/bin/env python3
"""LTX-2.3 22b adaptive benchmark — small first, then scale up."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json
import sys

MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/ltx23-22b-distilled-mlx")
TEXT_ENCODER_DIR = os.path.expanduser("~/.cache/huggingface/hub/LTX-2-text-local")

def run_bench(w, h, f, steps, label):
    from mlx_video.models.ltx_2.generate import generate_video, PipelineType
    
    print(f"\n{'='*60}")
    print(f"  {label}: {w}x{h} f{f} s{steps}")
    print(f"{'='*60}")
    
    t0 = time.time()
    try:
        output = generate_video(
            model_repo=MODEL_DIR,
            text_encoder_repo=TEXT_ENCODER_DIR,
            prompt="A golden retriever trots along a wet sandy beach, ocean waves in the background",
            negative_prompt="",
            height=h, width=w,
            num_frames=f,
            num_inference_steps=steps,
            cfg_scale=3.5,
            seed=42,
            fps=24,
            output_path=f"/tmp/ltx_{label}.mp4",
            verbose=True,
            pipeline=PipelineType.DEV,
        )
        elapsed = time.time() - t0
        print(f"  TOTAL: {elapsed:.1f}s ({elapsed/f:.2f}s/frame)")
        return {"label": label, "w": w, "h": h, "frames": f, "steps": steps,
                "seconds": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR ({elapsed:.1f}s): {e}")
        return {"label": label, "w": w, "h": h, "frames": f, "steps": steps,
                "seconds": elapsed, "error": str(e)[:200]}

def main():
    results = []
    
    # Phase 1: Small probe (256x160, 9 frames, 10 steps)
    r = run_bench(256, 160, 9, 10, "probe")
    results.append(r)
    if r["error"]:
        print(f"Probe failed: {r['error']}")
        sys.exit(1)
    
    # Phase 2: Standard benchmark (512x320, 41 frames)
    for steps in [20]:
        r = run_bench(512, 320, 41, steps, f"s{steps}")
        results.append(r)
    
    # Phase 3: More steps if memory allows
    if results[-1]["error"] is None:
        r = run_bench(512, 320, 41, 40, "s40")
        results.append(r)
    
    # Write results
    out_path = "artifacts/bottleneck_bench_ltx/adaptive_bench.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {out_path}")
    
    # Cleanup
    import glob
    for p in glob.glob("/tmp/ltx_*.mp4"):
        os.remove(p)

if __name__ == "__main__":
    main()
