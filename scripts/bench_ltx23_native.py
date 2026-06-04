#!/usr/bin/env python3
"""LTX-2.3 22b benchmark using mlx_video generate_video directly."""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import time
import json

MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub/ltx23-22b-distilled-mlx")
TEXT_ENCODER_DIR = os.path.expanduser("~/.cache/huggingface/hub/LTX-2-text-local")

def main():
    from mlx_video.models.ltx_2.generate import generate_video
    
    print("=== LTX-2.3 22b Distilled MLX Benchmark ===")
    print(f"Model:  {MODEL_DIR}")
    print(f"Text:   {TEXT_ENCODER_DIR}")
    print()

    # Small resolution first to avoid kernel panic
    # LTX-2.3 distilled: fewer steps needed
    results = []
    
    for steps in [20, 40]:
        for w, h, f in [(512, 320, 41)]:
            label = f"{w}x{h} f{f} s{steps}"
            print(f"--- {label} ---")
            t0 = time.time()
            try:
                from mlx_video.models.ltx_2.generate import PipelineType
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
                    output_path=f"/tmp/ltx_bench_{label.replace(' ','_')}.mp4",
                    verbose=True,
                    pipeline=PipelineType.DEV,
                )
                elapsed = time.time() - t0
                print(f"  TOTAL: {elapsed:.1f}s")
                results.append({"label": label, "seconds": elapsed, "error": None})
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR ({elapsed:.1f}s): {e}")
                results.append({"label": label, "seconds": elapsed, "error": str(e)[:200]})
            print()

    # Write results
    out_path = "artifacts/bottleneck_bench_ltx/native_bench.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()
