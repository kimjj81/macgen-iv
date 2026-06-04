#!/usr/bin/env python3
"""LTX-2.3 benchmark: profile denoise + VAE decode timing.

Uses DEV pipeline (single-stage, no spatial upscaler required).
Compiles with memory guard to prevent kernel panics.
"""
import os
os.environ["MACGEN_ALLOW_PARENT_MLX"] = "1"

import sys
import time
import json
from pathlib import Path

# Memory guard
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from fastgen_profiler.mlx_guard import check_memory_before_mlx
    check_memory_before_mlx()
except ImportError:
    print("Warning: mlx_guard not found, proceeding without memory check")

MODEL = Path(os.path.expanduser("~/.cache/huggingface/hub/ltx23-22b-distilled-mlx"))
TEXT_ENCODER = "google/gemma-3-12b-it"


def main():
    from mlx_video.models.ltx_2.generate import generate_video, PipelineType

    print("=== LTX-2.3 22B DEV Benchmark ===\n")
    print(f"Model: {MODEL}")
    print(f"Text encoder: {TEXT_ENCODER}")
    print(f"Pipeline: DEV (single-stage)\n")

    results = {}

    # ===== Test 1: Small resolution (512x512, 33 frames) =====
    if os.environ.get("SKIP_TEST1"):
        print("--- Test 1: SKIPPED (already run) ---")
        results["512x512_33f_20steps"] = 175.9  # from previous run
    else:
        print("--- Test 1: 512x512, 33 frames, 20 steps ---")
        t0 = time.time()
        generate_video(
            model_repo=str(MODEL),
            text_encoder_repo=TEXT_ENCODER,
            prompt="A golden retriever trots along a wet sandy beach, water splashing",
            negative_prompt="worst quality, low quality, static, blurry",
            height=512,
            width=512,
            num_frames=33,
            num_inference_steps=20,
            cfg_scale=4.0,
            seed=42,
            output_path="/tmp/ltx_bench_small.mp4",
            pipeline=PipelineType.DEV,
            tiling="auto",
        )
        total = time.time() - t0
        results["512x512_33f_20steps"] = total
        print(f"  Total: {total:.1f}s\n")

    # ===== Test 2: 1024x768 (divisible by 32), 33 frames =====
    if os.environ.get("SKIP_TEST2"):
        print("--- Test 2: SKIPPED (too heavy) ---\n")
    else:
        print("--- Test 2: 1024x768, 33 frames, 20 steps ---")
        t0 = time.time()
        generate_video(
            model_repo=str(MODEL),
            text_encoder_repo=TEXT_ENCODER,
            prompt="A golden retriever trots along a wet sandy beach, water splashing",
            negative_prompt="worst quality, low quality, static, blurry",
            height=768,
            width=1024,
            num_frames=33,
            num_inference_steps=20,
            cfg_scale=4.0,
            seed=42,
            output_path="/tmp/ltx_bench_1024x768.mp4",
            pipeline=PipelineType.DEV,
            tiling="auto",
        )
        total = time.time() - t0
        results["1024x768_33f_20steps"] = total
        print(f"  Total: {total:.1f}s\n")

    # ===== Test 3: Match Wan2.2 resolution (832x480, 33 frames) =====
    print("--- Test 3: 832x480, 33 frames, 20 steps ---")
    t0 = time.time()
    generate_video(
        model_repo=str(MODEL),
        text_encoder_repo=TEXT_ENCODER,
        prompt="A golden retriever trots along a wet sandy beach, water splashing",
        negative_prompt="worst quality, low quality, static, blurry",
        height=480,
        width=832,
        num_frames=33,
        num_inference_steps=20,
        cfg_scale=4.0,
        seed=42,
        output_path="/tmp/ltx_bench_832x480.mp4",
        pipeline=PipelineType.DEV,
        tiling="auto",
    )
    total = time.time() - t0
    results["832x480_33f_20steps"] = total
    print(f"  Total: {total:.1f}s\n")

    # ===== Summary =====
    print("=" * 60)
    print("LTX-2.3 BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:<35} {v:>7.1f}s")

    # Compare with Wan2.2 baseline
    print("\n  Reference: Wan2.2 832x480 41f 10steps = 68.5s")
    if "832x480_33f_20steps" in results:
        ltx = results["832x480_33f_20steps"]
        print(f"  LTX-2.3  832x480 33f 20steps   = {ltx:.1f}s")

    # Save
    out = "artifacts/bottleneck_bench/ltx23_benchmark.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for k, v in results.items():
            f.write(json.dumps({"config": k, "time_s": round(v, 1)}) + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
