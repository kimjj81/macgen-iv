#!/bin/bash
# Wan2.2-TI2V-5B 병목 타겟 벤치마크
# 832x480, 41 frames, g=3.5, 24fps, no video save (빠른 반복)

set -e
cd "$(dirname "$0")/.."

MODEL_PATH="$HOME/.cache/huggingface/hub/wan22-ti2v-5b-mlx"
OUTDIR="artifacts/bottleneck_bench"
JSONL_DIR="artifacts/bottleneck_bench"
mkdir -p "$OUTDIR"

PROMPT="A golden retriever trots along a wet sandy beach, shallow waves washing over its paws. The camera tracks alongside in a smooth lateral dolly shot. Warm sunset backlight creates golden rim light on the dog's fur. Sea spray catches the light. Natural fur texture, documentary style, handheld camera feel, film grain."

NEGATIVE="airbrushed, smooth, plastic, CGI, oversaturated, HDR, overprocessed, beauty filter, soft focus, vaseline lens, cartoon, anime, painting"

STEPS_LIST="20 28 32 40 48"

echo "========================================"
echo "Wan2.2 Bottleneck Benchmark"
echo "832x480, 41f, g=3.5, 24fps"
echo "Steps: $STEPS_LIST"
echo "========================================"

for STEPS in $STEPS_LIST; do
    echo ""
    echo ">>> steps=$STEPS ($(date +%H:%M:%S))"
    JSONL="$JSONL_DIR/steps${STEPS}.jsonl"
    rm -f "$JSONL"
    
    MACGEN_ALLOW_PARENT_MLX=1 uv run fastgen-profile run \
      --model wan2.2 --backend mlx \
      --model-path "$MODEL_PATH" \
      --prompt "$PROMPT" \
      --negative-prompt "$NEGATIVE" \
      --seed 42 \
      --width 832 --height 480 --frames 41 --fps 24 \
      --steps "$STEPS" --guidance 3.5 \
      --quant none --cache none --compile off \
      --output-dir "$OUTDIR/out_steps${STEPS}" \
      --result-jsonl "$JSONL" \
      --no-save-video 2>&1 | grep -E "guard|EXIT" || true
    
    echo "<<< steps=$STEPS done"
done

echo ""
echo "========================================"
echo "Analysis"
echo "========================================"

python3 << 'PYEOF'
import json, os, glob

steps_list = [20, 28, 32, 40, 48]
base = os.path.join("artifacts", "bottleneck_bench")

all_results = []

for steps in steps_list:
    path = os.path.join(base, f"steps{steps}.jsonl")
    phases = {}
    denoise_steps = []
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("error"):
                    continue
                if r["phase"] == "denoise_step":
                    denoise_steps.append({
                        "index": len(denoise_steps),
                        "seconds": r["seconds"],
                        "active_memory": r.get("active_memory"),
                        "cache_memory": r.get("cache_memory"),
                    })
                else:
                    phases[r["phase"]] = {
                        "seconds": r["seconds"],
                        "active_memory": r.get("active_memory"),
                        "cache_memory": r.get("cache_memory"),
                    }
    except FileNotFoundError:
        print(f"  steps={steps}: FILE NOT FOUND")
        continue

    dt = sum(s["seconds"] for s in denoise_steps)
    avg = dt / len(denoise_steps) if denoise_steps else 0

    # 초반(0-25%), 중반(25-75%), 후반(75-100%) 분석
    n = len(denoise_steps)
    early = denoise_steps[:max(1, n//4)]
    mid = denoise_steps[n//4:3*n//4] if n > 4 else denoise_steps[n//4:]
    late = denoise_steps[max(0, 3*n//4):]

    e_avg = sum(s["seconds"] for s in early) / len(early)
    m_avg = sum(s["seconds"] for s in mid) / len(mid) if mid else 0
    l_avg = sum(s["seconds"] for s in late) / len(late)

    all_results.append({
        "steps": steps,
        "total": phases.get("total", {}).get("seconds", 0),
        "model_load": phases.get("model_load", {}).get("seconds", 0),
        "text_encoder": phases.get("text_encoder", {}).get("seconds", 0),
        "denoise_total": dt,
        "denoise_avg": avg,
        "denoise_early_avg": e_avg,
        "denoise_mid_avg": m_avg,
        "denoise_late_avg": l_avg,
        "vae_decode": phases.get("vae_decode", {}).get("seconds", 0),
        "denoise_steps": denoise_steps,
    })

# 헤더
print(f"\n{'Steps':>5} | {'Total':>7} | {'Load':>5} | {'T5':>5} | {'Denoise':>7} | {'Early':>6} | {'Mid':>6} | {'Late':>6} | {'VAE':>6}")
print("-" * 78)

for r in all_results:
    print(f"{r['steps']:>5} | {r['total']:>7.1f} | {r['model_load']:>5.2f} | {r['text_encoder']:>5.3f} | {r['denoise_total']:>7.1f} | {r['denoise_early_avg']*1000:>5.0f}ms | {r['denoise_mid_avg']*1000:>5.0f}ms | {r['denoise_late_avg']*1000:>5.0f}ms | {r['vae_decode']:>6.1f}")

# 페이즈 비율 (steps=40 기준)
r40 = next((r for r in all_results if r["steps"] == 40), None)
if r40:
    t = r40["total"]
    print(f"\nPhase breakdown (steps=40, total={t:.1f}s):")
    for name, val in [("model_load", r40["model_load"]), ("text_encoder", r40["text_encoder"]), ("denoise_total", r40["denoise_total"]), ("vae_decode", r40["vae_decode"])]:
        pct = val / t * 100 if t else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:20s} {val:>8.2f}s ({pct:>5.1f}%) {bar}")
    other = t - r40["model_load"] - r40["text_encoder"] - r40["denoise_total"] - r40["vae_decode"]
    print(f"  {'other':20s} {other:>8.2f}s ({other/t*100:>5.1f}%)")

# Step별 타이밍 그래프 (steps=40)
if r40:
    steps_data = r40["denoise_steps"]
    if steps_data:
        max_s = max(s["seconds"] for s in steps_data)
        print(f"\nPer-step timing (steps=40, max={max_s*1000:.0f}ms):")
        for s in steps_data:
            bar_len = int(s["seconds"] / max_s * 40)
            print(f"  {s['index']:>3d} |{'█' * bar_len} {s['seconds']*1000:.0f}ms")

# 병목 분석
print("\n--- 병목 분석 ---")
if r40:
    t = r40["total"]
    d_pct = r40["denoise_total"] / t * 100
    v_pct = r40["vae_decode"] / t * 100
    print(f"  1차 병목: denoise ({d_pct:.1f}%) — transformer forward pass")
    print(f"  2차 병목: vae_decode ({v_pct:.1f}%) — 3D 디컨볼루션")
    print(f"  fixed cost: model_load+text_encoder ({(r40['model_load']+r40['text_encoder'])/t*100:.1f}%)")

    # 초반 vs 후반 step 속도 차이
    early_avg = r40["denoise_early_avg"]
    late_avg = r40["denoise_late_avg"]
    if early_avg > 0:
        ratio = late_avg / early_avg
        if ratio > 1.1:
            print(f"  후반 step이 초반보다 {ratio:.2f}x 느림 — noise schedule 영향")
        elif ratio < 0.9:
            print(f"  후반 step이 초반보다 {1/ratio:.2f}x 빠름 — 캐시 효과 추정")
        else:
            print(f"  step별 속도 균일 (early/late ratio: {ratio:.2f})")

# 최종 결과 JSONL로 저장
summary_path = os.path.join(base, "summary.jsonl")
with open(summary_path, "w") as f:
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "denoise_steps"}
        f.write(json.dumps(sr) + "\n")
print(f"\n요약: {summary_path}")
PYEOF
