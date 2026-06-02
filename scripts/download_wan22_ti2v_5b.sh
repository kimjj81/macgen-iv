#!/usr/bin/env bash
# Download Wan2.2-TI2V-5B from HuggingFace with 80Mbps bandwidth limit.
# Total: ~32GB. Output: ~/.cache/huggingface/hub/wan22-ti2v-5b-source/
#
# Usage:
#   bash scripts/download_wan22_ti2v_5b.sh              # default 10M (80Mbps)
#   LIMIT_RATE=5M bash scripts/download_wan22_ti2v_5b.sh # slower
#   LIMIT_RATE=0  bash scripts/download_wan22_ti2v_5b.sh # no limit

set -euo pipefail

REPO="Wan-AI/Wan2.2-TI2V-5B"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
OUTDIR="${1:-$HOME/.cache/huggingface/hub/wan22-ti2v-5b-source}"
LIMIT_RATE="${LIMIT_RATE:-10M}"  # 10M = ~80Mbps

mkdir -p "${OUTDIR}/google/umt5-xxl"

echo "=== Wan2.2-TI2V-5B Download ==="
echo "Output: ${OUTDIR}"
echo "Rate limit: ${LIMIT_RATE}/s"
echo "Estimated total: ~32 GB"
echo ""

download_file() {
  local relpath="$1"
  local dest="${OUTDIR}/${relpath}"
  local url="${BASE_URL}/${relpath}"

  if [ -f "${dest}" ]; then
    echo "  [SKIP] ${relpath} (exists)"
    return 0
  fi

  echo -n "  [DL]   ${relpath} ... "
  
  local rate_arg=""
  if [ "${LIMIT_RATE}" != "0" ]; then
    rate_arg="--limit-rate ${LIMIT_RATE}"
  fi

  # shellcheck disable=SC2086
  if curl -L --progress-bar ${rate_arg} -o "${dest}.part" "${url}" 2>&1; then
    mv "${dest}.part" "${dest}"
    echo "done"
  else
    echo "FAILED"
    rm -f "${dest}.part"
    return 1
  fi
}

echo "--- Transformer + Config ---"
for f in \
  "config.json" \
  "configuration.json" \
  "diffusion_pytorch_model.safetensors.index.json" \
  "diffusion_pytorch_model-00001-of-00003.safetensors" \
  "diffusion_pytorch_model-00002-of-00003.safetensors" \
  "diffusion_pytorch_model-00003-of-00003.safetensors"; do
  download_file "${f}"
done

echo ""
echo "--- VAE ---"
download_file "Wan2.2_VAE.pth"

echo ""
echo "--- T5 Encoder ---"
download_file "models_t5_umt5-xxl-enc-bf16.pth"

echo ""
echo "--- Tokenizer ---"
for f in \
  "google/umt5-xxl/special_tokens_map.json" \
  "google/umt5-xxl/spiece.model" \
  "google/umt5-xxl/tokenizer.json" \
  "google/umt5-xxl/tokenizer_config.json"; do
  download_file "${f}"
done

echo ""
echo "=== Download complete ==="
echo "Next: convert to MLX"
echo "  uv run python -m mlx_video.models.wan_2.convert --checkpoint-dir ${OUTDIR} --output-dir <target>"
