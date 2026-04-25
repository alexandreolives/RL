#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-ocr}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"
INPUT_GLOB="${INPUT_GLOB:-papers/bycloud/*.pdf}"
OUT_DIR="${OUT_DIR:-artifacts/step3_ocr_like/batch_extract}"
MAX_PAGES="${MAX_PAGES:-3}"
GPU_ID="${GPU_ID:-0}"

CMD="
  set -euo pipefail
  cd ${REMOTE_REPO}
  mkdir -p ${OUT_DIR}
  for f in ${INPUT_GLOB}; do
    [ -f \"\$f\" ] || continue
    b=\$(basename \"\$f\")
    out=\"${OUT_DIR}/\${b%.pdf}.json\"
    docker run --rm -u \$(id -u):\$(id -g) --gpus \"device=${GPU_ID}\" -e CUDA_VISIBLE_DEVICES=0 -v ${REMOTE_REPO}:/workspace ${IMAGE} \
      bash -lc \"cd /workspace && python experiments/step3_ocr_like/scripts/extract_doc_ocr_like.py --input \$f --out \$out --max-pages ${MAX_PAGES}\"
  done
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
