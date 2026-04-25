#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-ocr}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"
DATASET="${DATASET:-synthetic_text}"
LIMIT="${LIMIT:-32}"
SEED="${SEED:-0}"
OUT="${OUT:-artifacts/step3_ocr_like/bench_${DATASET}.json}"
GPU_ID="${GPU_ID:-0}"

CMD="
  set -euo pipefail
  cd ${REMOTE_REPO}
  mkdir -p \$(dirname \"${OUT}\")
  docker run --rm -u \$(id -u):\$(id -g) --gpus \"device=${GPU_ID}\" -e CUDA_VISIBLE_DEVICES=0 --ipc=host -v ${REMOTE_REPO}:/workspace ${IMAGE} \
    bash -lc \"cd /workspace && python experiments/step3_ocr_like/scripts/benchmark_ocr_like.py --dataset ${DATASET} --limit ${LIMIT} --seed ${SEED} --out ${OUT}\"
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
