#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-ocr}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"
GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-engram_noconv}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-artifacts/step3_ocr_like/train_${VARIANT}_seed${SEED}}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1024}"
EVAL_SAMPLES="${EVAL_SAMPLES:-128}"
MAX_TARGET_BYTES="${MAX_TARGET_BYTES:-192}"

CMD="
  set -euo pipefail
  cd ${REMOTE_REPO}
  mkdir -p ${OUT_DIR}
  docker run --rm -u \$(id -u):\$(id -g) --gpus \"device=${GPU_ID}\" -e CUDA_VISIBLE_DEVICES=0 --ipc=host -v ${REMOTE_REPO}:/workspace ${IMAGE} \
    bash -lc \"cd /workspace && PYTHONPATH=src:. python experiments/step3_ocr_like/scripts/train_deepseek_like_ocr_synth.py \
      --variant ${VARIANT} \
      --seed ${SEED} \
      --out-dir ${OUT_DIR} \
      --train-steps ${TRAIN_STEPS} \
      --batch-size ${BATCH_SIZE} \
      --lr ${LR} \
      --image-size ${IMAGE_SIZE} \
      --train-samples ${TRAIN_SAMPLES} \
      --eval-samples ${EVAL_SAMPLES} \
      --max-target-bytes ${MAX_TARGET_BYTES}\"
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
