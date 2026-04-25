#!/usr/bin/env bash
set -euo pipefail

# Repro grid for Engram variant comparison on the WSL2 Docker setup.
# Assumes:
# - repo is available in WSL2 (default: $HOME/RL/engram)
# - docker image tag is rl-engram:gpu (override with IMAGE)

TASKS=("passkey" "multi_query" "variable_tracking")
VARIANTS=(
  "engram"
  "engram_layerhash"
  "engram_compress"
  "engram_official_gate"
  "engram_noconv"
  "engram_fullconv"
)

TRAIN_STEPS="${TRAIN_STEPS:-300}"
EVAL_STEPS="${EVAL_STEPS:-8}"
TRAIN_CACHE_SIZE="${TRAIN_CACHE_SIZE:-64}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH="${BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SEEDS="${SEEDS:-0 1}"
ATTN_BACKEND="${ATTN_BACKEND:-flash}"
INPUT_MODE="${INPUT_MODE:-symbolic}"
MODEL_SIZE="${MODEL_SIZE:-tiny}"
WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"
IMAGE="${IMAGE:-rl-engram:gpu}"

for task in "${TASKS[@]}"; do
  echo "===== TASK: ${task} ====="
  CMD="
    cd ${REMOTE_REPO}
    docker run --rm --gpus all --ipc=host -v ${REMOTE_REPO}:/workspace ${IMAGE} \
      bash -lc \"cd /workspace && PYTHONPATH=src:. python -m eval.transformer.train_long_context_compare \
        --device cuda --attention-backend ${ATTN_BACKEND} --input-mode ${INPUT_MODE} --model-size ${MODEL_SIZE} \
        --train-task ${task} --eval-tasks ${task} --variants ${VARIANTS[*]} --seq-len ${SEQ_LEN} --batch ${BATCH} \
        --grad-accum ${GRAD_ACCUM} --train-steps ${TRAIN_STEPS} --eval-steps ${EVAL_STEPS} \
        --train-cache-size ${TRAIN_CACHE_SIZE} --seeds ${SEEDS}\"
  "
  if [ -n "${REMOTE_HOST}" ]; then
    ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
  else
    bash -lc "${CMD}"
  fi
done
