#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"
NPROC="${NPROC:-3}"

TRAIN_TASK="${TRAIN_TASK:-passkey}"
EVAL_TASKS="${EVAL_TASKS:-passkey multi_query variable_tracking}"
VARIANTS="${VARIANTS:-baseline engram}"

INPUT_MODE="${INPUT_MODE:-symbolic}"
MODEL_SIZE="${MODEL_SIZE:-tiny}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH="${BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"
EVAL_STEPS="${EVAL_STEPS:-8}"
TRAIN_CACHE_SIZE="${TRAIN_CACHE_SIZE:-32}"
SEEDS="${SEEDS:-0 1}"
ATTN_BACKEND="${ATTN_BACKEND:-flash}"

RUN_CMD="cd /workspace && PYTHONPATH=src:. torchrun --standalone --nproc_per_node=${NPROC} eval/transformer/train_long_context_compare.py \
  --device cuda \
  --input-mode ${INPUT_MODE} \
  --attention-backend ${ATTN_BACKEND} \
  --train-task ${TRAIN_TASK} \
  --eval-tasks ${EVAL_TASKS} \
  --variants ${VARIANTS} \
  --model-size ${MODEL_SIZE} \
  --seq-len ${SEQ_LEN} \
  --batch ${BATCH} \
  --grad-accum ${GRAD_ACCUM} \
  --train-steps ${TRAIN_STEPS} \
  --eval-steps ${EVAL_STEPS} \
  --train-cache-size ${TRAIN_CACHE_SIZE} \
  --seeds ${SEEDS}"

CMD="
  set -euo pipefail
  docker run --rm --gpus all --ipc=host \
    -v ${REMOTE_REPO}:/workspace \
    ${IMAGE} \
    bash -lc \"${RUN_CMD}\"
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
