#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-eval}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"

VARIANTS="${VARIANTS:-baseline engram}"
TRAIN_STEPS="${TRAIN_STEPS:-400}"
EVAL_STEPS="${EVAL_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-256}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-artifacts/text_lm_compare}"
INPUT_MODE="${INPUT_MODE:-byte}"
BYTE_PATCHING="${BYTE_PATCHING:-false}"
BYTE_PATCH_SIZE="${BYTE_PATCH_SIZE:-1}"
JEPA_MODE="${JEPA_MODE:-none}"
JEPA_LOSS_WEIGHT="${JEPA_LOSS_WEIGHT:-0.0}"
JEPA_MASK_RATIO="${JEPA_MASK_RATIO:-0.4}"
JEPA_PROJ_DIM="${JEPA_PROJ_DIM:-256}"
LEJEPA_ISOTROPY_WEIGHT="${LEJEPA_ISOTROPY_WEIGHT:-0.1}"
if [ "${BYTE_PATCHING}" = "true" ]; then
  BYTE_PATCH_FLAG="--byte-patching"
else
  BYTE_PATCH_FLAG="--no-byte-patching"
fi

CMD="
  set -euo pipefail
  docker run --rm --gpus all --ipc=host \
    -v ${REMOTE_REPO}:/workspace \
    ${IMAGE} \
    bash -lc \"\
      cd /workspace && \
      PYTHONPATH=src:. python -m eval.transformer.train_text_lm_compare \
        --device cuda \
        --variants ${VARIANTS} \
        --train-steps ${TRAIN_STEPS} \
        --eval-steps ${EVAL_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --seq-len ${SEQ_LEN} \
        --seed ${SEED} \
        --input-mode ${INPUT_MODE} \
        --byte-patch-size ${BYTE_PATCH_SIZE} \
        ${BYTE_PATCH_FLAG} \
        --jepa-mode ${JEPA_MODE} \
        --jepa-loss-weight ${JEPA_LOSS_WEIGHT} \
        --jepa-mask-ratio ${JEPA_MASK_RATIO} \
        --jepa-proj-dim ${JEPA_PROJ_DIM} \
        --lejepa-isotropy-weight ${LEJEPA_ISOTROPY_WEIGHT} \
        --out-dir ${OUT_DIR} \
    \"
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
