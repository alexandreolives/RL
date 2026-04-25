#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-eval}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"

BASELINE_CKPT="${BASELINE_CKPT:-artifacts/text_lm_compare/baseline_seed0/model.pt}"
ENGRAM_CKPT="${ENGRAM_CKPT:-artifacts/text_lm_compare/engram_seed0/model.pt}"
LIMIT="${LIMIT:-64}"
MAX_LEN="${MAX_LEN:-1024}"
OUT="${OUT:-artifacts/paper_tasks_compare.json}"

CMD="
  set -euo pipefail
  docker run --rm --gpus all --ipc=host \
    -v ${REMOTE_REPO}:/workspace \
    ${IMAGE} \
    bash -lc \"\
      cd /workspace && \
      PYTHONPATH=src:. python -m eval.transformer.paper_tasks_compare \
        --device cuda \
        --baseline-ckpt ${BASELINE_CKPT} \
        --engram-ckpt ${ENGRAM_CKPT} \
        --limit ${LIMIT} \
        --max-len ${MAX_LEN} \
        --out ${OUT} \
    \"
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
