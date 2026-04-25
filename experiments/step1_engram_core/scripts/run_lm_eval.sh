#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-eval}"
HARNESS_ROOT="${HARNESS_ROOT:-\$HOME/RL/harnesses}"
GPU_ID="${GPU_ID:-0}"

# lm-eval model backend (example: hf, local-completions, openai-completions)
LM_MODEL="${LM_MODEL:-hf}"
MODEL_ARGS="${MODEL_ARGS:-pretrained=EleutherAI/pythia-70m,dtype=float16}"
TASKS="${TASKS:-arc_challenge,hellaswag}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
OUTPUT_PATH="${OUTPUT_PATH:-/harnesses/results/lm_eval_$(date +%Y%m%d_%H%M%S).json}"
LIMIT_FLAG=""
if [ -n "${LIMIT}" ]; then
  LIMIT_FLAG="--limit ${LIMIT}"
fi

CMD="
  set -euo pipefail
  mkdir -p ${HARNESS_ROOT}/results
  docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=${GPU_ID} \
    -v ${HARNESS_ROOT}:/harnesses \
    ${IMAGE} \
    bash -lc \"\
      set -euo pipefail; \
      cd /harnesses/lm-evaluation-harness; \
      git config --global --add safe.directory /harnesses/lm-evaluation-harness || true; \
      lm_eval \
        --model ${LM_MODEL} \
        --model_args \\\"${MODEL_ARGS}\\\" \
        --tasks ${TASKS} \
        --device cuda:0 \
        --batch_size ${BATCH_SIZE} \
        ${LIMIT_FLAG} \
        --output_path ${OUTPUT_PATH}; \
      echo Results written to: ${OUTPUT_PATH}\
    \"
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
