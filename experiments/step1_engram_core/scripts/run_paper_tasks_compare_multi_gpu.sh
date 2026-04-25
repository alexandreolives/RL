#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-eval}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"

# Space-separated lists. Example:
#   SEEDS="0 1 2"
#   GPU_IDS="0 1 2"
SEEDS="${SEEDS:-0 1 2}"
GPU_IDS="${GPU_IDS:-0 1 2}"

LIMIT="${LIMIT:-128}"
MAX_LEN="${MAX_LEN:-1024}"
OUT_DIR="${OUT_DIR:-artifacts/paper_tasks_compare_multi}"
CKPT_DIR="${CKPT_DIR:-artifacts/text_lm_compare_fixed}"

read -r -a SEEDS_ARR <<< "${SEEDS}"
read -r -a GPUS_ARR <<< "${GPU_IDS}"

if [ "${#SEEDS_ARR[@]}" -ne "${#GPUS_ARR[@]}" ]; then
  echo "SEEDS and GPU_IDS must have the same length." >&2
  exit 1
fi

pids=()
for i in "${!SEEDS_ARR[@]}"; do
  seed="${SEEDS_ARR[$i]}"
  gpu="${GPUS_ARR[$i]}"
  out="${OUT_DIR}/seed${seed}.json"
  baseline_ckpt="${CKPT_DIR}/baseline_seed${seed}/model.pt"
  engram_ckpt="${CKPT_DIR}/engram_seed${seed}/model.pt"

  echo "[launch] seed=${seed} gpu=${gpu} out=${out}"
  CMD="
    set -euo pipefail
    docker run --rm --gpus \"device=${gpu}\" --ipc=host \
      -v ${REMOTE_REPO}:/workspace \
      ${IMAGE} \
      bash -lc \"\
        cd /workspace && mkdir -p ${OUT_DIR} && \
        PYTHONPATH=src:. python -m eval.transformer.paper_tasks_compare \
          --device cuda \
          --baseline-ckpt ${baseline_ckpt} \
          --engram-ckpt ${engram_ckpt} \
          --limit ${LIMIT} \
          --max-len ${MAX_LEN} \
          --out ${out} \
      \"
  "
  if [ -n "${REMOTE_HOST}" ]; then
    ssh "${REMOTE_HOST}" "bash -lc '${CMD}'" &
  else
    bash -lc "${CMD}" &
  fi
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "At least one parallel run failed." >&2
  exit 1
fi

echo "All parallel paper-task compare runs completed."
