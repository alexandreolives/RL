#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/alexandre/RL-d3a0407}"
HISTORICAL_ARTIFACTS="${HISTORICAL_ARTIFACTS:-/home/alexandre/RL/engram/artifacts}"
HF_CACHE="${HF_CACHE:-/home/alexandre/.cache/huggingface}"
IMAGE="${IMAGE:-faster-qwen3-tts-openai:latest}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8}"
GPU_IDS="${GPU_IDS:-0 1 2}"
PHASE="${PHASE:-train}"
OUT_ROOT="${OUT_ROOT:-artifacts/attnres_engram_v2_multiseed}"
VARIANT="${VARIANT:-engram_noconv_attnres_v2}"
VARIANTS="${VARIANTS:-${VARIANT}}"
MODEL_SIZE="${MODEL_SIZE:-tiny}"
TRAIN_STEPS="${TRAIN_STEPS:-400}"
EVAL_STEPS="${EVAL_STEPS:-64}"
BATCH_PLAN_IN="${BATCH_PLAN_IN:-/historical-artifacts/text_lm_compare_det/plan_seed}"
LIMIT="${LIMIT:-512}"
MAX_LEN="${MAX_LEN:-2048}"

read -r -a seed_array <<< "${SEEDS}"
read -r -a gpu_array <<< "${GPU_IDS}"
read -r -a variant_array <<< "${VARIANTS}"
if [[ "${#gpu_array[@]}" -eq 0 ]]; then
  echo "GPU_IDS cannot be empty" >&2
  exit 2
fi

mkdir -p "${REPO}/${OUT_ROOT}/logs" "${REPO}/${OUT_ROOT}/train" "${REPO}/${OUT_ROOT}/eval"

run_container() {
  local gpu="$1"
  shift
  docker run --rm --gpus "device=${gpu}" --ipc=host \
    -v "${REPO}:/workspace" \
    -v "${HISTORICAL_ARTIFACTS}:/historical-artifacts:ro" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -w /workspace \
    -e PYTHONPATH=/workspace/.remote-deps:/workspace:/workspace/src \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    --entrypoint python \
    "${IMAGE}" "$@"
}

train_seed() {
  local seed="$1"
  local gpu="$2"
  local plan_args=()
  if [[ -n "${BATCH_PLAN_IN}" ]]; then
    plan_args=(--batch-plan-in "${BATCH_PLAN_IN}${seed}.json")
  fi
  run_container "${gpu}" -m eval.transformer.train_text_lm_compare \
    --device cuda \
    --variants "${variant_array[@]}" \
    --model-size "${MODEL_SIZE}" \
    --seed "${seed}" \
    --seq-len 256 \
    --batch-size 8 \
    --train-steps "${TRAIN_STEPS}" \
    --eval-steps "${EVAL_STEPS}" \
    --input-mode byte \
    --no-byte-patching \
    --byte-patch-size 1 \
    "${plan_args[@]}" \
    --out-dir "${OUT_ROOT}/train"
}

eval_seed() {
  local seed="$1"
  local gpu="$2"
  run_container "${gpu}" -m eval.transformer.paper_tasks_compare \
    --device cuda \
    --baseline-ckpt "/historical-artifacts/text_lm_compare_det/engram_noconv_seed${seed}/model.pt" \
    --engram-ckpt "${OUT_ROOT}/train/${VARIANT}_seed${seed}/model.pt" \
    --limit "${LIMIT}" \
    --max-len "${MAX_LEN}" \
    --out "${OUT_ROOT}/eval/v2_vs_engram_seed${seed}.json"
}

run_waves() {
  local action="$1"
  local wave_size="${#gpu_array[@]}"
  local start
  for ((start = 0; start < ${#seed_array[@]}; start += wave_size)); do
    local pids=()
    local labels=()
    local offset
    for ((offset = 0; offset < wave_size && start + offset < ${#seed_array[@]}; offset++)); do
      local seed="${seed_array[$((start + offset))]}"
      local gpu="${gpu_array[${offset}]}"
      local log="${REPO}/${OUT_ROOT}/logs/${action}_seed${seed}.log"
      echo "[$(date --iso-8601=seconds)] launch ${action} seed=${seed} gpu=${gpu} log=${log}"
      "${action}_seed" "${seed}" "${gpu}" >"${log}" 2>&1 &
      pids+=("$!")
      labels+=("${action}:seed${seed}:gpu${gpu}")
    done
    local idx
    for idx in "${!pids[@]}"; do
      if wait "${pids[${idx}]}"; then
        echo "[$(date --iso-8601=seconds)] complete ${labels[${idx}]}"
      else
        echo "[$(date --iso-8601=seconds)] FAILED ${labels[${idx}]}" >&2
        return 1
      fi
    done
  done
}

case "${PHASE}" in
  train)
    run_waves train
    ;;
  eval)
    run_waves eval
    ;;
  *)
    echo "PHASE must be train or eval" >&2
    exit 2
    ;;
esac

echo "[$(date --iso-8601=seconds)] v2 campaign complete phase=${PHASE}"
