#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(pwd)}"
HISTORICAL_ARTIFACTS="${HISTORICAL_ARTIFACTS:-$REPO/artifacts}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-faster-qwen3-tts-openai:latest}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8}"
GPU_IDS="${GPU_IDS:-0 1 2}"
PHASE="${PHASE:-all}"
OUT_ROOT="${OUT_ROOT:-artifacts/lejepa_real_multiseed}"
LIMIT="${LIMIT:-512}"
MAX_LEN="${MAX_LEN:-2048}"

read -r -a seed_array <<< "${SEEDS}"
read -r -a gpu_array <<< "${GPU_IDS}"
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
    --entrypoint python \
    "${IMAGE}" "$@"
}

train_seed() {
  local seed="$1"
  local gpu="$2"
  run_container "${gpu}" -m eval.transformer.train_text_lm_compare \
    --device cuda \
    --variants engram_noconv \
    --seed "${seed}" \
    --seq-len 256 \
    --batch-size 8 \
    --train-steps 400 \
    --eval-steps 64 \
    --input-mode byte \
    --no-byte-patching \
    --byte-patch-size 1 \
    --jepa-mode lejepa \
    --jepa-loss-weight 0.05 \
    --jepa-mask-ratio 0.4 \
    --lejepa-lambda 0.1 \
    --lejepa-num-views 2 \
    --lejepa-num-slices 256 \
    --lejepa-num-knots 17 \
    --lejepa-t-max 5.0 \
    --batch-plan-in "/historical-artifacts/text_lm_compare_det/plan_seed${seed}.json" \
    --out-dir "${OUT_ROOT}/train"
}

eval_seed() {
  local seed="$1"
  local gpu="$2"
  run_container "${gpu}" -m eval.transformer.paper_tasks_compare \
    --device cuda \
    --baseline-ckpt "/historical-artifacts/text_lm_compare_det/baseline_seed${seed}/model.pt" \
    --engram-ckpt "${OUT_ROOT}/train/engram_noconv_seed${seed}/model.pt" \
    --limit "${LIMIT}" \
    --max-len "${MAX_LEN}" \
    --out "${OUT_ROOT}/eval/seed${seed}.json"
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
  all)
    run_waves train
    run_waves eval
    ;;
  *)
    echo "PHASE must be train, eval, or all" >&2
    exit 2
    ;;
esac

echo "[$(date --iso-8601=seconds)] campaign complete phase=${PHASE}"
