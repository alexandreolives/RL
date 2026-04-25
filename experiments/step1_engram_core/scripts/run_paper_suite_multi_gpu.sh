#!/usr/bin/env bash
set -euo pipefail

# Parallel paper-style lm-eval launcher (1 process per GPU).
WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
GPU_IDS="${GPU_IDS:-0 1 2}"
MODEL_LABEL="${MODEL_LABEL:-model}"
LIMIT="${LIMIT:-}"

# Comma-separated task groups, one group per GPU id.
# Example:
#   TASK_GROUPS="mmlu,arc_challenge;bbh,drop;gsm8k,hendrycks_math,hellaswag"
#
# Default is tuned for 3 GPUs with better wall-clock balance on common paper tasks:
#   g0: mmlu,arc_challenge
#   g1: bbh
#   g2: gsm8k,hendrycks_math,drop,hellaswag
TASK_GROUPS="${TASK_GROUPS:-mmlu,arc_challenge;bbh;gsm8k,hendrycks_math,drop,hellaswag}"

TS="$(date +%Y%m%d_%H%M%S)"
read -r -a GPUS_ARR <<< "${GPU_IDS}"
IFS=';' read -r -a GROUPS_ARR <<< "${TASK_GROUPS}"

if [ "${#GPUS_ARR[@]}" -ne "${#GROUPS_ARR[@]}" ]; then
  echo "GPU_IDS and TASK_GROUPS must have the same number of entries." >&2
  exit 1
fi

pids=()
for i in "${!GPUS_ARR[@]}"; do
  gpu="${GPUS_ARR[$i]}"
  tasks="${GROUPS_ARR[$i]}"
  out="/harnesses/results/paper_suite_${MODEL_LABEL}_g${gpu}_${TS}.json"
  echo "[launch] gpu=${gpu} tasks=${tasks} out=${out}"

  if [ -n "${LIMIT}" ]; then
    REMOTE_HOST="${REMOTE_HOST}" GPU_ID="${gpu}" TASKS="${tasks}" LIMIT="${LIMIT}" OUTPUT_PATH="${out}" ./scripts/run_lm_eval.sh &
  else
    REMOTE_HOST="${REMOTE_HOST}" GPU_ID="${gpu}" TASKS="${tasks}" OUTPUT_PATH="${out}" ./scripts/run_lm_eval.sh &
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
  echo "At least one parallel lm-eval run failed." >&2
  exit 1
fi

echo "All parallel runs completed."
