#!/usr/bin/env bash
set -euo pipefail

# Runs a standard paper-style lm-eval suite in Docker on WSL2.

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
MODEL_LABEL="${MODEL_LABEL:-model}"
TASKS="${TASKS:-mmlu,bbh,drop,hellaswag,arc_challenge,gsm8k,math}"
LIMIT="${LIMIT:-}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="/harnesses/results/paper_suite_${MODEL_LABEL}_${TS}.json"

if [ -z "${LIMIT}" ]; then
  REMOTE_HOST="${REMOTE_HOST}" TASKS="${TASKS}" OUTPUT_PATH="${OUT}" ./scripts/run_lm_eval.sh
else
  REMOTE_HOST="${REMOTE_HOST}" TASKS="${TASKS}" LIMIT="${LIMIT}" OUTPUT_PATH="${OUT}" ./scripts/run_lm_eval.sh
fi

echo "Paper suite output: ${OUT}"
