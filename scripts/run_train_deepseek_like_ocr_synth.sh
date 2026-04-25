#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../experiments/step3_ocr_like/scripts/run_train_deepseek_like_ocr_synth.sh" "$@"
