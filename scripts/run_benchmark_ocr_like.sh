#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../experiments/step3_ocr_like/scripts/run_benchmark_ocr_like.sh" "$@"
