#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../experiments/step1_engram_core/scripts/run_paper_suite_multi_gpu.sh" "$@"
