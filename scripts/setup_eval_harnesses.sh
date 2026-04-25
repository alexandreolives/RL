#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../experiments/step1_engram_core/scripts/setup_eval_harnesses.sh" "$@"
