#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../experiments/step1_engram_core/scripts/run_long_context_compare_ddp.sh" "$@"
