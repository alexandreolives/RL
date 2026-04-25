#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../experiments/step1_engram_core/scripts/run_paper_tasks_compare.sh" "$@"
