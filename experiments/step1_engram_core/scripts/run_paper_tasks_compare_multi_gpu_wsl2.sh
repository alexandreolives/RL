#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_paper_tasks_compare_multi_gpu.sh" "$@"
