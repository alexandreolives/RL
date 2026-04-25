#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_paper_suite_multi_gpu.sh" "$@"
