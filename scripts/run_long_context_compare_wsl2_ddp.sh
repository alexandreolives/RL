#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_long_context_compare_ddp.sh" "$@"
