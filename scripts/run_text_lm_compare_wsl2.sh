#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_text_lm_compare.sh" "$@"
