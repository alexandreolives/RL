#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_paper_suite.sh" "$@"
