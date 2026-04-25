#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-eval}"
HARNESS_ROOT="${HARNESS_ROOT:-\$HOME/RL/harnesses}"

remote_exec() {
  local cmd="$1"
  if [ -n "${REMOTE_HOST}" ]; then
    ssh "${REMOTE_HOST}" "bash -lc '${cmd}'"
  else
    bash -lc "${cmd}"
  fi
}

echo "[1/4] Preparing harness workspace at ${HARNESS_ROOT}"
remote_exec "mkdir -p ${HARNESS_ROOT}"

echo "[2/4] Cloning/updating harness repositories"
remote_exec '
  set -euo pipefail
  cd ${HARNESS_ROOT}
  if [ ! -d lm-evaluation-harness/.git ]; then
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
  else
    git -C lm-evaluation-harness pull --ff-only
  fi
  if [ ! -d RULER/.git ]; then
    git clone --depth 1 https://github.com/NVIDIA/RULER.git
  else
    git -C RULER pull --ff-only
  fi
  if [ ! -d LongPPL/.git ]; then
    git clone --depth 1 https://github.com/PKU-ML/LongPPL.git
  else
    git -C LongPPL pull --ff-only
  fi
'

echo "[3/4] Checking lm-eval availability in ${IMAGE}"
remote_exec '
  set -euo pipefail
  docker run --rm ${IMAGE} bash -lc \"lm_eval --help >/dev/null\"
'

echo "[4/4] Sanity check repositories mounted with eval image"
remote_exec '
  set -euo pipefail
  docker run --rm -v ${HARNESS_ROOT}:/harnesses ${IMAGE} bash -lc \"\
    set -euo pipefail
    test -d /harnesses/lm-evaluation-harness
    test -d /harnesses/RULER
    test -d /harnesses/LongPPL
    lm_eval --help >/dev/null
    echo \"Harness install OK: lm-eval available in container global Python\"
  \"
'

cat <<EOF
Done (Docker-only runtime for Python deps).
- Harness root: ${HARNESS_ROOT}
- lm-eval runtime image: ${IMAGE}
- Sources:
  - https://github.com/EleutherAI/lm-evaluation-harness
  - https://github.com/NVIDIA/RULER
  - https://github.com/PKU-ML/LongPPL
EOF
