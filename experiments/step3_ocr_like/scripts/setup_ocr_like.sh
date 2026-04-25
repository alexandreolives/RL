#!/usr/bin/env bash
set -euo pipefail

WSL_HOST="${WSL_HOST:-}"
REMOTE_HOST="${REMOTE_HOST:-${WSL_HOST}}"
IMAGE="${IMAGE:-rl-engram:gpu-ocr}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/RL/engram}"

CMD="
  set -euo pipefail
  cd ${REMOTE_REPO}
  docker build -f docker/Dockerfile.wsl2.gpu.ocr -t ${IMAGE} .
"

if [ -n "${REMOTE_HOST}" ]; then
  ssh "${REMOTE_HOST}" "bash -lc '${CMD}'"
else
  bash -lc "${CMD}"
fi
