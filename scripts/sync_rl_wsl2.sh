#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sync_rl_wsl2.sh [--dry-run]

Sync the local RL tree to the WSL2 RL tree over ssh.

Behavior:
- tracked files are synced by applying a git binary patch
- untracked visible files are copied with rsync
- nothing is deleted on either side
- the script refuses to run if the local and remote HEAD commits differ

Defaults use the current directory as the source and `$DEST_PATH` (or
`/home/$USER/RL/`) on host `wsl2` as the destination. Override with
`REPO_ROOT`, `DEST_HOST`, and `DEST_PATH` when needed.

The destination is created if missing.
EOF
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi

if [[ $# -ne 0 ]]; then
  usage
  exit 1
fi

repo_root="${REPO_ROOT:-$(pwd)}"
dest_host="${DEST_HOST:-wsl2}"
dest_path="${DEST_PATH:-/home/$USER/RL/}"

local_head="$(git -C "$repo_root" rev-parse HEAD)"
remote_head="$(ssh "$dest_host" "git -C '$dest_path' rev-parse HEAD")"

if [[ "$local_head" != "$remote_head" ]]; then
  printf 'HEAD mismatch: local=%s remote=%s\n' "$local_head" "$remote_head" >&2
  exit 1
fi

remote_tracked_dirty="$(
  ssh "$dest_host" "git -C '$dest_path' status --porcelain --untracked-files=no"
)"
if [[ -n "$remote_tracked_dirty" ]]; then
  printf 'Remote working tree has tracked modifications; refusing to sync.\n' >&2
  printf '%s\n' "$remote_tracked_dirty" >&2
  exit 1
fi

if [[ "$dry_run" -eq 1 ]]; then
  if ! git -C "$repo_root" diff --binary --quiet --ignore-submodules HEAD --; then
    git -C "$repo_root" diff --binary --no-color --ignore-submodules HEAD -- | \
      ssh "$dest_host" "cd '$dest_path' && git apply --check --binary --whitespace=nowarn"
  fi
  git -C "$repo_root" ls-files -z -o --exclude-standard | \
    rsync -aH --dry-run --info=progress2 --stats --from0 --files-from=- --relative -e ssh "$repo_root/./" "${dest_host}:${dest_path}"
  exit 0
fi

# Apply tracked changes as a binary patch so the working tree moves without commits.
if ! git -C "$repo_root" diff --binary --quiet --ignore-submodules HEAD --; then
  git -C "$repo_root" diff --binary --no-color --ignore-submodules HEAD -- | \
    ssh "$dest_host" "cd '$dest_path' && git apply --binary --whitespace=nowarn"
fi

ssh "$dest_host" "mkdir -p ${dest_path%/}"
# Copy only visible untracked files. Tracked files are handled by git apply.
git -C "$repo_root" ls-files -z -o --exclude-standard | \
  rsync -aH --info=progress2 --stats --from0 --files-from=- --relative -e ssh "$repo_root/./" "${dest_host}:${dest_path}"
