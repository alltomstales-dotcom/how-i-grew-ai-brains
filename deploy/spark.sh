#!/usr/bin/env bash
# Local driver: copies the repo to your GPU machine, or runs deploy/remote.sh there.
# AUTOLEARN_HOST (required): SSH host of the machine. AUTOLEARN_REMOTE_DIR: where the code goes, default autolearn in its home.
set -euo pipefail
HOST="${AUTOLEARN_HOST:?set AUTOLEARN_HOST to the SSH host of your GPU machine}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${AUTOLEARN_REMOTE_DIR:-autolearn}"
cmd="${1:-help}"
if [ "$cmd" = sync ]; then
  tar -C "$ROOT" --exclude=./runs --exclude=./.git --exclude=./.venv \
      --exclude='__pycache__' --exclude='*.pyc' -czf - . \
    | ssh "$HOST" "mkdir -p $(printf '%q' "$DIR") && tar -C $(printf '%q' "$DIR") -xzf -"
  echo "synced to $HOST:$DIR"
else
  ssh "$HOST" "bash $(printf '%q' "$DIR")/deploy/remote.sh $(printf '%q ' "$@")"
fi
