#!/usr/bin/env bash
# Stage 11a: carries animals between machines. Runs on a DGX Spark as the user who can reach the far node.
#   relay.sh LOCAL_DIR USER@FAR FAR_DIR FAR_CONTINENT [LOCAL_CONTINENT ...]
# Files for FAR_CONTINENT go there; files there for a local continent come here. Stops when LOCAL_DIR/STOP appears.
set -u
local_dir="$1"; far="$2"; far_dir="$3"; far_id="$4"; shift 4
while [ ! -e "$local_dir/STOP" ]; do
  if compgen -G "$local_dir/to-$far_id/*.npz" >/dev/null; then
    rsync -aq --chmod=F666 --remove-source-files --exclude='.*' "$local_dir/to-$far_id/" "$far:$far_dir/to-$far_id/" || true
  fi
  for id in "$@"; do
    rsync -aq --chmod=F666 --remove-source-files --exclude='.*' "$far:$far_dir/to-$id/" "$local_dir/to-$id/" 2>/dev/null || true
  done
  sleep 3
done
