#!/usr/bin/env bash
# Runs ON the GPU machine, from its copy of the repo. Only ever touches containers named autolearn-*.
# Settings: AUTOLEARN_IMAGE, AUTOLEARN_DASH_IP, HEADROOM_GB, EVAL_GB, MEMINFO.
set -euo pipefail
cd "$(dirname "$0")/.."
APP="$PWD"
IMAGE="${AUTOLEARN_IMAGE:-autolearn:latest}"
RUN_AS="$(id -u):$(id -g)"
GPU=(--gpus all)
# A node whose container GPU setup is out of date, and no sudo to refresh it (one of our machines): deploy/.gpu_flags
# passes the devices and driver libraries to the container by hand. It is made on that node and never synced.
if [ -f deploy/.gpu_flags ]; then read -r -a GPU < deploy/.gpu_flags; GPU+=(-e LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu); for g in $(id -G); do GPU+=(--group-add "$g"); done; fi
COMMON=("${GPU[@]}" --user "$RUN_AS" -e HOME=/tmp -v "$APP":/app -w /app)
# Stage 11a: where animals wait to cross. Outside any home directory, since the relay runs as another user.
if [ -d /var/tmp/autolearn-transit ]; then COMMON+=(-v /var/tmp/autolearn-transit:/transit); fi
MEMINFO="${MEMINFO:-/proc/meminfo}"
HEADROOM_GB="${HEADROOM_GB:-8}"  # always leave this much of the Spark's memory free (GiB)

# GiB that running trainers may still claim: each one's cap (label autolearn.mem_gb) minus what it uses.
reserved_gb() {
  local n
  for n in $(docker ps --filter label=autolearn.mem_gb --format '{{.Names}}'); do
    echo "$(docker inspect -f '{{index .Config.Labels "autolearn.mem_gb"}}' "$n")" \
         "$(docker stats --no-stream --format '{{.MemUsage}}' "$n" | awk '{print $1}')"
  done | awk '{u = $2; f = 1 / 1073741824
               if (u ~ /GiB$/) f = 1; else if (u ~ /MiB$/) f = 1 / 1024; else if (u ~ /KiB$/) f = 1 / 1048576
               sub(/[A-Za-z]+$/, "", u); r = $1 - u * f; if (r > 0) s += r}
         END {printf "%.1f\n", s + 0}'
}
total_gb() { awk '/^MemTotal:/ {printf "%.1f\n", $2 / 1048576}' "$MEMINFO"; }
# What a new trainer may cap itself at: available memory, less headroom and other trainers' growth.
budget_gb() {
  awk -v h="$HEADROOM_GB" -v r="$1" '/^MemAvailable:/ {printf "%.1f\n", $2 / 1048576 - h - r}' "$MEMINFO"
}

# Refuse a container whose memory cap would cut into the headroom; set FRAC for its XLA cap.
check_budget() {
  local mem_gb="$1" reserved budget
  reserved="$(reserved_gb)"
  budget="$(budget_gb "$reserved")"
  if awk -v m="$mem_gb" -v b="$budget" 'BEGIN {exit !(m > b)}'; then
    echo "refusing: $mem_gb GiB is over the budget of $budget GiB" \
         "(available less ${HEADROOM_GB} GiB headroom and $reserved GiB other trainers may still use)" >&2
    exit 1
  fi
  FRAC="$(awk -v m="$mem_gb" -v t="$(total_gb)" 'BEGIN {printf "%.4f", m / t}')"
}
EVAL_GB="${EVAL_GB:-8}"

cmd="${1:-help}"
shift || true
case "$cmd" in
  build)
    docker build -t "$IMAGE" deploy ;;
  smoke)
    docker run --rm "${COMMON[@]}" "$IMAGE" python -m scripts.gpu_smoke ;;
  budget)
    reserved="$(reserved_gb)"
    echo "budget_gb=$(budget_gb "$reserved") total_gb=$(total_gb) headroom_gb=$HEADROOM_GB reserved_gb=$reserved" ;;
  train)
    # train [--slot NAME] [--mem-gb GIB] [train.py args]; each slot is its own container.
    slot=""; mem_gb=12
    while [ $# -gt 0 ]; do
      case "$1" in
        --slot) slot="$2"; shift 2 ;;
        --mem-gb) mem_gb="$2"; shift 2 ;;
        *) break ;;
      esac
    done
    if [ -n "$slot" ] && ! [[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$ ]]; then
      echo "bad --slot name: $slot" >&2; exit 1
    fi
    name="autolearn-train${slot:+-$slot}"
    if [ -n "$(docker ps -q --filter name="^$name\$")" ]; then
      echo "$name is already running; stop it first" >&2
      exit 1
    fi
    check_budget "$mem_gb"
    limit="$(awk -v m="$mem_gb" 'BEGIN {printf "%.2f", m * 1.073741824}')"  # GiB to the GB train.py checks
    docker rm "$name" >/dev/null 2>&1 || true
    mkdir -p runs
    docker run -d --name "$name" --label autolearn.mem_gb="$mem_gb" -e XLA_PYTHON_CLIENT_MEM_FRACTION="$FRAC" \
      "${COMMON[@]}" "$IMAGE" python -m scripts.train --mem-limit-gb "$limit" "$@" ;;
  eval|league|compare|species|chase|signals|alarm|hills|episode3d|ambush|lifetime|scatter|meals|omnivore|persist|lasting|mapbench|transplant|train_indiv|eval_indiv|cross_indiv|zone_births|cross_shared|office|packs|pack_lineage)
    check_budget "$EVAL_GB"
    docker run --rm -e XLA_PYTHON_CLIENT_MEM_FRACTION="$FRAC" "${COMMON[@]}" "$IMAGE" python -m "scripts.$cmd" "$@" ;;
  dash)
    # The dashboard has no login, so it only binds a private address: AUTOLEARN_DASH_IP, or the Tailscale one.
    ip="${AUTOLEARN_DASH_IP:-$(tailscale ip -4 2>/dev/null | head -n1 || true)}"
    if [ -z "$ip" ]; then
      echo "no AUTOLEARN_DASH_IP and no Tailscale IPv4 address; refusing to publish the dashboard on other interfaces" >&2
      exit 1
    fi
    mkdir -p runs
    docker rm -f autolearn-dash >/dev/null 2>&1 || true
    # Serves runs/ and rebuilds the home page (runs list + roadmap) every minute.
    docker run -d --name autolearn-dash --restart unless-stopped --user "$RUN_AS" \
      -p "$ip:9001:9001" -v "$APP":/app:ro -v "$APP/runs":/app/runs -w /app "$IMAGE" \
      sh -c 'python -m scripts.site --runs-dir runs --every 60 & exec python -m http.server 9001 --directory runs'
    echo "dashboard: http://$ip:9001/" ;;
  logs)
    # logs [n] [slot]
    docker logs --tail "${1:-40}" "autolearn-train${2:+-$2}" ;;
  status)
    docker ps -a --filter name=autolearn- --format '{{.Names}}\t{{.Status}}' ;;
  stop)
    # stop [slot] stops one trainer; plain stop stops every trainer and the dashboard.
    if [ -n "${1:-}" ]; then
      docker stop "autolearn-train-$1" 2>/dev/null || true
    else
      docker ps -q --filter name='^autolearn-' | xargs -r docker stop
    fi ;;
  *)
    echo "usage: remote.sh build|smoke|budget|train [--slot S] [--mem-gb G] [args]|eval [args]|league [args]|compare [args]|species [args]|chase [args]|signals [args]|hills [args]|episode3d [args]|ambush [args]|lifetime [args]|scatter [args]|meals [args]|omnivore [args]|persist [args]|lasting [args]|mapbench [args]|dash|logs [n] [slot]|status|stop [slot]" >&2
    exit 1 ;;
esac
