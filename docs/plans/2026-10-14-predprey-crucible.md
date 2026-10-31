# Predator–Prey Crucible Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-JAX predator–prey world where two species learn by PPO on a DGX Spark's GPU, with MP4 clips and metric curves on a tailnet dashboard, and a scripted trained-vs-random evaluation.

**Architecture:** `simulations/predprey.py` is a static-shape, jit/vmap-able world (reset, step, observe). `agents/` holds a small conv actor-critic, PPO in the PureJaxRL style that trains both species at once over agent slots, checkpoints, policies and an evaluator. `viz/` renders states to MP4 and holds a static dashboard. `scripts/train.py` and `scripts/eval.py` are the entry points. `deploy/` builds an NVIDIA JAX container on a DGX Spark and runs everything as `autolearn-*` containers.

**Tech Stack:** Python 3.12, JAX 0.11 (CPU locally, CUDA in `nvcr.io/nvidia/jax:26.09-py3` on a DGX Spark), Flax linen, Optax, NumPy, Pillow, imageio + imageio-ffmpeg, pytest, uv, Docker, Chart.js 4.4.1.

**Spec:** `docs/specs/2026-10-14-predprey-crucible-design.md`

## Global Constraints

- Repo root: `<repo root>`. Every local command starts with `cd <repo root> &&` (the shell resets its directory between calls).
- Local Python runs through uv with the environment on the Linux filesystem: prefix commands with `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run`.
- Tests are CPU-only and must pass locally: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest -q`.
- All world arrays have static shapes. Positions are `(x, y)` int32; 2-D maps are indexed `[y, x]`.
- Memory budget on a DGX Spark: **12 GB**. Enforced by `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` (of 121 GB) and the `check_memory` guard (exit code 3).
- On a DGX Spark: SSH host `$AUTOLEARN_HOST`, code at `~/autolearn`. Touch only containers named `autolearn-train` and `autolearn-dash`. No sudo, no systemd. Never stop, restart or remove any other container. Gemma on :8090 must stay up.
- The dashboard binds port 9001 on a DGX Spark's Tailscale IP only, never `0.0.0.0`.
- If the GPU smoke test (Task 1) fails, STOP and report to the user before any fallback to PyTorch (spec: "tell the user before switching").
- Every commit message ends with these two lines (pass them as a second `-m`):
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx
  ```

## Review Focus

1. **A slot freed by a death and refilled by a birth in the same step.** The dying agent's transition must still be terminal with the −1 reward, or GAE leaks value from the newborn into the dead agent. Pinned by `test_slot_freed_by_death_and_refilled_still_ends_trajectory` (Task 3).
2. **A species goes extinct mid-rollout.** The world must auto-reset, and metrics must stay finite (no NaN from an empty "nearest predator" set). Pinned by `test_nearest_dist_handles_empty_and_wraps`, `test_autoreset_after_done_gives_fresh_world` (Task 3) and the finite-metrics assertion in `test_trainer_smoke_tiny` (Task 5).
3. **Re-running with an existing or unsafe `--run-id`.** It must refuse rather than overwrite a previous run or write outside `runs/`. Pinned by `test_existing_run_id_is_refused` and `test_bad_run_id_is_refused` (Task 7).
4. **Device memory stats missing on GB10 unified memory** (`memory_stats()` may return `None`). The guard must not crash, and an over-limit peak must abort with exit code 3. Pinned by `test_check_memory_handles_missing_stats` and `test_check_memory_over_limit` (Task 7).
5. **A clip episode that ends early by extinction.** It must write a shorter clip rather than crash or pad. Pinned by `test_clip_stops_early_when_episode_ends` (Task 6).

The dashboard must also tolerate a half-written last line in `metrics.jsonl`. `parseJsonl` skips unparsable lines, and `clips.json` is written atomically. Task 8 checks this by hand in a browser.

---

### Task 1: Scaffold and GPU feasibility on a DGX Spark

**Files:**
- Create: `pyproject.toml`, `simulations/__init__.py`, `agents/__init__.py`, `viz/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`
- Create: `scripts/gpu_smoke.py`
- Create: `deploy/Dockerfile`, `deploy/remote.sh`, `deploy/spark.sh`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `deploy/spark.sh sync|build|smoke|train ARGS|eval ARGS|dash|logs [N]|status|stop`, used by Task 8. `deploy/spark.sh <cmd>` runs `deploy/remote.sh <cmd>` on a DGX Spark.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "autolearn"
version = "0.1.0"
description = "Predator-prey Crucible: emergent behaviour from pure RL in JAX"
requires-python = ">=3.11"
dependencies = [
    "jax>=0.7",
    "flax>=0.10",
    "optax>=0.2",
    "numpy>=1.26",
    "pillow>=10",
    "imageio>=2.34",
    "imageio-ffmpeg>=0.5",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create the empty package files: `simulations/__init__.py`, `agents/__init__.py`, `viz/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`. Replace `.gitignore` with:

```
runs/
__pycache__/
*.pyc
.venv/
*.tmp
```

- [ ] **Step 2: Create the local environment and check imports**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv sync && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run python -c "import jax, flax, optax, imageio_ffmpeg; print(jax.__version__, jax.devices(), imageio_ffmpeg.get_ffmpeg_exe())"`
Expected: a JAX version ≥ 0.7, `[CpuDevice(id=0)]`, and a path to an ffmpeg binary.

- [ ] **Step 3: Write `scripts/gpu_smoke.py`**

```python
"""GPU feasibility check for a DGX Spark: JAX must see the GB10 and run jit + vmap."""
from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp


def main() -> int:
    print("jax", jax.__version__, flush=True)
    devices = jax.devices()
    print("devices", devices, flush=True)
    if devices[0].platform != "gpu":
        print("FAIL: no GPU device visible to JAX", flush=True)
        return 1

    x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
    f = jax.jit(lambda a: (a @ a.T).sum())
    f(x).block_until_ready()
    t0 = time.time()
    f(x).block_until_ready()
    print(f"matmul 4096x4096: {time.time() - t0:.4f} s", flush=True)

    grids = jnp.zeros((4096, 48, 48), jnp.int32)
    shifts = jnp.arange(4096) % 48
    roll = jax.jit(jax.vmap(lambda g, s: jnp.roll(g + 1, s, axis=0)))
    roll(grids, shifts).block_until_ready()
    t0 = time.time()
    for _ in range(100):
        grids = roll(grids, shifts)
    grids.block_until_ready()
    print(f"vmapped grid steps/s: {4096 * 100 / (time.time() - t0):,.0f}", flush=True)

    import flax
    import imageio_ffmpeg
    import optax

    print("flax", flax.__version__, "optax", optax.__version__, flush=True)
    print("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe(), flush=True)
    print("memory_stats", devices[0].memory_stats(), flush=True)
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `deploy/Dockerfile`**

```dockerfile
FROM nvcr.io/nvidia/jax:26.09-py3

# Keep the container's own JAX build (made for this GPU); pin it while adding project deps.
RUN pip list --format=freeze | grep -iE '^(jax|jaxlib|jax[-_]cuda[0-9]*[-_a-z]*)==' > /tmp/pins.txt \
 && cat /tmp/pins.txt \
 && pip install --no-cache-dir -c /tmp/pins.txt flax optax "imageio>=2.34" "imageio-ffmpeg>=0.5" pillow

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
```

- [ ] **Step 5: Write `deploy/remote.sh` (runs on a DGX Spark)**

```bash
#!/usr/bin/env bash
# Runs ON a DGX Spark from ~/autolearn. Only ever touches containers named autolearn-*.
set -euo pipefail
cd "$(dirname "$0")/.."
APP="$PWD"
IMAGE=autolearn:latest
RUN_AS="$(id -u):$(id -g)"
COMMON=(--gpus all --user "$RUN_AS" -e HOME=/tmp -v "$APP":/app -w /app)

cmd="${1:-help}"
shift || true
case "$cmd" in
  build)
    docker build -t "$IMAGE" deploy ;;
  smoke)
    docker run --rm "${COMMON[@]}" "$IMAGE" python -m scripts.gpu_smoke ;;
  train)
    if [ -n "$(docker ps -q --filter name='^autolearn-train$')" ]; then
      echo "autolearn-train is already running; run 'stop' first" >&2
      exit 1
    fi
    docker rm autolearn-train >/dev/null 2>&1 || true
    mkdir -p runs
    docker run -d --name autolearn-train "${COMMON[@]}" "$IMAGE" python -m scripts.train "$@" ;;
  eval)
    docker run --rm "${COMMON[@]}" "$IMAGE" python -m scripts.eval "$@" ;;
  dash)
    mkdir -p runs
    docker rm -f autolearn-dash >/dev/null 2>&1 || true
    ip="$(tailscale ip -4 | head -n1)"
    docker run -d --name autolearn-dash --restart unless-stopped --user "$RUN_AS" \
      -p "$ip:9001:9001" -v "$APP/runs":/runs:ro -w /runs "$IMAGE" python -m http.server 9001
    echo "dashboard: http://$ip:9001/" ;;
  logs)
    docker logs --tail "${1:-40}" autolearn-train ;;
  status)
    docker ps -a --filter name=autolearn- --format '{{.Names}}\t{{.Status}}' ;;
  stop)
    docker stop autolearn-train autolearn-dash 2>/dev/null || true ;;
  *)
    echo "usage: remote.sh build|smoke|train [args]|eval [args]|dash|logs [n]|status|stop" >&2
    exit 1 ;;
esac
```

- [ ] **Step 6: Write `deploy/spark.sh` (runs locally)**

```bash
#!/usr/bin/env bash
# Local driver: copies the repo to a DGX Spark, or runs deploy/remote.sh there.
set -euo pipefail
HOST="${AUTOLEARN_HOST:?set AUTOLEARN_HOST to the SSH host of your GPU machine}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmd="${1:-help}"
if [ "$cmd" = sync ]; then
  tar -C "$ROOT" --exclude=./runs --exclude=./.git --exclude=./.venv \
      --exclude='__pycache__' --exclude='*.pyc' -czf - . \
    | ssh "$HOST" 'mkdir -p ~/autolearn && tar -C ~/autolearn -xzf -'
  echo "synced to $HOST:~/autolearn"
else
  ssh "$HOST" "bash ~/autolearn/deploy/remote.sh $(printf '%q ' "$@")"
fi
```

Run: `chmod +x deploy/*.sh`

- [ ] **Step 7: Sync, build and run the smoke test on a DGX Spark**

Run: `deploy/spark.sh sync && deploy/spark.sh build`
Expected: the image pull (several GB, minutes) and then `naming to docker.io/library/autolearn:latest`. The `cat /tmp/pins.txt` output lists the container's jax packages.

Run: `deploy/spark.sh smoke`
Expected: `devices [CudaDevice(id=0)]` (or similar), a matmul time, the vmapped steps/s, flax/optax versions, an ffmpeg path, a `memory_stats` line (record whether it is `None`), and `OK`.
If it prints `FAIL` or errors: STOP and report the full output to the user (Global Constraints).

- [ ] **Step 8: Confirm nothing else on a DGX Spark changed**

Run: `ssh $AUTOLEARN_HOST 'docker ps --format "{{.Names}} {{.Status}}"'; curl -s -o /dev/null -w '%{http_code}\n' http://<tailscale-ip>:8090/v1/models`
Expected: the same non-autolearn containers as before (terrain-diff, sparkproxy-pg, zq-imagegen) still up, and `200` from Gemma.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "chore: scaffold project and a DGX Spark GPU deploy" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 2: World state, reset, movement and observation

**Files:**
- Create: `simulations/predprey.py`
- Test: `tests/test_world.py`

**Interfaces:**
- Produces (used by every later task):
  - `NUM_ACTIONS = 5`, `OBS_CHANNELS = 4`, `DELTAS` (5×2 int32, (dx, dy): stay, north, south, east, west)
  - `WorldConfig` (frozen dataclass; fields below)
  - `Species(pos: (N,2) int32, energy: (N,) f32, alive: (N,) bool)`, `State(prey: Species, pred: Species, grass: (size,size) int32, t: () int32)`
  - `reset(cfg, key) -> State`
  - `make_state(cfg, prey=(), pred=(), grass=None, t=0) -> State` (agents as `(x, y, energy)`; `grass=None` = all grown)
  - `move(pos, actions, size) -> pos`
  - `observe(cfg, state) -> (prey_obs (max_prey, view, view, 4) bf16, pred_obs (max_pred, view, view, 4) bf16)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_world.py
import jax
import jax.numpy as jnp
import numpy as np

from simulations import predprey as pp

CFG = pp.WorldConfig()


def test_reset_counts_and_energy():
    s = pp.reset(CFG, jax.random.PRNGKey(0))
    assert int(s.prey.alive.sum()) == CFG.init_prey
    assert int(s.pred.alive.sum()) == CFG.init_pred
    assert np.all(np.asarray(s.prey.energy)[: CFG.init_prey] == CFG.prey_start_energy)
    assert np.all(np.asarray(s.prey.energy)[CFG.init_prey:] == 0.0)
    assert s.prey.pos.shape == (CFG.max_prey, 2)
    assert s.grass.shape == (CFG.size, CFG.size)
    assert int(s.t) == 0
    grown = float((s.grass == 0).mean())
    assert 0.4 < grown < 0.6


def test_move_wraps_on_torus():
    pos = jnp.array([[0, 0], [0, 0], [47, 47], [47, 47], [5, 5]], jnp.int32)
    acts = jnp.array([4, 1, 3, 2, 0], jnp.int32)  # west, north, east, south, stay
    out = np.asarray(pp.move(pos, acts, 48))
    assert out.tolist() == [[47, 0], [0, 47], [0, 47], [47, 0], [5, 5]]


def test_make_state_places_agents():
    s = pp.make_state(CFG, prey=[(3, 4, 10.0)], pred=[(5, 6, 20.0)])
    assert np.asarray(s.prey.pos)[0].tolist() == [3, 4]
    assert bool(s.prey.alive[0]) and not bool(s.prey.alive[1])
    assert float(s.pred.energy[0]) == 20.0
    assert int((s.grass == 0).sum()) == CFG.size * CFG.size


def test_observation_sees_neighbour_with_offset():
    # Predator at (7, 5), prey at (5, 5): the prey is 2 cells west -> row 4 (dy=0), column 2 (dx=-2).
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(7, 5, 30.0)])
    prey_obs, pred_obs = pp.observe(CFG, s)
    assert prey_obs.shape == (CFG.max_prey, 9, 9, 4)
    assert pred_obs.shape == (CFG.max_pred, 9, 9, 4)
    w = np.asarray(pred_obs[0], dtype=np.float32)
    assert w[4, 2, 1] == 1.0  # prey channel
    assert w[:, :, 1].sum() == 1.0
    assert w[4, 4, 2] == 1.0  # itself, in the predator channel
    assert np.allclose(w[:, :, 3], 30.0 / CFG.pred_energy_cap, atol=1e-2)
    assert np.all(w[:, :, 0] == 1.0)  # all grass grown


def test_observation_wraps():
    s = pp.make_state(CFG, prey=[(47, 47, 10.0)], pred=[(0, 0, 30.0)])
    _, pred_obs = pp.observe(CFG, s)
    w = np.asarray(pred_obs[0], dtype=np.float32)
    assert w[3, 3, 1] == 1.0


def test_dead_slots_observe_zeros():
    s = pp.make_state(CFG, prey=[(1, 1, 10.0)], pred=[(2, 2, 30.0)])
    prey_obs, _ = pp.observe(CFG, s)
    assert float(jnp.abs(prey_obs[1].astype(jnp.float32)).sum()) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_world.py -q`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` on `simulations.predprey`.

- [ ] **Step 3: Write `simulations/predprey.py` (state, reset, move, observe)**

```python
"""Predator-prey Crucible: a wrapping grid world in pure JAX.

All shapes are static, so `step` can be jit-compiled and vmapped over
thousands of worlds. Positions are (x, y); 2-D maps are indexed [y, x].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

NUM_ACTIONS = 5
# stay, north, south, east, west as (dx, dy); north is y - 1.
DELTAS = jnp.array([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]], dtype=jnp.int32)
OBS_CHANNELS = 4  # grass grown, prey count, predator count, own energy / cap


@dataclass(frozen=True)
class WorldConfig:
    size: int = 48
    max_prey: int = 64
    max_pred: int = 32
    init_prey: int = 24
    init_pred: int = 8
    init_grass_frac: float = 0.5
    regrow_steps: int = 20
    prey_start_energy: float = 20.0
    pred_start_energy: float = 30.0
    grass_energy: float = 5.0
    catch_energy: float = 15.0
    prey_energy_cap: float = 40.0
    pred_energy_cap: float = 60.0
    prey_repro_energy: float = 30.0
    pred_repro_energy: float = 45.0
    view: int = 9
    max_steps: int = 500
    alive_reward: float = 0.01
    birth_reward: float = 1.0
    death_reward: float = -1.0


class Species(NamedTuple):
    pos: jax.Array  # (N, 2) int32, (x, y)
    energy: jax.Array  # (N,) float32
    alive: jax.Array  # (N,) bool


class State(NamedTuple):
    prey: Species
    pred: Species
    grass: jax.Array  # (size, size) int32: 0 = grown, k > 0 = regrows in k steps
    t: jax.Array  # () int32


def _spawn(key, n, n_alive, size, start_energy) -> Species:
    pos = jax.random.randint(key, (n, 2), 0, size, dtype=jnp.int32)
    alive = jnp.arange(n) < n_alive
    energy = jnp.where(alive, start_energy, 0.0).astype(jnp.float32)
    return Species(pos, energy, alive)


def reset(cfg: WorldConfig, key: jax.Array) -> State:
    k_prey, k_pred, k_grown, k_timer = jax.random.split(key, 4)
    grown = jax.random.uniform(k_grown, (cfg.size, cfg.size)) < cfg.init_grass_frac
    timers = jax.random.randint(
        k_timer, (cfg.size, cfg.size), 1, cfg.regrow_steps + 1, dtype=jnp.int32
    )
    return State(
        prey=_spawn(k_prey, cfg.max_prey, cfg.init_prey, cfg.size, cfg.prey_start_energy),
        pred=_spawn(k_pred, cfg.max_pred, cfg.init_pred, cfg.size, cfg.pred_start_energy),
        grass=jnp.where(grown, 0, timers).astype(jnp.int32),
        t=jnp.int32(0),
    )


def _species_from_list(agents, n) -> Species:
    pos = np.zeros((n, 2), np.int32)
    energy = np.zeros(n, np.float32)
    alive = np.zeros(n, bool)
    for i, (x, y, e) in enumerate(agents):
        pos[i] = (x, y)
        energy[i] = e
        alive[i] = True
    return Species(jnp.asarray(pos), jnp.asarray(energy), jnp.asarray(alive))


def make_state(cfg: WorldConfig, prey=(), pred=(), grass=None, t=0) -> State:
    """Build a hand-placed state for tests and debugging. Agents are (x, y, energy)."""
    if grass is None:
        grass = jnp.zeros((cfg.size, cfg.size), jnp.int32)
    return State(
        prey=_species_from_list(prey, cfg.max_prey),
        pred=_species_from_list(pred, cfg.max_pred),
        grass=jnp.asarray(grass, jnp.int32),
        t=jnp.int32(t),
    )


def move(pos: jax.Array, actions: jax.Array, size: int) -> jax.Array:
    return (pos + DELTAS[actions]) % size


def _count_map(sp: Species, size: int) -> jax.Array:
    return (
        jnp.zeros((size, size), jnp.float32)
        .at[sp.pos[:, 1], sp.pos[:, 0]]
        .add(sp.alive.astype(jnp.float32))
    )


def _windows(maps: jax.Array, pos: jax.Array, view: int, size: int) -> jax.Array:
    r = view // 2
    offs = jnp.arange(-r, r + 1)
    ys = (pos[:, 1, None] + offs) % size  # (N, view)
    xs = (pos[:, 0, None] + offs) % size
    return maps[ys[:, :, None], xs[:, None, :]]  # (N, view, view, C)


def observe(cfg: WorldConfig, state: State) -> tuple[jax.Array, jax.Array]:
    maps = jnp.stack(
        [
            (state.grass == 0).astype(jnp.float32),
            _count_map(state.prey, cfg.size),
            _count_map(state.pred, cfg.size),
        ],
        axis=-1,
    )

    def one(sp: Species, cap: float) -> jax.Array:
        w = _windows(maps, sp.pos, cfg.view, cfg.size)
        e = jnp.broadcast_to((sp.energy / cap)[:, None, None, None], w.shape[:-1] + (1,))
        obs = jnp.concatenate([w, e], axis=-1)
        return jnp.where(sp.alive[:, None, None, None], obs, 0.0).astype(jnp.bfloat16)

    return one(state.prey, cfg.prey_energy_cap), one(state.pred, cfg.pred_energy_cap)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_world.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add simulations tests && git commit -m "feat: predator-prey world state, reset, movement and observation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 3: World step: predation, grazing, death, reproduction, auto-reset

**Files:**
- Modify: `simulations/predprey.py` (append)
- Test: `tests/test_step.py`

**Interfaces:**
- Consumes: everything Task 2 produced.
- Produces:
  - `StepInfo` with int32 scalars `catches, prey_births, pred_births, prey_deaths, pred_deaths, prey_alive, pred_alive, prey_pred_dist_count, pred_pred_dist_count` and f32 scalars `prey_pred_dist_sum, pred_pred_dist_sum`
  - `StepResult(state, prey_reward (max_prey,) f32, pred_reward (max_pred,) f32, prey_done (max_prey,) bool, pred_done (max_pred,) bool, done () bool, info: StepInfo)`
  - `predation(prey, pred) -> (caught (P,) bool, eaten (Q,) bool)`
  - `graze(prey, grass, size) -> eats (Q,) bool`
  - `reproduce(sp, threshold) -> (Species, born (N,) bool)`
  - `nearest_dist(a, b, size, exclude_self=False) -> (sum f32, count int32)`
  - `step(cfg, state, prey_act, pred_act) -> StepResult` (deterministic, no key)
  - `step_autoreset(cfg, key, state, prey_act, pred_act) -> (StepResult, prey_obs, pred_obs)`. `StepResult.state` is the fresh world when `done`, while `info` still describes the world before the reset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_step.py
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from simulations import predprey as pp

CFG = pp.WorldConfig()


def stay(n):
    return jnp.zeros((n,), jnp.int32)


def bare():
    """No grown grass anywhere, so grazing never interferes."""
    return jnp.full((CFG.size, CFG.size), CFG.regrow_steps, jnp.int32)


def run(s, prey_act=None, pred_act=None):
    prey_act = stay(CFG.max_prey) if prey_act is None else prey_act
    pred_act = stay(CFG.max_pred) if pred_act is None else pred_act
    return pp.step(CFG, s, prey_act, pred_act)


def test_predator_eats_lowest_index_prey_once():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0), (5, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    r = run(s)
    assert not bool(r.state.prey.alive[0]) and bool(r.state.prey.alive[1])
    assert float(r.state.pred.energy[0]) == 30.0 + CFG.catch_energy - 1.0
    assert int(r.info.catches) == 1
    assert bool(r.prey_done[0]) and float(r.prey_reward[0]) == CFG.death_reward
    assert not bool(r.done)


def test_two_predators_one_prey_single_catch():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(5, 5, 30.0), (5, 5, 30.0)], grass=bare())
    r = run(s)
    assert int(r.info.catches) == 1
    assert float(r.state.pred.energy[0]) == 44.0
    assert float(r.state.pred.energy[1]) == 29.0


def test_moving_onto_prey_catches():
    s = pp.make_state(CFG, prey=[(6, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    pred_act = stay(CFG.max_pred).at[0].set(3)  # east
    r = run(s, pred_act=pred_act)
    assert int(r.info.catches) == 1


def test_grazing_lowest_index_eats_and_starts_regrowth():
    grass = bare().at[5, 5].set(0)
    s = pp.make_state(CFG, prey=[(5, 5, 10.0), (5, 5, 10.0)], pred=[(20, 20, 30.0)], grass=grass)
    r = run(s)
    e = np.asarray(r.state.prey.energy)
    assert e[0] == 10.0 + CFG.grass_energy - 1.0
    assert e[1] == 9.0
    assert int(r.state.grass[5, 5]) == CFG.regrow_steps - 1


def test_grass_timer_counts_down_to_grown():
    grass = bare().at[0, 0].set(1)
    s = pp.make_state(CFG, prey=[(30, 30, 10.0)], pred=[(20, 20, 30.0)], grass=grass)
    r = run(s)
    assert int(r.state.grass[0, 0]) == 0
    assert int(r.state.grass[1, 1]) == CFG.regrow_steps - 1


def test_energy_is_capped():
    grass = bare().at[5, 5].set(0)
    s = pp.make_state(CFG, prey=[(5, 5, 39.0)], pred=[(20, 20, 30.0)], grass=grass)
    r = run(s)
    # 39 + 5 - 1 = 43 is capped at 40, then the prey reproduces (>= 30) and halves to 20.
    assert float(r.state.prey.energy[0]) == CFG.prey_energy_cap / 2


def test_starvation_kills_with_penalty():
    s = pp.make_state(CFG, prey=[(1, 1, 10.0)], pred=[(20, 20, 1.0), (30, 30, 30.0)], grass=bare())
    r = run(s)
    assert not bool(r.state.pred.alive[0]) and bool(r.state.pred.alive[1])
    assert bool(r.pred_done[0]) and not bool(r.pred_done[1])
    assert float(r.pred_reward[0]) == CFG.death_reward
    assert abs(float(r.pred_reward[1]) - CFG.alive_reward) < 1e-7
    assert int(r.info.pred_deaths) == 1
    assert float(r.pred_reward[5]) == 0.0 and bool(r.pred_done[5])  # empty slot


def test_reproduce_splits_energy_into_first_free_slot():
    sp = pp.Species(
        pos=jnp.array([[3, 4], [0, 0], [7, 7]], jnp.int32),
        energy=jnp.array([30.0, 0.0, 10.0]),
        alive=jnp.array([True, False, True]),
    )
    out, born = pp.reproduce(sp, 30.0)
    assert np.asarray(born).tolist() == [True, False, False]
    assert np.asarray(out.alive).tolist() == [True, True, True]
    assert np.asarray(out.energy).tolist() == [15.0, 15.0, 10.0]
    assert np.asarray(out.pos)[1].tolist() == [3, 4]


def test_reproduce_without_free_slot_does_nothing():
    sp = pp.Species(
        pos=jnp.zeros((2, 2), jnp.int32),
        energy=jnp.array([40.0, 40.0]),
        alive=jnp.array([True, True]),
    )
    out, born = pp.reproduce(sp, 30.0)
    assert not np.asarray(born).any()
    assert np.asarray(out.energy).tolist() == [40.0, 40.0]


def test_slot_freed_by_death_and_refilled_still_ends_trajectory():
    # Prey 0 is eaten; prey 1 is fat and gives birth into slot 0 in the same step.
    s = pp.make_state(CFG, prey=[(5, 5, 10.0), (9, 9, 39.0)], pred=[(5, 5, 30.0)], grass=bare())
    r = run(s)
    assert bool(r.state.prey.alive[0])  # the newborn
    assert np.asarray(r.state.prey.pos)[0].tolist() == [9, 9]
    assert bool(r.prey_done[0]) and float(r.prey_reward[0]) == CFG.death_reward
    assert abs(float(r.prey_reward[1]) - (CFG.alive_reward + CFG.birth_reward)) < 1e-6
    assert int(r.info.prey_deaths) == 1 and int(r.info.prey_births) == 1


def test_extinction_ends_episode_and_cuts_all_trajectories():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    r = run(s)
    assert bool(r.done)
    assert bool(r.pred_done[0])


def test_time_limit_ends_episode():
    s = pp.make_state(CFG, prey=[(1, 1, 10.0)], pred=[(20, 20, 30.0)], grass=bare(), t=CFG.max_steps - 1)
    r = run(s)
    assert bool(r.done)


def test_nearest_dist_handles_empty_and_wraps():
    a = pp.Species(jnp.array([[0, 0]], jnp.int32), jnp.ones(1), jnp.array([True]))
    b = pp.Species(jnp.array([[47, 0], [10, 10]], jnp.int32), jnp.ones(2), jnp.array([True, False]))
    total, count = pp.nearest_dist(a, b, 48)
    assert float(total) == 1.0 and int(count) == 1
    nobody = b._replace(alive=jnp.array([False, False]))
    total, count = pp.nearest_dist(a, nobody, 48)
    assert float(total) == 0.0 and int(count) == 0


def test_nearest_dist_excludes_self():
    a = pp.Species(jnp.array([[0, 0], [3, 4]], jnp.int32), jnp.ones(2), jnp.array([True, True]))
    total, count = pp.nearest_dist(a, a, 48, exclude_self=True)
    assert abs(float(total) - 10.0) < 1e-5 and int(count) == 2


def test_autoreset_after_done_gives_fresh_world():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    r, prey_obs, pred_obs = pp.step_autoreset(
        CFG, jax.random.PRNGKey(1), s, stay(CFG.max_prey), stay(CFG.max_pred)
    )
    assert bool(r.done)
    assert int(r.state.t) == 0
    assert int(r.state.prey.alive.sum()) == CFG.init_prey
    assert int(r.info.prey_alive) == 0  # info describes the world before the reset
    assert prey_obs.shape == (CFG.max_prey, 9, 9, 4)


def test_vmapped_jitted_step_is_deterministic():
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    states = jax.vmap(partial(pp.reset, CFG))(keys)
    step_v = jax.jit(jax.vmap(partial(pp.step, CFG)))
    pa = jax.random.randint(jax.random.PRNGKey(1), (8, CFG.max_prey), 0, pp.NUM_ACTIONS)
    qa = jax.random.randint(jax.random.PRNGKey(2), (8, CFG.max_pred), 0, pp.NUM_ACTIONS)
    r1 = step_v(states, pa, qa)
    r2 = step_v(states, pa, qa)
    assert r1.state.prey.pos.shape == (8, CFG.max_prey, 2)
    for x, y in zip(jax.tree.leaves(r1), jax.tree.leaves(r2)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_step.py -q`
Expected: FAIL with `AttributeError: module 'simulations.predprey' has no attribute 'step'` (or `predation`, `reproduce`, ...).

- [ ] **Step 3: Append the step logic to `simulations/predprey.py`**

```python
class StepInfo(NamedTuple):
    catches: jax.Array
    prey_births: jax.Array
    pred_births: jax.Array
    prey_deaths: jax.Array
    pred_deaths: jax.Array
    prey_alive: jax.Array
    pred_alive: jax.Array
    prey_pred_dist_sum: jax.Array
    prey_pred_dist_count: jax.Array
    pred_pred_dist_sum: jax.Array
    pred_pred_dist_count: jax.Array


class StepResult(NamedTuple):
    state: State
    prey_reward: jax.Array  # (max_prey,)
    pred_reward: jax.Array  # (max_pred,)
    prey_done: jax.Array  # (max_prey,) bool: this slot's trajectory ends here
    pred_done: jax.Array  # (max_pred,) bool
    done: jax.Array  # () bool: the episode ended
    info: StepInfo


def _first_claim(claiming: jax.Array, key: jax.Array) -> jax.Array:
    """True where an index claims `key` and no lower index claims the same key."""
    n = claiming.shape[0]
    rivals = (key[:, None] == key[None, :]) & claiming[:, None] & claiming[None, :]
    earlier = jnp.tril(jnp.ones((n, n), bool), k=-1)  # earlier[j, k] = k < j
    return claiming & ~jnp.any(rivals & earlier, axis=1)


def predation(prey: Species, pred: Species) -> tuple[jax.Array, jax.Array]:
    """Each predator on a prey's cell picks the lowest-index prey there.
    If several predators pick the same prey, the lowest-index predator gets it.
    Returns (caught per predator, eaten per prey)."""
    same = (
        jnp.all(pred.pos[:, None, :] == prey.pos[None, :, :], axis=-1)
        & pred.alive[:, None]
        & prey.alive[None, :]
    )
    has = same.any(axis=1)
    choice = jnp.argmax(same, axis=1)
    caught = _first_claim(has, choice)
    eaten = jnp.zeros(prey.alive.shape[0], jnp.int32).at[choice].max(caught.astype(jnp.int32)) > 0
    return caught, eaten


def graze(prey: Species, grass: jax.Array, size: int) -> jax.Array:
    """Prey on grown grass eat it; if several share a cell, the lowest index eats."""
    on = prey.alive & (grass[prey.pos[:, 1], prey.pos[:, 0]] == 0)
    cell = prey.pos[:, 1] * size + prey.pos[:, 0]
    return _first_claim(on, cell)


def reproduce(sp: Species, threshold: float) -> tuple[Species, jax.Array]:
    """Each agent with energy >= threshold gives half its energy to a child on
    its own cell. Parents fill free slots in index order; parents beyond the
    number of free slots wait. Returns (species, gave_birth)."""
    n = sp.alive.shape[0]
    parents = sp.alive & (sp.energy >= threshold)
    free = ~sp.alive
    rank = jnp.cumsum(parents) - 1
    can = parents & (rank < free.sum())
    free_idx = jnp.nonzero(free, size=n, fill_value=0)[0]
    target = jnp.where(can, free_idx[jnp.clip(rank, 0, n - 1)], n)  # n = dropped
    half = sp.energy / 2
    energy = jnp.where(can, half, sp.energy).at[target].set(half, mode="drop")
    alive = sp.alive.at[target].set(True, mode="drop")
    pos = sp.pos.at[target].set(sp.pos, mode="drop")
    return Species(pos, energy, alive), can


def nearest_dist(a: Species, b: Species, size: int, exclude_self: bool = False):
    """Sum and count, over living agents in `a`, of the wrapped Euclidean distance
    to the nearest living agent in `b`. Agents with no such neighbour are not counted."""
    d = jnp.abs(a.pos[:, None, :] - b.pos[None, :, :])
    d = jnp.minimum(d, size - d).astype(jnp.float32)
    dist = jnp.sqrt((d**2).sum(-1))
    invalid = ~b.alive[None, :]
    if exclude_self:
        invalid = invalid | jnp.eye(a.alive.shape[0], b.alive.shape[0], dtype=bool)
    nearest = jnp.where(invalid, jnp.inf, dist).min(axis=1)
    valid = a.alive & jnp.isfinite(nearest)
    return jnp.where(valid, nearest, 0.0).sum(), valid.sum().astype(jnp.int32)


def _metabolise(sp: Species, cap: float) -> Species:
    energy = jnp.minimum(sp.energy - sp.alive.astype(jnp.float32), cap)
    return sp._replace(energy=jnp.where(sp.alive, energy, 0.0))


def step(cfg: WorldConfig, state: State, prey_act: jax.Array, pred_act: jax.Array) -> StepResult:
    prey, pred = state.prey, state.pred
    prey_was, pred_was = prey.alive, pred.alive

    # 1. Simultaneous movement.
    prey = prey._replace(pos=jnp.where(prey.alive[:, None], move(prey.pos, prey_act, cfg.size), prey.pos))
    pred = pred._replace(pos=jnp.where(pred.alive[:, None], move(pred.pos, pred_act, cfg.size), pred.pos))

    # 2. Predation.
    caught, eaten = predation(prey, pred)
    pred = pred._replace(energy=pred.energy + cfg.catch_energy * caught)
    prey = prey._replace(alive=prey.alive & ~eaten)

    # 3. Grazing.
    eats = graze(prey, state.grass, cfg.size)
    prey = prey._replace(energy=prey.energy + cfg.grass_energy * eats)
    grass = state.grass.at[prey.pos[:, 1], prey.pos[:, 0]].max(jnp.where(eats, cfg.regrow_steps, 0))

    # 4. Metabolism and energy caps.
    prey = _metabolise(prey, cfg.prey_energy_cap)
    pred = _metabolise(pred, cfg.pred_energy_cap)

    # 5. Death by starvation. Deaths are recorded before any slot is reused.
    prey = prey._replace(alive=prey.alive & (prey.energy > 0))
    pred = pred._replace(alive=pred.alive & (pred.energy > 0))
    prey_died = prey_was & ~prey.alive
    pred_died = pred_was & ~pred.alive

    # 6. Reproduction into free slots.
    prey, prey_born = reproduce(prey, cfg.prey_repro_energy)
    pred, pred_born = reproduce(pred, cfg.pred_repro_energy)

    # 7. Grass regrowth.
    grass = jnp.maximum(grass - 1, 0)

    t = state.t + 1
    new_state = State(prey, pred, grass, t)
    done = (t >= cfg.max_steps) | ~prey.alive.any() | ~pred.alive.any()

    def reward(was, died, born):
        r = jnp.where(died, cfg.death_reward, cfg.alive_reward) + cfg.birth_reward * born
        return jnp.where(was, r, 0.0).astype(jnp.float32)

    pp_sum, pp_count = nearest_dist(prey, pred, cfg.size)
    qq_sum, qq_count = nearest_dist(pred, pred, cfg.size, exclude_self=True)
    info = StepInfo(
        catches=caught.sum().astype(jnp.int32),
        prey_births=prey_born.sum().astype(jnp.int32),
        pred_births=pred_born.sum().astype(jnp.int32),
        prey_deaths=prey_died.sum().astype(jnp.int32),
        pred_deaths=pred_died.sum().astype(jnp.int32),
        prey_alive=prey.alive.sum().astype(jnp.int32),
        pred_alive=pred.alive.sum().astype(jnp.int32),
        prey_pred_dist_sum=pp_sum,
        prey_pred_dist_count=pp_count,
        pred_pred_dist_sum=qq_sum,
        pred_pred_dist_count=qq_count,
    )
    return StepResult(
        state=new_state,
        prey_reward=reward(prey_was, prey_died, prey_born),
        pred_reward=reward(pred_was, pred_died, pred_born),
        prey_done=prey_died | ~prey_was | done,
        pred_done=pred_died | ~pred_was | done,
        done=done,
        info=info,
    )


def step_autoreset(cfg: WorldConfig, key: jax.Array, state: State, prey_act: jax.Array, pred_act: jax.Array):
    """Step; if the episode ended, swap in a fresh world. `info` still describes
    the finished world. Returns (result, prey_obs, pred_obs) for the next state."""
    res = step(cfg, state, prey_act, pred_act)
    fresh = reset(cfg, key)
    next_state = jax.tree.map(lambda a, b: jnp.where(res.done, a, b), fresh, res.state)
    prey_obs, pred_obs = observe(cfg, next_state)
    return res._replace(state=next_state), prey_obs, pred_obs
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest -q`
Expected: `22 passed` (6 world + 16 step).

- [ ] **Step 5: Commit**

```bash
git add simulations tests && git commit -m "feat: world step with predation, grazing, death, reproduction and auto-reset" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 4: Network, policies and checkpoints

**Files:**
- Create: `agents/networks.py`, `agents/policies.py`, `agents/checkpoint.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `simulations.predprey.NUM_ACTIONS`, `OBS_CHANNELS`.
- Produces:
  - `ActorCritic()` (Flax linen): `net.apply(params, obs[..., view, view, 4]) -> (logits[..., 5] f32, value[...] f32)`
  - `init_params(net, key, view=9) -> params`
  - `random_policy(key, obs) -> actions[...] int32`; `network_policy(net, params) -> policy` with the same signature (samples from the logits)
  - `save_checkpoint(path, prey_params, pred_params, meta: dict) -> None` (atomic); `load_checkpoint(path, template) -> (prey_params, pred_params, meta)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agents.py
import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_checkpoint, save_checkpoint
from agents.networks import ActorCritic, init_params
from agents.policies import network_policy, random_policy


def test_actor_critic_shapes():
    net = ActorCritic()
    params = init_params(net, jax.random.PRNGKey(0), 9)
    obs = jnp.zeros((3, 7, 9, 9, 4), jnp.bfloat16)
    logits, value = net.apply(params, obs)
    assert logits.shape == (3, 7, 5) and value.shape == (3, 7)
    assert logits.dtype == jnp.float32


def test_initial_policy_is_near_uniform():
    net = ActorCritic()
    params = init_params(net, jax.random.PRNGKey(0), 9)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (16, 9, 9, 4))
    logits, _ = net.apply(params, obs)
    probs = np.asarray(jax.nn.softmax(logits))
    assert np.all(np.abs(probs - 0.2) < 0.05)


def test_random_policy_range_and_shape():
    a = random_policy(jax.random.PRNGKey(0), jnp.zeros((4, 6, 9, 9, 4)))
    assert a.shape == (4, 6)
    assert int(a.min()) >= 0 and int(a.max()) <= 4


def test_network_policy_samples_valid_actions():
    net = ActorCritic()
    policy = network_policy(net, init_params(net, jax.random.PRNGKey(0), 9))
    a = policy(jax.random.PRNGKey(2), jnp.zeros((4, 6, 9, 9, 4)))
    assert a.shape == (4, 6)
    assert int(a.min()) >= 0 and int(a.max()) <= 4


def test_checkpoint_roundtrip(tmp_path):
    net = ActorCritic()
    prey = init_params(net, jax.random.PRNGKey(0), 9)
    pred = init_params(net, jax.random.PRNGKey(1), 9)
    path = tmp_path / "ckpt" / "c.msgpack"
    save_checkpoint(path, prey, pred, {"update": 7, "env_steps": 1234})
    template = init_params(net, jax.random.PRNGKey(99), 9)
    prey2, pred2, meta = load_checkpoint(path, template)
    for a, b in zip(jax.tree.leaves(prey), jax.tree.leaves(prey2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(jax.tree.leaves(pred), jax.tree.leaves(pred2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert meta == {"update": 7, "env_steps": 1234}
    assert not list(path.parent.glob("*.tmp"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_agents.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.checkpoint'`.

- [ ] **Step 3: Write `agents/networks.py`**

```python
"""Policy/value network shared in shape by both species (each has its own params)."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from simulations.predprey import NUM_ACTIONS, OBS_CHANNELS


class ActorCritic(nn.Module):
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        lead = obs.shape[:-3]
        x = obs.reshape((-1,) + obs.shape[-3:]).astype(jnp.float32)
        x = nn.relu(nn.Conv(32, (3, 3), padding="VALID")(x))
        x = nn.relu(nn.Conv(32, (3, 3), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128)(x))
        logits = nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)[:, 0]
        return logits.reshape(lead + (self.num_actions,)), value.reshape(lead)


def init_params(net: ActorCritic, key: jax.Array, view: int = 9):
    return net.init(key, jnp.zeros((1, view, view, OBS_CHANNELS), jnp.float32))
```

- [ ] **Step 4: Write `agents/policies.py`**

```python
"""Policies map (key, obs[..., view, view, C]) to integer actions[...]."""
from __future__ import annotations

import jax

from simulations.predprey import NUM_ACTIONS


def random_policy(key: jax.Array, obs: jax.Array) -> jax.Array:
    return jax.random.randint(key, obs.shape[:-3], 0, NUM_ACTIONS)


def network_policy(net, params):
    def policy(key: jax.Array, obs: jax.Array) -> jax.Array:
        logits, _ = net.apply(params, obs)
        return jax.random.categorical(key, logits)

    return policy
```

- [ ] **Step 5: Write `agents/checkpoint.py`**

```python
"""Save and load both species' parameters in one msgpack file."""
from __future__ import annotations

from pathlib import Path

from flax import serialization


def save_checkpoint(path, prey_params, pred_params, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = serialization.msgpack_serialize(
        {
            "prey": serialization.to_state_dict(prey_params),
            "pred": serialization.to_state_dict(pred_params),
            "meta": dict(meta),
        }
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_checkpoint(path, template):
    raw = serialization.msgpack_restore(Path(path).read_bytes())
    prey = serialization.from_state_dict(template, raw["prey"])
    pred = serialization.from_state_dict(template, raw["pred"])
    return prey, pred, dict(raw["meta"])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_agents.py -q`
Expected: `5 passed`.

- [ ] **Step 7: Commit**

```bash
git add agents tests && git commit -m "feat: actor-critic network, policies and checkpoints" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 5: PPO for two species

**Files:**
- Create: `agents/ppo.py`
- Test: `tests/test_ppo.py`

**Interfaces:**
- Consumes: `simulations.predprey` (`WorldConfig`, `State`, `reset`, `observe`, `step_autoreset`, `StepInfo`), `agents.networks` (`ActorCritic`, `init_params`).
- Produces:
  - `TrainConfig` (frozen dataclass: `num_envs=1024, rollout_len=32, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, epochs=4, num_minibatches=16, seed=0`)
  - `tiny_configs() -> (WorldConfig, TrainConfig)` for tests and `--tiny`
  - `compute_gae(reward, value, done, last_value, gamma, lam) -> (adv, ret)` (time-major)
  - `masked_normalize(x, mask) -> x`
  - `ppo_loss(params, mb: dict, apply_fn, cfg) -> (loss, aux dict)`
  - `Learner(params, opt_state)`, `Runner(prey: Learner, pred: Learner, env_state, prey_obs, pred_obs, key)`
  - `make_trainer(wcfg, tcfg) -> Trainer(net, init: key -> Runner, update: Runner -> (Runner, metrics: dict[str, scalar]))`
  - Metric keys: `prey_alive, pred_alive, catches_per_step, prey_births_per_step, pred_births_per_step, prey_deaths_per_step, pred_deaths_per_step, prey_pred_dist, pred_cluster_dist, prey_reward, pred_reward`, plus `prey_pg_loss, prey_v_loss, prey_entropy, pred_pg_loss, pred_v_loss, pred_entropy`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ppo.py
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.networks import ActorCritic, init_params
from agents.ppo import (
    TrainConfig,
    compute_gae,
    make_trainer,
    masked_normalize,
    ppo_loss,
    tiny_configs,
)


def numpy_gae(r, v, d, last, gamma, lam):
    T = r.shape[0]
    adv = np.zeros_like(r)
    gae = np.zeros_like(last)
    next_v = last
    for t in reversed(range(T)):
        nd = 1.0 - d[t].astype(np.float64)
        delta = r[t] + gamma * next_v * nd - v[t]
        gae = delta + gamma * lam * nd * gae
        adv[t] = gae
        next_v = v[t]
    return adv


def test_gae_matches_numpy_reference():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(6, 3)).astype(np.float32)
    v = rng.normal(size=(6, 3)).astype(np.float32)
    d = np.zeros((6, 3), bool)
    d[2, 0] = True
    d[5, 1] = True
    last = rng.normal(size=3).astype(np.float32)
    adv, ret = compute_gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(d), jnp.asarray(last), 0.99, 0.95)
    ref = numpy_gae(r, v, d, last, 0.99, 0.95)
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), ref + v, rtol=1e-5, atol=1e-5)


def test_masked_normalize_ignores_masked_entries():
    x = jnp.array([1.0, 2.0, 3.0, 1000.0])
    mask = jnp.array([True, True, True, False])
    out = np.asarray(masked_normalize(x, mask))
    assert abs(out[:3].mean()) < 1e-5
    assert abs(out[:3].std() - 1.0) < 1e-3


def test_ppo_loss_policy_term_zero_when_unchanged_and_adv_zero():
    net = ActorCritic()
    params = init_params(net, jax.random.PRNGKey(0), 9)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (8, 9, 9, 4))
    logits, value = net.apply(params, obs)
    action = jnp.zeros((8,), jnp.int32)
    logp = jax.nn.log_softmax(logits)[:, 0]
    mb = {
        "obs": obs,
        "action": action,
        "logp": logp,
        "mask": jnp.ones((8,), bool),
        "adv": jnp.zeros((8,)),
        "ret": value,
    }
    _, aux = ppo_loss(params, mb, net.apply, TrainConfig())
    assert abs(float(aux["pg_loss"])) < 1e-6
    assert abs(float(aux["v_loss"])) < 1e-6
    assert float(aux["entropy"]) > 1.5  # near log(5) = 1.609 at init


def test_minibatch_divisibility_is_checked():
    wcfg, tcfg = tiny_configs()
    bad = TrainConfig(num_envs=3, rollout_len=1, num_minibatches=5)
    with pytest.raises(ValueError):
        make_trainer(wcfg, bad)


def test_trainer_smoke_tiny():
    wcfg, tcfg = tiny_configs()
    trainer = make_trainer(wcfg, tcfg)
    runner = trainer.init(jax.random.PRNGKey(0))
    before = jax.tree.leaves(runner.prey.params)[0].copy()
    for _ in range(3):
        runner, metrics = trainer.update(runner)
    metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
    expected = {
        "prey_alive", "pred_alive", "catches_per_step", "prey_births_per_step",
        "pred_births_per_step", "prey_deaths_per_step", "pred_deaths_per_step",
        "prey_pred_dist", "pred_cluster_dist", "prey_reward", "pred_reward",
        "prey_pg_loss", "prey_v_loss", "prey_entropy", "pred_pg_loss", "pred_v_loss", "pred_entropy",
    }
    assert expected <= set(metrics)
    assert all(math.isfinite(v) for v in metrics.values())
    after = jax.tree.leaves(runner.prey.params)[0]
    assert not np.array_equal(np.asarray(before), np.asarray(after))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_ppo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.ppo'`.

- [ ] **Step 3: Write `agents/ppo.py`**

```python
"""PPO for two species at once, in the style of PureJaxRL.

Each agent slot is an independent actor. A slot's trajectory ends when its
agent dies, when the slot was empty, or when the episode ends. Transitions
from empty slots are masked out of the loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from agents.networks import ActorCritic, init_params
from simulations import predprey as pp


@dataclass(frozen=True)
class TrainConfig:
    num_envs: int = 1024
    rollout_len: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    num_minibatches: int = 16
    seed: int = 0


def tiny_configs() -> tuple[pp.WorldConfig, TrainConfig]:
    """Small enough to train a few updates on a laptop CPU in seconds."""
    return (
        pp.WorldConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40),
        TrainConfig(num_envs=4, rollout_len=8, num_minibatches=2, epochs=1),
    )


class Learner(NamedTuple):
    params: Any
    opt_state: Any


class SpeciesBatch(NamedTuple):
    obs: jax.Array  # (T, E, N, view, view, C)
    action: jax.Array  # (T, E, N)
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array  # bool
    mask: jax.Array  # bool: the slot held a living agent when it acted


class Runner(NamedTuple):
    prey: Learner
    pred: Learner
    env_state: pp.State  # batched over envs
    prey_obs: jax.Array
    pred_obs: jax.Array
    key: jax.Array


class Trainer(NamedTuple):
    net: ActorCritic
    init: Callable[[jax.Array], Runner]
    update: Callable[[Runner], tuple[Runner, dict]]


def compute_gae(reward, value, done, last_value, gamma, lam):
    """Generalised advantage estimation over a time-major rollout.
    done[t] stops bootstrapping from step t + 1."""
    not_done = 1.0 - done.astype(jnp.float32)

    def body(carry, x):
        gae, next_value = carry
        r, v, nd = x
        delta = r + gamma * next_value * nd - v
        gae = delta + gamma * lam * nd * gae
        return (gae, v), gae

    _, adv = jax.lax.scan(
        body, (jnp.zeros_like(last_value), last_value), (reward, value, not_done), reverse=True
    )
    return adv, adv + value


def masked_normalize(x, mask):
    m = mask.astype(jnp.float32)
    n = jnp.maximum(m.sum(), 1.0)
    mean = (x * m).sum() / n
    var = (((x - mean) ** 2) * m).sum() / n
    return (x - mean) / jnp.sqrt(var + 1e-8)


def ppo_loss(params, mb: dict, apply_fn, cfg: TrainConfig):
    logits, value = apply_fn(params, mb["obs"])
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, mb["action"][:, None], axis=-1)[:, 0]
    m = mb["mask"].astype(jnp.float32)
    n = jnp.maximum(m.sum(), 1.0)
    ratio = jnp.exp(logp - mb["logp"])
    adv = mb["adv"]
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
    pg_loss = (pg * m).sum() / n
    v_loss = 0.5 * (((value - mb["ret"]) ** 2) * m).sum() / n
    entropy = ((-(jnp.exp(logp_all) * logp_all).sum(-1)) * m).sum() / n
    total = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
    return total, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy}


def rollout_metrics(info: pp.StepInfo, prey_b: SpeciesBatch, pred_b: SpeciesBatch) -> dict:
    steps = info.catches.size  # T * E world-steps

    def total(x):
        return x.sum().astype(jnp.float32)

    def ratio(num, den):
        return total(num) / jnp.maximum(total(den), 1.0)

    return {
        "prey_alive": total(info.prey_alive) / steps,
        "pred_alive": total(info.pred_alive) / steps,
        "catches_per_step": total(info.catches) / steps,
        "prey_births_per_step": total(info.prey_births) / steps,
        "pred_births_per_step": total(info.pred_births) / steps,
        "prey_deaths_per_step": total(info.prey_deaths) / steps,
        "pred_deaths_per_step": total(info.pred_deaths) / steps,
        "prey_pred_dist": ratio(info.prey_pred_dist_sum, info.prey_pred_dist_count),
        "pred_cluster_dist": ratio(info.pred_pred_dist_sum, info.pred_pred_dist_count),
        "prey_reward": ratio(prey_b.reward * prey_b.mask, prey_b.mask),
        "pred_reward": ratio(pred_b.reward * pred_b.mask, pred_b.mask),
    }


def make_trainer(wcfg: pp.WorldConfig, tcfg: TrainConfig) -> Trainer:
    E, T = tcfg.num_envs, tcfg.rollout_len
    for n in (wcfg.max_prey, wcfg.max_pred):
        if (T * E * n) % tcfg.num_minibatches:
            raise ValueError(
                f"rollout_len*num_envs*{n} = {T * E * n} is not divisible by "
                f"num_minibatches={tcfg.num_minibatches}"
            )

    net = ActorCritic()
    tx = optax.chain(optax.clip_by_global_norm(tcfg.max_grad_norm), optax.adam(tcfg.lr, eps=1e-5))
    reset_v = jax.vmap(partial(pp.reset, wcfg))
    observe_v = jax.vmap(partial(pp.observe, wcfg))
    step_v = jax.vmap(partial(pp.step_autoreset, wcfg))
    loss_fn = partial(ppo_loss, apply_fn=net.apply, cfg=tcfg)

    def init(key):
        k_env, k_prey, k_pred, key = jax.random.split(key, 4)
        env_state = reset_v(jax.random.split(k_env, E))
        prey_obs, pred_obs = observe_v(env_state)
        prey_params = init_params(net, k_prey, wcfg.view)
        pred_params = init_params(net, k_pred, wcfg.view)
        return Runner(
            Learner(prey_params, tx.init(prey_params)),
            Learner(pred_params, tx.init(pred_params)),
            env_state,
            prey_obs,
            pred_obs,
            key,
        )

    def act(params, key, obs):
        logits, value = net.apply(params, obs)
        action = jax.random.categorical(key, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]
        return action, logp, value

    def env_step(runner, _):
        key, k_prey, k_pred, k_env = jax.random.split(runner.key, 4)
        prey_a, prey_logp, prey_v = act(runner.prey.params, k_prey, runner.prey_obs)
        pred_a, pred_logp, pred_v = act(runner.pred.params, k_pred, runner.pred_obs)
        res, prey_obs, pred_obs = step_v(jax.random.split(k_env, E), runner.env_state, prey_a, pred_a)
        prey_b = SpeciesBatch(runner.prey_obs, prey_a, prey_logp, prey_v, res.prey_reward, res.prey_done,
                              runner.env_state.prey.alive)
        pred_b = SpeciesBatch(runner.pred_obs, pred_a, pred_logp, pred_v, res.pred_reward, res.pred_done,
                              runner.env_state.pred.alive)
        runner = runner._replace(env_state=res.state, prey_obs=prey_obs, pred_obs=pred_obs, key=key)
        return runner, (prey_b, pred_b, res.info)

    def learn(learner, batch: SpeciesBatch, last_value, key):
        adv, ret = compute_gae(batch.reward, batch.value, batch.done, last_value, tcfg.gamma, tcfg.gae_lambda)
        adv = masked_normalize(adv, batch.mask)
        flat = {"obs": batch.obs, "action": batch.action, "logp": batch.logp,
                "mask": batch.mask, "adv": adv, "ret": ret}
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), flat)
        total = flat["action"].shape[0]

        def epoch(learner, k):
            perm = jax.random.permutation(k, total)
            mbs = jax.tree.map(lambda x: x[perm].reshape((tcfg.num_minibatches, -1) + x.shape[1:]), flat)

            def minibatch(learner, mb):
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(learner.params, mb)
                updates, opt_state = tx.update(grads, learner.opt_state, learner.params)
                return Learner(optax.apply_updates(learner.params, updates), opt_state), aux

            return jax.lax.scan(minibatch, learner, mbs)

        learner, aux = jax.lax.scan(epoch, learner, jax.random.split(key, tcfg.epochs))
        return learner, jax.tree.map(jnp.mean, aux)

    def update(runner):
        runner, (prey_b, pred_b, info) = jax.lax.scan(env_step, runner, None, length=T)
        _, prey_last = net.apply(runner.prey.params, runner.prey_obs)
        _, pred_last = net.apply(runner.pred.params, runner.pred_obs)
        key, k_prey, k_pred = jax.random.split(runner.key, 3)
        prey, prey_aux = learn(runner.prey, prey_b, prey_last, k_prey)
        pred, pred_aux = learn(runner.pred, pred_b, pred_last, k_pred)
        metrics = rollout_metrics(info, prey_b, pred_b)
        metrics.update({f"prey_{k}": v for k, v in prey_aux.items()})
        metrics.update({f"pred_{k}": v for k, v in pred_aux.items()})
        return runner._replace(prey=prey, pred=pred, key=key), metrics

    return Trainer(net, jax.jit(init), jax.jit(update, donate_argnums=0))
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest -q`
Expected: `32 passed` (22 + 5 + 5). The smoke test compiles in under a minute on CPU.

- [ ] **Step 5: Commit**

```bash
git add agents tests && git commit -m "feat: two-species PPO trainer with masked agent slots" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 6: Clip rendering and evaluation

**Files:**
- Create: `viz/render.py`, `agents/evaluate.py`
- Test: `tests/test_render_eval.py`

**Interfaces:**
- Consumes: `simulations.predprey` (`WorldConfig`, `State`, `reset`, `step`, `observe`, `make_state`), `agents.policies.random_policy`.
- Produces:
  - `CELL = 10`, `HEADER = 32`, colour constants `SOIL, GRASS, PREY, PRED` (RGB tuples)
  - `render_frame(cfg, state, label="") -> np.ndarray (HEADER + size*CELL, size*CELL, 3) uint8`
  - `rollout_states(cfg, prey_policy, pred_policy, key, max_frames) -> list[State]` (initial state first; stops after the terminal state)
  - `write_clip(cfg, prey_policy, pred_policy, key, path, max_frames=None, fps=15, label="") -> int` (number of frames written)
  - `evaluate(cfg, prey_policy, pred_policy, key, num_episodes) -> {"catches_per_pred_step": float, "prey_lifetime": float, "mean_episode_len": float}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_render_eval.py
import math
from dataclasses import replace

import imageio.v2 as imageio
import jax
import numpy as np

from agents.evaluate import evaluate
from agents.policies import random_policy
from agents.ppo import tiny_configs
from simulations import predprey as pp
from viz.render import HEADER, PRED, PREY, render_frame, write_clip

CFG = pp.WorldConfig()


def test_render_frame_size_and_colours():
    s = pp.make_state(CFG, prey=[(0, 0, 10.0)], pred=[(1, 0, 30.0)])
    f = render_frame(CFG, s, label="test")
    assert f.shape == (HEADER + 480, 480, 3) and f.dtype == np.uint8
    assert tuple(int(c) for c in f[HEADER + 5, 5]) == PREY
    assert tuple(int(c) for c in f[HEADER + 5, 15]) == PRED


def test_write_clip_frame_count(tmp_path):
    path = tmp_path / "c.mp4"
    n = write_clip(CFG, random_policy, random_policy, jax.random.PRNGKey(0), path, max_frames=20)
    assert n == 20
    reader = imageio.get_reader(path)
    assert reader.count_frames() == 20
    assert reader.get_data(0).shape == (HEADER + 480, 480, 3)


def test_clip_stops_early_when_episode_ends(tmp_path):
    # One starving predator dies on the first step, so the episode ends at once.
    cfg = replace(CFG, init_pred=1, init_prey=1, pred_start_energy=1.0)
    n = write_clip(cfg, random_policy, random_policy, jax.random.PRNGKey(0), tmp_path / "e.mp4", max_frames=50)
    assert n == 2


def test_evaluate_is_finite_and_deterministic():
    wcfg, _ = tiny_configs()
    a = evaluate(wcfg, random_policy, random_policy, jax.random.PRNGKey(3), 8)
    b = evaluate(wcfg, random_policy, random_policy, jax.random.PRNGKey(3), 8)
    assert a == b
    assert set(a) == {"catches_per_pred_step", "prey_lifetime", "mean_episode_len"}
    assert all(math.isfinite(v) for v in a.values())
    assert a["prey_lifetime"] > 0
    assert 0 < a["mean_episode_len"] <= wcfg.max_steps
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_render_eval.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.evaluate'`.

- [ ] **Step 3: Write `viz/render.py`**

```python
"""Render world states to RGB frames and MP4 clips."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np
from PIL import Image, ImageDraw

from simulations import predprey as pp

CELL = 10
HEADER = 32
SOIL = (48, 36, 24)
GRASS = (46, 150, 60)
PREY = (235, 235, 235)
PRED = (220, 45, 45)
HEADER_BG = (16, 16, 16)
HEADER_TEXT = (230, 230, 230)


def render_frame(cfg: pp.WorldConfig, state: pp.State, label: str = "") -> np.ndarray:
    grass = np.asarray(state.grass)
    regrown = 1.0 - np.clip(grass / cfg.regrow_steps, 0.0, 1.0)
    frac = np.where(grass == 0, 1.0, 0.6 * regrown)[..., None]
    rgb = (np.array(SOIL) * (1 - frac) + np.array(GRASS) * frac).astype(np.uint8)
    img = np.repeat(np.repeat(rgb, CELL, axis=0), CELL, axis=1)

    def paint(sp: pp.Species, colour, inset: int) -> None:
        for x, y in np.asarray(sp.pos)[np.asarray(sp.alive)]:
            img[y * CELL + inset:(y + 1) * CELL - inset, x * CELL + inset:(x + 1) * CELL - inset] = colour

    paint(state.prey, PREY, 2)
    paint(state.pred, PRED, 1)

    side = cfg.size * CELL
    canvas = Image.new("RGB", (side, side + HEADER), HEADER_BG)
    canvas.paste(Image.fromarray(img), (0, HEADER))
    n_prey = int(np.asarray(state.prey.alive).sum())
    n_pred = int(np.asarray(state.pred.alive).sum())
    text = f"t={int(state.t):3d}   prey {n_prey:2d}   predators {n_pred:2d}   {label}"
    ImageDraw.Draw(canvas).text((8, 10), text, fill=HEADER_TEXT)
    return np.asarray(canvas)


def rollout_states(cfg, prey_policy, pred_policy, key, max_frames: int) -> list:
    @jax.jit
    def advance(state, k):
        kp, kq = jax.random.split(k)
        prey_obs, pred_obs = pp.observe(cfg, state)
        res = pp.step(cfg, state, prey_policy(kp, prey_obs), pred_policy(kq, pred_obs))
        return res.state, res.done

    k_reset, key = jax.random.split(key)
    state = pp.reset(cfg, k_reset)
    states = [jax.device_get(state)]
    for k in jax.random.split(key, max(max_frames - 1, 0)):
        state, done = advance(state, k)
        states.append(jax.device_get(state))
        if bool(done):
            break
    return states


def write_clip(cfg, prey_policy, pred_policy, key, path, max_frames=None, fps=15, label="") -> int:
    states = rollout_states(cfg, prey_policy, pred_policy, key, max_frames or cfg.max_steps)
    frames = [render_frame(cfg, s, label) for s in states]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1, quality=7)
    return len(frames)
```

- [ ] **Step 4: Write `agents/evaluate.py`**

```python
"""Fixed-seed evaluation of a (prey policy, predator policy) pairing."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from simulations import predprey as pp


def evaluate(cfg: pp.WorldConfig, prey_policy, pred_policy, key, num_episodes: int) -> dict:
    """Runs one episode in each of `num_episodes` worlds and returns:
    catches_per_pred_step: catches / living-predator-steps;
    prey_lifetime: living-prey-steps / prey that ever lived (initial + born);
    mean_episode_len: steps until each episode ended."""

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(pp.reset, cfg))(jax.random.split(k_reset, num_episodes))
        names = ("catches", "pred_alive_steps", "prey_alive_steps", "prey_births", "episode_steps")
        acc0 = {name: jnp.zeros((), jnp.float32) for name in names}

        def body(carry, k):
            state, active, acc = carry
            kp, kq = jax.random.split(k)
            prey_obs, pred_obs = jax.vmap(partial(pp.observe, cfg))(state)
            res = jax.vmap(partial(pp.step, cfg))(state, prey_policy(kp, prey_obs), pred_policy(kq, pred_obs))
            a = active.astype(jnp.float32)
            acc = {
                "catches": acc["catches"] + (res.info.catches * a).sum(),
                "pred_alive_steps": acc["pred_alive_steps"] + (state.pred.alive.sum(-1) * a).sum(),
                "prey_alive_steps": acc["prey_alive_steps"] + (state.prey.alive.sum(-1) * a).sum(),
                "prey_births": acc["prey_births"] + (res.info.prey_births * a).sum(),
                "episode_steps": acc["episode_steps"] + a.sum(),
            }
            return (res.state, active & ~res.done, acc), None

        init = (state, jnp.ones(num_episodes, bool), acc0)
        (_, _, acc), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    acc = {k: float(v) for k, v in jax.device_get(run(key)).items()}
    return {
        "catches_per_pred_step": acc["catches"] / max(acc["pred_alive_steps"], 1.0),
        "prey_lifetime": acc["prey_alive_steps"] / (cfg.init_prey * num_episodes + acc["prey_births"]),
        "mean_episode_len": acc["episode_steps"] / num_episodes,
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_render_eval.py -q`
Expected: `4 passed`. If `test_clip_stops_early_when_episode_ends` returns more than 2 frames, the random predator happened to land on the prey with seed 0. Change only the test's `PRNGKey(0)` to another seed, and note it in the commit.

- [ ] **Step 6: Commit**

```bash
git add viz agents tests && git commit -m "feat: MP4 clip rendering and fixed-seed evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 7: Training script, evaluation script and dashboard

**Files:**
- Create: `scripts/train.py`, `scripts/eval.py`, `viz/dashboard/index.html`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `agents.ppo` (`TrainConfig`, `make_trainer`, `tiny_configs`), `agents.checkpoint`, `agents.policies`, `agents.networks`, `agents.evaluate.evaluate`, `viz.render.write_clip`, `simulations.predprey.WorldConfig`.
- Produces:
  - `python -m scripts.train [--run-id ID] [--runs-dir runs] [--updates N] [--max-hours H] [--ckpt-every K] [--mem-limit-gb G] [--seed S] [--num-envs E] [--tiny]`. Exit codes: 0 ok, 2 bad or existing run id, 3 memory over limit.
  - Run directory `runs/<id>/`: `config.json`, `metrics.jsonl` (one JSON row per update with `update, env_steps, sps, wall_s` + every trainer metric), `clips.json` (list of `{file, update, env_steps, frames, label}`, the random baseline at `update` 0), `clips/*.mp4`, `checkpoints/ckpt_NNNNNN.msgpack` and `latest.msgpack`, `index.html`. Also `runs/index.html` listing runs.
  - `check_memory(stats: dict | None, limit_gb: float) -> (ok: bool, message: str)`
  - `python -m scripts.eval --run-dir runs/<id> [--checkpoint PATH] [--episodes 256]` writes `runs/<id>/eval.json` with `random_vs_random, random_prey_vs_trained_pred, trained_vs_trained, predator_ratio, prey_ratio, predator_pass, prey_pass`. Exit code 0 when both pass, 1 otherwise.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scripts.py
import json
import math

import scripts.eval as eval_script
import scripts.train as train


def test_check_memory_handles_missing_stats():
    ok, msg = train.check_memory(None, 12.0)
    assert ok and "unavailable" in msg
    ok, _ = train.check_memory({}, 12.0)
    assert ok


def test_check_memory_over_limit():
    ok, msg = train.check_memory({"peak_bytes_in_use": 13e9}, 12.0)
    assert not ok and "exceeds" in msg
    ok, _ = train.check_memory({"peak_bytes_in_use": 2e9}, 12.0)
    assert ok


def test_tiny_training_run_writes_outputs(tmp_path):
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "t1",
                     "--updates", "3", "--ckpt-every", "2"])
    assert rc == 0
    run = tmp_path / "t1"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == [1, 2, 3]
    assert all(math.isfinite(v) for r in rows for v in r.values())
    assert {"sps", "env_steps", "prey_alive", "catches_per_step"} <= set(rows[0])
    assert (run / "checkpoints" / "ckpt_000002.msgpack").exists()
    assert (run / "checkpoints" / "latest.msgpack").exists()
    clips = json.loads((run / "clips.json").read_text())
    assert [c["update"] for c in clips] == [0, 2, 3]
    assert all((run / c["file"]).exists() for c in clips)
    assert (run / "index.html").exists()
    assert 'href="t1/"' in (tmp_path / "index.html").read_text()
    config = json.loads((run / "config.json").read_text())
    assert config["world"]["size"] == 12


def test_existing_run_id_is_refused(tmp_path):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "keep.txt").write_text("old run")
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "t1", "--updates", "1"])
    assert rc == 2
    assert (tmp_path / "t1" / "keep.txt").read_text() == "old run"


def test_bad_run_id_is_refused(tmp_path):
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "../escape", "--updates", "1"])
    assert rc == 2
    assert not (tmp_path.parent / "escape").exists()


def test_eval_script_on_tiny_run(tmp_path):
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "t2",
                       "--updates", "2", "--ckpt-every", "2"]) == 0
    rc = eval_script.main(["--run-dir", str(tmp_path / "t2"), "--episodes", "4"])
    assert rc in (0, 1)
    result = json.loads((tmp_path / "t2" / "eval.json").read_text())
    assert {"random_vs_random", "random_prey_vs_trained_pred", "trained_vs_trained",
            "predator_ratio", "prey_ratio", "predator_pass", "prey_pass"} <= set(result)
    assert isinstance(result["predator_pass"], bool)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest tests/test_scripts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval'` (or `scripts.train`).

- [ ] **Step 3: Write `scripts/train.py`**

```python
"""One-command training: python -m scripts.train --run-id NAME"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax

from agents.checkpoint import save_checkpoint
from agents.policies import network_policy, random_policy
from agents.ppo import TrainConfig, make_trainer, tiny_configs
from simulations.predprey import WorldConfig
from viz.render import write_clip

DASHBOARD = Path(__file__).resolve().parents[1] / "viz" / "dashboard" / "index.html"
CLIP_SEED = 2026
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def check_memory(stats, limit_gb: float) -> tuple[bool, str]:
    """Judge jax Device.memory_stats() output against the memory budget."""
    if not stats or "peak_bytes_in_use" not in stats:
        return True, "device memory stats unavailable; relying on XLA_PYTHON_CLIENT_MEM_FRACTION cap"
    peak = stats["peak_bytes_in_use"] / 1e9
    if peak > limit_gb:
        return False, f"peak device memory {peak:.2f} GB exceeds limit {limit_gb} GB"
    return True, f"peak device memory {peak:.2f} GB (limit {limit_gb} GB)"


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def write_runs_index(runs_dir: Path) -> None:
    runs = sorted((p.name for p in runs_dir.iterdir() if (p / "index.html").exists()), reverse=True)
    items = "\n".join(f'<li><a href="{r}/">{r}</a></li>' for r in runs)
    (runs_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        "<title>autolearn runs</title>"
        "<body style='background:#111418;color:#e6e6e6;font:15px system-ui;padding:16px'>"
        f"<h1>autolearn runs</h1><ul>\n{items}\n</ul>\n"
    )


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--updates", type=int, default=100_000)
    p.add_argument("--max-hours", type=float, default=4.0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--mem-limit-gb", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--tiny", action="store_true", help="tiny world and batch, for tests")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not RUN_ID.fullmatch(args.run_id):
        print(f"invalid --run-id {args.run_id!r}: use letters, digits, '-' and '_'", file=sys.stderr)
        return 2
    if args.tiny:
        wcfg, tcfg = tiny_configs()
    else:
        wcfg, tcfg = WorldConfig(), TrainConfig()
    tcfg = replace(tcfg, seed=args.seed)
    if args.num_envs:
        tcfg = replace(tcfg, num_envs=args.num_envs)

    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_id
    if run_dir.exists():
        print(f"run directory {run_dir} already exists; pick another --run-id", file=sys.stderr)
        return 2
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "clips").mkdir()
    shutil.copy(DASHBOARD, run_dir / "index.html")
    write_json_atomic(run_dir / "config.json", {"world": asdict(wcfg), "train": asdict(tcfg), "args": vars(args)})
    write_runs_index(runs_dir)
    print(f"run {run_dir}  devices {jax.devices()}", flush=True)

    clips: list[dict] = []

    def add_clip(name, update, env_steps, prey_pi, pred_pi, label):
        frames = write_clip(wcfg, prey_pi, pred_pi, jax.random.PRNGKey(CLIP_SEED),
                            run_dir / "clips" / name, label=label)
        clips.append({"file": f"clips/{name}", "update": update, "env_steps": env_steps,
                      "frames": frames, "label": label})
        write_json_atomic(run_dir / "clips.json", clips)

    add_clip("random.mp4", 0, 0, random_policy, random_policy, "random policies")

    trainer = make_trainer(wcfg, tcfg)
    runner = trainer.init(jax.random.PRNGKey(tcfg.seed))
    steps_per_update = tcfg.num_envs * tcfg.rollout_len
    start = time.time()
    env_steps = 0
    with open(run_dir / "metrics.jsonl", "a") as log:
        for update in range(1, args.updates + 1):
            t0 = time.time()
            runner, metrics = trainer.update(runner)
            metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            dt = max(time.time() - t0, 1e-9)
            env_steps += steps_per_update
            row = {"update": update, "env_steps": env_steps, "sps": steps_per_update / dt,
                   "wall_s": time.time() - start, **metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()

            if update == 1:
                ok, msg = check_memory(jax.devices()[0].memory_stats(), args.mem_limit_gb)
                print(msg, flush=True)
                if not ok:
                    return 3

            last = update == args.updates or time.time() - start > args.max_hours * 3600
            if update % args.ckpt_every == 0 or last:
                ckpt = run_dir / "checkpoints" / f"ckpt_{update:06d}.msgpack"
                meta = {"update": update, "env_steps": env_steps}
                save_checkpoint(ckpt, runner.prey.params, runner.pred.params, meta)
                shutil.copy(ckpt, run_dir / "checkpoints" / "latest.msgpack")
                add_clip(f"clip_{update:06d}.mp4", update, env_steps,
                         network_policy(trainer.net, runner.prey.params),
                         network_policy(trainer.net, runner.pred.params), f"update {update}")
                print(f"update {update}  env_steps {env_steps:,}  sps {row['sps']:,.0f}  "
                      f"prey {metrics['prey_alive']:.1f}  pred {metrics['pred_alive']:.1f}  "
                      f"catches/step {metrics['catches_per_step']:.3f}", flush=True)
            if last:
                break
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/eval.py`**

```python
"""Trained-vs-random evaluation: python -m scripts.eval --run-dir runs/<id>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

from agents.checkpoint import load_checkpoint
from agents.evaluate import evaluate
from agents.networks import ActorCritic, init_params
from agents.policies import network_policy, random_policy
from simulations.predprey import WorldConfig

EVAL_SEED = 12345
PREDATOR_TARGET = 2.0  # trained predators catch >= 2x as often as random ones (vs random prey)
PREY_TARGET = 1.5  # trained prey live >= 1.5x as long as random prey (vs trained predators)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=256)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    wcfg = WorldConfig(**json.loads((run_dir / "config.json").read_text())["world"])
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "latest.msgpack"
    net = ActorCritic()
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view)
    prey_params, pred_params, meta = load_checkpoint(ckpt, template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    key = jax.random.PRNGKey(EVAL_SEED)

    rr = evaluate(wcfg, random_policy, random_policy, key, args.episodes)
    rt = evaluate(wcfg, random_policy, pred_pi, key, args.episodes)
    tt = evaluate(wcfg, prey_pi, pred_pi, key, args.episodes)
    predator_ratio = rt["catches_per_pred_step"] / max(rr["catches_per_pred_step"], 1e-9)
    prey_ratio = tt["prey_lifetime"] / max(rt["prey_lifetime"], 1e-9)
    result = {
        "checkpoint": str(ckpt),
        "meta": meta,
        "episodes": args.episodes,
        "random_vs_random": rr,
        "random_prey_vs_trained_pred": rt,
        "trained_vs_trained": tt,
        "predator_ratio": predator_ratio,
        "prey_ratio": prey_ratio,
        "predator_pass": bool(predator_ratio >= PREDATOR_TARGET),
        "prey_pass": bool(prey_ratio >= PREY_TARGET),
    }
    (run_dir / "eval.json").write_text(json.dumps(result, indent=1))
    print(f"predators: {predator_ratio:.2f}x random catch rate "
          f"(target {PREDATOR_TARGET}x) -> {'PASS' if result['predator_pass'] else 'FAIL'}")
    print(f"prey: {prey_ratio:.2f}x random lifetime vs trained predators "
          f"(target {PREY_TARGET}x) -> {'PASS' if result['prey_pass'] else 'FAIL'}")
    return 0 if result["predator_pass"] and result["prey_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `viz/dashboard/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autolearn run</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<style>
  :root { --bg:#111418; --panel:#1b2027; --text:#e6e6e6; --muted:#9aa4b2; --prey:#e8e8e8; --pred:#e04848; --accent:#5fb36b; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, sans-serif; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .muted { color: var(--muted); }
  .grid { display: grid; gap: 12px; margin-top: 12px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); }
  figure, .chart { margin: 0; background: var(--panel); border-radius: 8px; padding: 8px; min-width: 0; }
  video { width: 100%; display: block; border-radius: 4px; background: #000; }
  figcaption { margin-top: 6px; color: var(--muted); }
  .chart canvas { width: 100% !important; height: 220px !important; }
</style>
</head>
<body>
<h1 id="title">autolearn run</h1>
<div class="muted" id="status">loading…</div>
<div class="grid" id="clips"></div>
<div class="grid" id="charts"></div>
<script>
const CHARTS = [
  {title: "Population (mean alive per world)", keys: [["prey_alive", "prey", "--prey"], ["pred_alive", "predators", "--pred"]]},
  {title: "Events per world-step", keys: [["catches_per_step", "catches", "--pred"], ["prey_births_per_step", "prey births", "--prey"], ["pred_births_per_step", "predator births", "--accent"]]},
  {title: "Prey distance to nearest predator (cells)", keys: [["prey_pred_dist", "distance", "--prey"]]},
  {title: "Predator distance to nearest predator (cells)", keys: [["pred_cluster_dist", "distance", "--pred"]]},
  {title: "Reward per living agent-step", keys: [["prey_reward", "prey", "--prey"], ["pred_reward", "predators", "--pred"]]},
  {title: "Policy entropy", keys: [["prey_entropy", "prey", "--prey"], ["pred_entropy", "predators", "--pred"]]},
  {title: "Throughput (world-steps per second)", keys: [["sps", "steps/s", "--accent"]]},
];
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const charts = {};

function parseJsonl(text) {
  const rows = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); } catch (e) { /* a line still being written */ }
  }
  return rows;
}

function drawCharts(rows) {
  const box = document.getElementById("charts");
  for (const spec of CHARTS) {
    const datasets = spec.keys.map(([key, label, colour]) => ({
      label, data: rows.map(r => ({x: r.env_steps, y: r[key]})),
      borderColor: css(colour), backgroundColor: css(colour), pointRadius: 0, borderWidth: 1.5,
    }));
    if (charts[spec.title]) {
      charts[spec.title].data.datasets = datasets;
      charts[spec.title].update();
      continue;
    }
    const div = document.createElement("div");
    div.className = "chart";
    const title = document.createElement("div");
    title.className = "muted";
    title.textContent = spec.title;
    const canvas = document.createElement("canvas");
    div.append(title, canvas);
    box.appendChild(div);
    charts[spec.title] = new Chart(canvas, {
      type: "line",
      data: {datasets},
      options: {
        animation: false, parsing: false, maintainAspectRatio: false,
        scales: {
          x: {type: "linear", title: {display: true, text: "world-steps", color: css("--muted")}, ticks: {color: css("--muted")}},
          y: {ticks: {color: css("--muted")}},
        },
        plugins: {legend: {labels: {color: css("--text")}}},
      },
    });
  }
}

function drawClips(clips) {
  const box = document.getElementById("clips");
  const pick = clips.filter(c => c.update === 0).concat(clips.filter(c => c.update > 0).slice(-3).reverse());
  const key = pick.map(c => c.file).join("|");
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.replaceChildren();
  for (const c of pick) {
    const fig = document.createElement("figure");
    const v = document.createElement("video");
    Object.assign(v, {src: c.file, controls: true, muted: true, loop: true, playsInline: true, preload: "metadata"});
    const cap = document.createElement("figcaption");
    cap.textContent = `${c.label} · ${c.env_steps.toLocaleString()} world-steps · ${c.frames} frames`;
    fig.append(v, cap);
    box.appendChild(fig);
  }
}

async function refresh() {
  const status = document.getElementById("status");
  try {
    const [m, c] = await Promise.all([
      fetch("metrics.jsonl", {cache: "no-store"}),
      fetch("clips.json", {cache: "no-store"}),
    ]);
    const rows = m.ok ? parseJsonl(await m.text()) : [];
    const clips = c.ok ? await c.json() : [];
    drawClips(clips);
    drawCharts(rows);
    const last = rows[rows.length - 1];
    status.textContent = last
      ? `update ${last.update} · ${last.env_steps.toLocaleString()} world-steps · ${(last.wall_s / 3600).toFixed(2)} h · refreshed ${new Date().toLocaleTimeString()}`
      : "waiting for the first update…";
  } catch (e) {
    status.textContent = "could not load run data: " + e;
  }
}

document.getElementById("title").textContent =
  "autolearn · " + (location.pathname.split("/").filter(Boolean).pop() || "run");
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
```

- [ ] **Step 6: Run all tests to verify they pass**

Run: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run pytest -q`
Expected: `42 passed` (36 + 6).

- [ ] **Step 7: Check the dashboard renders from a tiny run**

Run: `rm -rf /tmp/al-dash && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn uv run python -m scripts.train --tiny --runs-dir /tmp/al-dash --run-id local --updates 4 --ckpt-every 2 && printf '{"update": 5, "env' >> /tmp/al-dash/local/metrics.jsonl && (cd /tmp/al-dash && timeout 120 python3 -m http.server 8799 >/dev/null 2>&1 &) && sleep 1 && curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8799/local/`
Expected: `200`. Open `http://127.0.0.1:8799/local/` with the Playwright browser tools and take a screenshot. Check that seven charts show four points each (the half-written fifth line is skipped), three clips are listed (random, update 4, update 2), and the console has no errors.

- [ ] **Step 8: Commit**

```bash
git add scripts viz tests && git commit -m "feat: training and evaluation scripts with live dashboard" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

---

### Task 8: Docs, first GPU runs and evaluation on a DGX Spark

**Files:**
- Create: `README.md`, `docs/README.md`, `scaling/README.md`

**Interfaces:**
- Consumes: `deploy/spark.sh` (Task 1), `scripts.train`, `scripts.eval` (Task 7).
- Produces: a running dashboard, a long training run, `eval.json`, and a short report to the user.

- [ ] **Step 1: Write `README.md`**

````markdown
# autolearn

A first "Crucible" in the sense of David Silver and Lasse Espeholt: a simple
predator–prey world where both species learn from scratch by reinforcement
learning, with no human data and no hand-written behaviour. Each species gets
one goal (stay alive and have offspring), and we watch what emerges.

Everything (world, networks, PPO) is pure JAX, compiled together onto the GPU of
a DGX Spark. Design: `docs/specs/2026-10-14-predprey-crucible-design.md`.

## Layout

| Path | What it is |
|---|---|
| `simulations/predprey.py` | the world: reset, step, observe |
| `agents/` | network, PPO, policies, checkpoints, evaluation |
| `viz/` | MP4 rendering and the dashboard page |
| `scripts/train.py`, `scripts/eval.py` | entry points |
| `deploy/` | Docker image and scripts for a DGX Spark |
| `docs/README.md` | reading list |

## Local (CPU) development

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/autolearn
uv sync
uv run pytest -q
uv run python -m scripts.train --tiny --runs-dir /tmp/al-runs --run-id try --updates 4 --ckpt-every 2
```

## On a DGX Spark

```bash
deploy/spark.sh sync                     # copy code to ~/autolearn on a DGX Spark
deploy/spark.sh build                    # build the JAX image (first time, or after Dockerfile changes)
deploy/spark.sh smoke                    # check JAX sees the GPU
deploy/spark.sh dash                     # dashboard on the Tailscale IP, port 9001
deploy/spark.sh train --run-id pp-001    # start training (4 h cap by default)
deploy/spark.sh logs                     # follow progress
deploy/spark.sh eval --run-dir runs/pp-001
deploy/spark.sh stop                     # stop autolearn containers only
```

Memory is capped at 12 GB (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.10`, plus a check
after the first update). Only containers named `autolearn-*` are ever touched.
````

- [ ] **Step 2: Write `docs/README.md`**

```markdown
# Reading list

Links are from memory (2026-10-14); check each one when you first open it.

## Why drop human data

- [Reward is Enough](https://doi.org/10.1016/j.artint.2021.103535): Silver, Singh, Precup, Sutton (2021). Intelligence as maximising one scalar reward in a rich environment.
- [Mastering the game of Go without human knowledge](https://doi.org/10.1038/nature24270): AlphaGo Zero (2017).
- [A general reinforcement learning algorithm that masters chess, shogi and Go through self-play](https://arxiv.org/abs/1712.01815): AlphaZero (2017/2018).
- [Mastering Atari, Go, chess and shogi by planning with a learned model](https://arxiv.org/abs/1911.08265): MuZero (2019/2020).

## Scaling the learner

- [IMPALA](https://arxiv.org/abs/1802.01561): Espeholt et al. (2018). Actor–learner split and V-trace.
- [SEED RL](https://arxiv.org/abs/1910.06591): Espeholt et al. (2019/2020). Central batched inference.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): environment and PPO compiled together; the style this repo follows.

## Emergence from multi-agent play

- [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528): Baker et al. (2019). Hide and seek.
- [Open-Ended Learning Leads to Generally Capable Agents](https://arxiv.org/abs/2107.12808): XLand (2021).

## Algorithms used here

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347): Schulman et al. (2017).
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438): Schulman et al. (2015).

## Toolkits for later

- [JaxMARL](https://github.com/FLAIROx/JaxMARL), [Gymnax](https://github.com/RobertTLange/gymnax), [Craftax](https://github.com/MichaelTMatthews/Craftax), [PufferLib](https://github.com/PufferAI/PufferLib), [CleanRL](https://github.com/vwxyzjn/cleanrl).
```

- [ ] **Step 3: Write `scaling/README.md`**

```markdown
# Scaling (sub-project 4)

Not built yet. Planned: steps-per-second against batch size, policy staleness
against sample efficiency (V-trace vs PPO clipping), and Elo or league play
across checkpoints. Until then, `sps` in each run's `metrics.jsonl` is the only
throughput number.
```

- [ ] **Step 4: Commit the docs**

```bash
git add README.md docs/README.md scaling && git commit -m "docs: README, reading list and scaling placeholder" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011EsFs2dc42eHSnvAdiNDfx"
```

- [ ] **Step 5: Short GPU run (about 5 minutes)**

Run: `deploy/spark.sh sync && deploy/spark.sh train --run-id smoke-1 --updates 30 --ckpt-every 10`
Then, after about 3 minutes: `deploy/spark.sh logs 30 && deploy/spark.sh status`
Expected:
- the logs show a GPU device, a `peak device memory ... (limit 12.0 GB)` or `unavailable` line, and `update 10/20/30` lines with SPS, ending in `done`;
- `autolearn-train` shows as `Exited (0)`.

If the memory guard fails (exit 3) or XLA reports out-of-memory: rerun with `--num-envs 512` under a new run id, and record the number used.
Record the seconds per update (`wall_s` / 30 from the last metrics row) to predict how many updates fit in 4 hours.

- [ ] **Step 6: Start the dashboard and check it**

Run: `deploy/spark.sh dash`
Expected: `dashboard: http://100.x.y.z:9001/`. Open `http://<that ip>:9001/smoke-1/` with the Playwright browser tools and screenshot it. The charts should show 30 points and the clips should play (random, update 30, 20, 10). Confirm Gemma still answers: `curl -s -o /dev/null -w '%{http_code}\n' http://<tailscale-ip>:8090/v1/models` → `200`.

- [ ] **Step 7: Start the long run**

Run: `deploy/spark.sh train --run-id pp-001 --max-hours 4`
(Add `--num-envs 512` if Step 5 needed it.) Expected: a container id. Check `deploy/spark.sh logs 10` once after 5 minutes. Then wait for the run to end with long fallback wakeups (≥ 20 minutes apart), not tight polling.

- [ ] **Step 8: Evaluate**

After `deploy/spark.sh status` shows `autolearn-train` exited:
Run: `deploy/spark.sh eval --run-dir runs/pp-001`
Expected: two lines, `predators: N.NNx ... PASS/FAIL` and `prey: N.NNx ... PASS/FAIL`, with `runs/pp-001/eval.json` written on a DGX Spark.

- [ ] **Step 9: Report**

Tell the user:
- the dashboard link;
- both eval ratios and whether each passed;
- updates and world-steps reached, and peak memory;
- a link to the latest clip, plus the random baseline for comparison.

Ask the user whether they can name a behaviour in the clips (success criterion 4). If either eval criterion failed, name the first `WorldConfig` knobs to turn (spec, "Risks and fallbacks") and do not change them without the user's go-ahead.
