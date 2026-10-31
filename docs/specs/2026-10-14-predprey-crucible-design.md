# Sub-project 1 — Predator–Prey Crucible you can watch

Date: 2026-10-14. Status: draft for review.

## Intent

Build the first "Crucible" in the Silver/Espeholt sense: simple rules and one
simple objective per species, with behaviour that is learned rather than
scripted. The point is **watchable emergence**: the user should be able to see,
in video clips and in measured curves, agents change from random wandering to
something recognisable (prey fleeing, predators chasing, perhaps pack hunting).

What the user decided (2026-10-14):

- Purpose: watchable emergence, not a teaching lab or a benchmark suite.
- World: predators and prey.
- Viewing: MP4 clips at checkpoints, plus a small web page that shows the latest
  clips next to the metric curves.
- Approach A: pure JAX, with world and learner compiled together on the GPU of
  a DGX Spark.

Assumptions (mine, not the user's):

- Training runs on a DGX Spark (GB10 GPU, CUDA 13 driver). WSL has no CUDA and is
  used for editing and CPU-only tests.
- The machine also runs other services, so this project must stay within a
  **12 GB** memory budget and must not stop or restart other containers.
- The web page is reachable on the tailnet only, never public.

Out of scope for this sub-project (later sub-projects): Elo and league play
against old checkpoints, V-trace/IMPALA, recurrent or memory agents, scaling
benchmarks, the full reading library.

## Success criteria

1. A training run is started with one command and runs unattended on a DGX Spark.
2. Clips and curves for the run appear on the dashboard while it trains.
3. After at most 4 hours of training, a fixed evaluation (below) shows learned
   behaviour beating the random baseline:
   - trained predators catch prey at least **2×** as often as random predators
     do, both against random prey;
   - trained prey survive at least **1.5×** as long as random prey do, both
     against trained predators.
4. At least one clip shows a behaviour the user can name without looking at the
   metrics.

Criterion 4 is a human judgement. Criteria 1–3 are checked by scripts.

## Repository layout

```
autolearn/
  README.md                 what this is, how to run it
  pyproject.toml            uv project; jax, flax, optax, numpy, imageio[ffmpeg], pytest
  docs/README.md            starter reading list (papers from the brief)
  docs/specs/   this spec
  simulations/predprey.py   the world: reset, step, observe (pure JAX)
  agents/networks.py        policy/value network (Flax)
  agents/ppo.py             rollout, GAE, PPO update for two species
  viz/render.py             state -> RGB frame; episode -> MP4
  viz/dashboard/index.html  static page: clips + Chart.js curves from metrics.jsonl
  scripts/train.py          the one-command training entry point
  scripts/eval.py           fixed trained-vs-random evaluation
  deploy/Dockerfile         FROM NVIDIA's JAX container for arm64/Blackwell
  deploy/spark.sh           sync code, build, start/stop/logs on a DGX Spark
  scaling/README.md         placeholder for sub-project 4
  tests/                    pytest, CPU-only
  runs/<run_id>/            (git-ignored) metrics.jsonl, checkpoints/, clips/
```

## The world (`simulations/predprey.py`)

All shapes are static so the step function can be `jit`- and `vmap`-compiled.

- **Grid**: 48×48, wrapping at the edges (a torus), so there are no walls.
- **Agent slots**: fixed arrays of `MAX_PREY = 64` and `MAX_PRED = 32` slots, each
  with `alive`, `x`, `y`, `energy`. Dead slots are reused for newborns.
- **Start**: 24 prey and 8 predators at random cells; grass is fully grown on a
  random 50% of cells.
- **Grass**: each cell is grown or regrowing. An eaten cell regrows after 20 steps.
- **Actions**: 5 (stay, north, south, east, west), one per living agent per step.
- **Step order** (simultaneous moves, then resolution):
  1. All living agents move.
  2. Predation: each predator on a cell holding at least one prey eats the
     lowest-index prey there (+15 energy). One catch per predator per step; each
     prey can be eaten only once.
  3. Grazing: each surviving prey on grown grass eats it (+5 energy). If several
     prey share the cell, the lowest index eats.
  4. Metabolism: every living agent loses 1 energy. Energy is capped at 40 for
     prey and 60 for predators.
  5. Death: agents at energy ≤ 0 die.
  6. Reproduction: a prey with energy ≥ 30 or a predator with energy ≥ 45 gives
     half its energy to a child placed on its own cell, if a free slot exists.
  7. Grass regrowth timers advance.
- **Starting energy**: prey 20, predators 30.
- **Episode end**: 500 steps, or earlier if either species is extinct.
- **Observation** (per agent): a 9×9 window centred on the agent with 4
  channels (grass, prey count, predator count, own energy / cap broadcast over
  the window), stored as `uint8`/`bfloat16` to save memory.
- **Reward** (per agent, per step): +0.01 for being alive, +1 for each child
  produced, −1 on death. Hunting and fleeing are never rewarded directly.
- **Info returned per step** (for metrics): catches, births, deaths and alive
  counts per species.

All numeric constants live in one frozen `WorldConfig` dataclass, so tuning
needs no code changes.

## The learner (`agents/`)

- **Network**: two separate parameter sets, one per species, with the same
  architecture: conv 3×3 ×32 → conv 3×3 ×32 → flatten → dense 128 → policy
  head (5 logits) and value head (1).
- **Algorithm**: PPO in the style of PureJaxRL. Rollout, GAE and update are one
  `jit`-compiled function, and the environment is `vmap`ped over `num_envs`.
- **Per-slot trajectories**: each agent slot is treated as an independent actor.
  Death sets `done`. A newborn in a reused slot starts a fresh trajectory.
  Samples from dead slots are masked out of the loss and the advantages.
- **Defaults** (`TrainConfig`): `num_envs = 512` (1024 ran out of memory on a DGX Spark, 2026-10-14), `rollout_len = 32`,
  `lr = 3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`, `clip = 0.2`,
  `ent_coef = 0.01`, 4 epochs × 16 minibatches (16, not 4, keeps each
  minibatch's activations to a few GB). `num_envs` can be set to 4096 if
  measured memory stays under budget.
- **Memory**: `XLA_PYTHON_CLIENT_PREALLOCATE=false` plus a memory-fraction cap.
  The training script logs peak device memory at the first update and aborts if
  it exceeds 12 GB.

## Watching (`viz/`)

- **Clip rendering**: after each checkpoint (every 50 updates by default), run one
  episode of a single world with the current policies and a fixed seed, and
  write an MP4: 10 px cells (a 480×480 grid under a 32 px header, so frames
  are 480 wide × 512 tall), 15 fps. Grass is shaded green by its
  state, prey are white, predators red, and step and population counts are
  drawn in a header. The same seed is used at every checkpoint so clips compare
  like with like. A clip of random policies is rendered once at the start as the
  baseline.
- **Metrics** appended to `runs/<id>/metrics.jsonl` every update:
  - update, environment steps, steps per second (SPS);
  - mean alive prey and predators;
  - catches, births and deaths per step;
  - mean reward per living agent-step, per species (per-episode returns are
    ill-defined when slots are reused by newborns);
  - mean distance from each prey to the nearest predator, on the torus;
  - mean distance from each predator to the nearest other predator (clustering);
  - PPO losses and entropy per species.
- **Dashboard**: a static `index.html` that fetches `metrics.jsonl` and a
  `clips.json` index. It shows Chart.js curves and the newest clips. Serving it
  is a `python -m http.server 9001` container serving `runs/` (one page per
  run at `/<run_id>/`, plus a generated index), bound to a DGX Spark's
  Tailscale IP only.

## Evaluation (`scripts/eval.py`)

Loads a checkpoint and runs 256 episodes with fixed seeds for each of three
pairings:

| # | Predators | Prey | Used for |
|---|---|---|---|
| 1 | random | random | predator baseline: catches per predator-step |
| 2 | trained | random | predator skill (compared with #1) and prey baseline lifetime |
| 3 | trained | trained | prey skill: lifetime compared with #2 |

Predator criterion: catches per predator-step in #2 ≥ 2 × #1.
Prey criterion: mean prey lifetime in #3 ≥ 1.5 × #2.
It writes `eval.json` and prints pass or fail for each criterion.

## Deployment (`deploy/`)

- Image: `FROM nvcr.io/nvidia/jax:<tag>`, using the newest tag with arm64 and
  GB10 support, plus the project's extra dependencies.
- `deploy/spark.sh` does `sync | build | train | dash | logs | stop`:
  - `sync` copies the repo to `~/autolearn` on a DGX Spark with tar over SSH
    (not from a Windows drive mounted in WSL to a remote builder, avoiding the known DrvFs upload
    problem);
  - `train` starts container `autolearn-train` with `--gpus all`, mounting
    `~/autolearn/runs`;
  - `dash` starts `autolearn-dash` on port 9001.
- It touches only containers whose names start with `autolearn-`.

## Testing

CPU-only pytest in WSL, using JAX on CPU:

- **World rules**: wrapping movement; grazing and regrowth timer; a predator eats
  exactly one prey; one catch per predator; death at zero energy; reproduction
  splits energy and respects free slots; the episode ends on extinction.
- **Shapes and determinism**: a `vmap`ped step over 8 worlds has static shapes,
  and the same seed gives the same trajectory.
- **PPO**: GAE against a hand-computed NumPy reference, including done flags and
  dead-slot masking; a 3-update smoke run on a tiny config finishes and the loss
  is finite.
- **Rendering**: a 20-step episode gives an MP4 with 20 frames at 480×512.
- **On a DGX Spark**: first a GPU smoke test (`jax.devices()` shows a CUDA device
  and a jitted step runs). Then a 5-minute training run that must produce
  metrics, a checkpoint and a clip before a long run is started.

## Risks and fallbacks

- **JAX on GB10 (aarch64, Blackwell)**: this is the first plan step. If no NVIDIA
  JAX container runs on this chip, fall back to approach B (the same world in
  PyTorch tensors in the `pytorch:25.11` container already on a DGX Spark), and
  tell the user before switching.
- **No emergence within 4 hours**: the first knobs to turn are grass regrowth,
  energy gains and predator starting count, all in `WorldConfig`. Ecosystem
  collapse (one species dies out every episode) shows up in the population
  curves.
- **Memory contention with Gemma**: the 12 GB guard aborts the run instead of
  starving the other service.
