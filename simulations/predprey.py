"""Predator-prey Crucible: a wrapping grid world in pure JAX.

All shapes are static, so `step` can be jit-compiled and vmapped over
thousands of worlds. Positions are (x, y); 2-D maps are indexed [y, x].
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

NUM_ACTIONS = 5
# stay, north, south, east, west as (dx, dy); north is y - 1.
DELTAS = jnp.array([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]], dtype=jnp.int32)
OBS_CHANNELS = 4  # grass grown, prey count, predator count, own energy / cap
PREY_CHANNEL = 1  # where prey show up in an observation


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


# --- World interface shared with simulations/evo.py (see simulations/worlds.py) ---

def advance(cfg: WorldConfig, key: jax.Array, state: State, prey_act: jax.Array, pred_act: jax.Array) -> StepResult:
    """`step` with the key-taking signature of the world interface; this world is deterministic."""
    return step(cfg, state, prey_act, pred_act)


def batch_step(cfg: WorldConfig, key: jax.Array, states: State, prey_act: jax.Array, pred_act: jax.Array):
    """Step a batch of worlds with per-world autoreset."""
    keys = jax.random.split(key, prey_act.shape[0])
    return jax.vmap(partial(step_autoreset, cfg))(keys, states, prey_act, pred_act)


def extra_metrics(info: StepInfo) -> dict:
    return {}
