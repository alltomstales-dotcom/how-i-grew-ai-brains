"""Stage 7d test: is the map the memory? python -m scripts.scatter --run-dir runs/NAME [--evolved]

Plays a fickle world, and once 40 steps have passed since the clock last restarted, moves every living grazer to a random cell, once.
Compares the good-bite share over the next 10 steps with the same seeds unscattered. Writes scatter.json.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_brains
from agents.networks import init_params
from agents.policies import forget_after, network_policy, pair_act, pair_memory
from scripts.eval import EVAL_SEED, load_world_config
from simulations.predprey import _windows
from simulations.worlds import make_net, obs_channels

AT, WINDOW = 40, 10
DROP_TARGET = 0.15


def measure(world, cfg, prey_pi, pred_pi, key, episodes: int, pool, scatter: bool) -> dict:
    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def near(st):
            kind = (st.plant_genes[..., 1] >= 0.5).astype(jnp.int32) == st.good
            edible = st.plant & ~st.bitten
            w = _windows(jnp.stack([edible & kind, edible & ~kind], -1).astype(jnp.float32), st.prey.pos, 5, cfg.size)
            return (w.sum((1, 2)) * st.prey.alive[:, None]).sum(0)

        def body(carry, k):
            state, acc, mem = carry
            k_act, k_prey, k_step, k_scatter = jax.random.split(k, 4)
            due = state.since_flip == AT
            if scatter:
                new = jax.random.randint(k_scatter, state.prey.pos.shape, 0, cfg.size)
                pos = jnp.where(due[:, None, None], new, state.prey.pos)
                state = state._replace(prey=state.prey._replace(pos=pos))
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            inside = (state.since_flip >= AT) & (state.since_flip < AT + WINDOW)
            counts = jax.vmap(near)(state)
            acc = acc.at[0].add((res.info.good_bites * inside).sum().astype(jnp.float32))
            acc = acc.at[1].add((res.info.bad_bites * inside).sum().astype(jnp.float32))
            acc = acc.at[2].add((counts[:, 0] * inside).sum())
            acc = acc.at[3].add((counts[:, 1] * inside).sum())
            return (res.state, acc, forget_after(mem, res)), None

        init = (state, jnp.zeros(4, jnp.float32), pair_memory(prey_pi, pred_pi, state))
        (_, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    g, b, ng, nb = (float(v) for v in np.asarray(run(key)))
    return {"good_share": g / max(g + b, 1.0), "bites": g + b, "near_share": ng / max(ng + nb, 1.0)}


def by_world(world, cfg, prey_pi, pred_pi, key, episodes: int, pool) -> dict:
    """Whole worlds as the unit: by steps since the clock restarted, how many worlds are still going, the share of edible plants that
    are the good kind counted once per world and weighted by grazers, and the good-bite share within worlds rich and poor in the good kind."""
    edges = jnp.asarray((0, 10, 40, 80))

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def body(carry, k):
            state, active, acc, mem = carry
            k_act, k_step = jax.random.split(k)
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            kind = (state.plant_genes[..., 1] >= 0.5).astype(jnp.int32) == state.good[:, None, None]
            edible = state.plant & ~state.bitten
            ng, nb = (edible & kind).sum((1, 2)).astype(jnp.float32), (edible & ~kind).sum((1, 2)).astype(jnp.float32)
            share = ng / jnp.maximum(ng + nb, 1.0)
            tough = ((state.plant_genes[..., 1] >= 0.5) & state.plant).sum((1, 2)) / jnp.maximum(state.plant.sum((1, 2)), 1)
            b = jnp.searchsorted(edges, state.since_flip, side="right") - 1
            n = state.prey.alive.sum(1).astype(jnp.float32)
            a = active.astype(jnp.float32)
            acc = acc.at[0, b].add(a)                  # world-steps still going
            acc = acc.at[1, b].add(a * share)          # good share, each world counted once
            acc = acc.at[2, b].add(a * n * share)      # good share, weighted by grazers
            acc = acc.at[3, b].add(a * n)
            acc = acc.at[4, b].add(a * tough)
            cls = jnp.where(share < 0.35, 0, jnp.where(share < 0.65, 1, 2))
            acc = acc.at[5 + cls * 2, b].add(a * res.info.good_bites)
            acc = acc.at[6 + cls * 2, b].add(a * res.info.bad_bites)
            return (res.state, active & ~res.done, acc, forget_after(mem, res)), None

        init = (state, jnp.ones(episodes, bool), jnp.zeros((11, 4), jnp.float32), pair_memory(prey_pi, pred_pi, state))
        (_, _, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    A = np.asarray(run(key)).astype(np.float64)
    names = ("0-10", "10-40", "40-80", "80+")
    safe = lambda x, y: x / np.maximum(y, 1.0)
    return {"bins": names, "world_steps": A[0].tolist(), "good_share_per_world": safe(A[1], A[0]).tolist(),
            "good_share_by_grazers": safe(A[2], A[3]).tolist(), "tough_share": safe(A[4], A[0]).tolist(),
            "bites_poor": safe(A[5], A[5] + A[6]).tolist(), "n_poor": (A[5] + A[6]).tolist(),
            "bites_mid": safe(A[7], A[7] + A[8]).tolist(), "n_mid": (A[7] + A[8]).tolist(),
            "bites_rich": safe(A[9], A[9] + A[10]).tolist(), "n_rich": (A[9] + A[10]).tolist()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true")
    p.add_argument("--worlds", action="store_true", help="whole worlds as the unit (scatter_worlds.json)")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, _ = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    pool = None
    if args.evolved:
        from scripts.league import evolved_pool
        pool, _ = evolved_pool(run_dir)
    key = jax.random.PRNGKey(EVAL_SEED)
    if args.worlds:
        out = by_world(world, wcfg, prey_pi, pred_pi, key, args.episodes, pool)
        (run_dir / "scatter_worlds.json").write_text(json.dumps(out, indent=1))
        print(run_dir.name)
        for k, v in out.items():
            print(f"  {k:22s}", v if k == "bins" else [round(x, 3) if x < 10 else int(x) for x in v])
        return 0
    same = measure(world, wcfg, prey_pi, pred_pi, key, args.episodes, pool, False)
    moved = measure(world, wcfg, prey_pi, pred_pi, key, args.episodes, pool, True)
    out = {"memory": config["train"].get("memory", 0), "unscattered": same, "scattered": moved, "drop": same["good_share"] - moved["good_share"]}
    (run_dir / "scatter.json").write_text(json.dumps(out, indent=1))
    print(f"{run_dir.name}: good bites {same['good_share']:.3f} -> {moved['good_share']:.3f} (drop {out['drop']:+.3f}); "
          f"good kind near grazers {same['near_share']:.3f} -> {moved['near_share']:.3f}; bites {int(same['bites'])} / {int(moved['bites'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
