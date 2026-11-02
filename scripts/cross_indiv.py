"""Hunters from one run against grazers from another: python -m scripts.cross_indiv --hunters runs/A --grazers runs/B

Each side brings its own genes and brains, and brains are frozen: no learning, no mutation. --hunters may be a shared-brain run from
scripts.train (--founder), whose one brain goes to every hunter. Written to <grazers>/cross_<hunters>.json.
Spec: docs/specs/2026-11-01-head-to-head-design.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from agents.checkpoint import load_brains
from agents.indiv import Carry, IndivConfig, Pop, make_step, tile
from agents.networks import init_params
from scripts.eval import EVAL_SEED, load_world_config
from scripts.eval_indiv import draw
from scripts.league import evolved_pool
from simulations.worlds import make_net, obs_channels


def frozen(snapshot, kind, template, e, n, key) -> Pop:
    pop = draw(snapshot, kind, template, e, n, key)
    return Pop(pop.params, pop.params)  # children copy the brain as it stands


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hunters", required=True)
    p.add_argument("--grazers", required=True)
    p.add_argument("--founder", action="store_true", help="--hunters is a shared-brain run from scripts.train")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args(argv)

    hunters, grazers = Path(args.hunters), Path(args.grazers)
    config = json.loads((grazers / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, 0)
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    pool_h, _ = evolved_pool(hunters)
    pool_g, _ = evolved_pool(grazers)
    pool = pool_g._replace(pred_genes=pool_h.pred_genes, pred_gen=pool_h.pred_gen, pred_w=pool_h.pred_w)
    snap_g = dict(np.load(grazers / "checkpoints" / "pop_latest.npz"))
    E = args.episodes
    key, k_reset, k_prey, k_pred = jax.random.split(jax.random.PRNGKey(EVAL_SEED), 4)
    if args.founder:
        one = load_brains(hunters / "checkpoints" / "latest.msgpack", template)[1]
        pred = tile(one, E, wcfg.max_pred, k_pred, 0.0)
    else:
        pred = frozen(dict(np.load(hunters / "checkpoints" / "pop_latest.npz")), "pred", template, E, wcfg.max_pred, k_pred)
    state = jax.vmap(lambda c: world.reset(wcfg, c, pool))(jax.random.split(k_reset, E))
    prey_obs, pred_obs = jax.vmap(lambda s: world.observe(wcfg, s))(state)
    carry = Carry(state, frozen(snap_g, "prey", template, E, wcfg.max_prey, k_prey), pred, prey_obs, pred_obs, key)
    step = make_step(net, world, wcfg, IndivConfig(mode="darwin", sigma=0.0), restart=False)

    @jax.jit
    def chunk(c):
        c, info = jax.lax.scan(step, c, None, length=100)
        return c, info.catches.sum(), info.pred_alive.sum()

    catches, hunter_steps = 0.0, 0.0
    for _ in range(max(args.steps // 100, 1)):
        carry, c, h = chunk(carry)
        catches, hunter_steps = catches + float(c), hunter_steps + float(h)
    out = {"hunters": hunters.name, "grazers": grazers.name, "catches_per_hunter_step": catches / max(hunter_steps, 1.0),
           "hunters_at_end": float(carry.state.pred.alive.sum(-1).mean()), "grazers_at_end": float(carry.state.prey.alive.sum(-1).mean())}
    (grazers / f"cross_{hunters.name}.json").write_text(json.dumps(out, indent=1))
    print(f"{hunters.name} against {grazers.name}: {out['catches_per_hunter_step']:.4f} catches per hunter-step; hunters {out['hunters_at_end']:.1f}, grazers {out['grazers_at_end']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
