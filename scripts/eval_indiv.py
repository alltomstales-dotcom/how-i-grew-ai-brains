"""Evaluate individual brains: python -m scripts.eval_indiv --run-dir runs/NAME [--episodes 64] [--steps 2000]

Worlds start from the run's gene snapshot, each animal drawing a brain from the population saved at the end of training, and run
with no restarts under the run's own rules. Reports both species alive at the last step, catches per hunter-step, and the same
catches with a fixed chaser hunting these grazers. Written to <run>/eval_indiv.json.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.indiv import Carry, IndivConfig, Pop, make_step
from agents.networks import init_params
from agents.policies import chase_policy
from scripts.eval import EVAL_SEED, load_world_config
from scripts.league import evolved_pool
from simulations.worlds import make_net, obs_channels


def draw(snapshot, kind: str, template, e: int, n: int, key) -> Pop:
    leaves, tree = jax.tree.flatten(template)
    count = snapshot[f"{kind}_0"].shape[0]
    index = jax.random.randint(key, (e, n), 0, max(count, 1))
    now = [jnp.asarray(snapshot[f"{kind}_{i}"])[index] for i in range(len(leaves))]
    was = [jnp.asarray(snapshot[f"{kind}_birth_{i}"])[index] for i in range(len(leaves))]
    return Pop(jax.tree.unflatten(tree, now), jax.tree.unflatten(tree, was))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, 0)
    cfg = IndivConfig(**config["indiv"])
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    pool, _ = evolved_pool(run_dir)
    snapshot = dict(np.load(run_dir / "checkpoints" / "pop_latest.npz"))
    E = args.episodes

    def run(fixed_pred, key):
        k_reset, k_prey, k_pred, k = jax.random.split(key, 4)
        state = jax.vmap(lambda c: world.reset(wcfg, c, pool))(jax.random.split(k_reset, E))
        prey_obs, pred_obs = jax.vmap(lambda s: world.observe(wcfg, s))(state)
        carry = Carry(state, draw(snapshot, "prey", template, E, wcfg.max_prey, k_prey), draw(snapshot, "pred", template, E, wcfg.max_pred, k_pred), prey_obs, pred_obs, k)
        step = make_step(net, world, wcfg, cfg, restart=False, fixed_pred=fixed_pred)

        @jax.jit
        def chunk(c):
            c, info = jax.lax.scan(step, c, None, length=100)
            return c, info.catches.sum(), info.pred_alive.sum()

        catches, hunter_steps = 0.0, 0.0
        for _ in range(max(args.steps // 100, 1)):
            carry, c, h = chunk(carry)
            catches, hunter_steps = catches + float(c), hunter_steps + float(h)
        both = float(((carry.state.pred.alive.sum(-1) > 0) & (carry.state.prey.alive.sum(-1) > 0)).mean())
        return {"both_at_end": both, "catches_per_hunter_step": catches / max(hunter_steps, 1.0),
                "hunters_at_end": float(carry.state.pred.alive.sum(-1).mean()), "grazers_at_end": float(carry.state.prey.alive.sum(-1).mean())}

    key = jax.random.PRNGKey(EVAL_SEED)
    out = {"mode": cfg.mode, "learned": run(None, key), "chaser": run(chase_policy(world.PREY_CHANNEL), key)}
    (run_dir / "eval_indiv.json").write_text(json.dumps(out, indent=1))
    l, c = out["learned"], out["chaser"]
    print(f"{run_dir.name} {cfg.mode}: both alive {l['both_at_end']:.3f}; hunters {l['hunters_at_end']:.1f}, grazers {l['grazers_at_end']:.1f}; catches per hunter-step {l['catches_per_hunter_step']:.4f}, chaser {c['catches_per_hunter_step']:.4f} ({c['catches_per_hunter_step'] / max(l['catches_per_hunter_step'], 1e-9):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
