"""Train individual brains (Stage 14a): python -m scripts.train_indiv --world tiny_stealth --mode lamarck --founders runs/NAME --run-id X

--founders is a run of the same world from scripts.train, whose brains every animal starts from. Writes metrics.jsonl and
checkpoints/pop_latest.npz (living animals' brains and genes). Spec: docs/specs/2026-11-01-individual-brains-design.md
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_brains
from agents.indiv import Carry, IndivConfig, MODES, make_step, spread, tile
from agents.networks import init_params
from simulations.worlds import get_world, make_net, obs_channels


def save_population(path: Path, carry: Carry, world) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, pop, sp in (("prey", carry.prey, carry.state.prey), ("pred", carry.pred, carry.state.pred)):
        alive = np.asarray(sp.alive).reshape(-1)
        for i, (leaf, birth) in enumerate(zip(jax.tree.leaves(pop.params), jax.tree.leaves(pop.birth))):
            leaf, birth = np.asarray(leaf), np.asarray(birth)
            out[f"{kind}_{i}"] = leaf.reshape((-1,) + leaf.shape[2:])[alive]
            out[f"{kind}_birth_{i}"] = birth.reshape((-1,) + birth.shape[2:])[alive]
    np.savez_compressed(path, **out)
    pool = jax.device_get(world.gene_pool(carry.state))
    np.savez_compressed(path.parent / "genes_latest.npz", **world.pool_to_arrays(pool, seed=0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", required=True)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--founders", default=None, help="a run from scripts.train; blank brains without one")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--founder-noise", type=float, default=0.01)
    p.add_argument("--gene-lr", action="store_true", help="each animal's learning rate follows its own brain gene (Stage 14b)")
    p.add_argument("--ckpt-every", type=int, default=15)
    p.add_argument("--tiny", action="store_true", help="a tiny world, for tests")
    args = p.parse_args(argv)

    world, cls = get_world(args.world)
    wcfg = cls()
    if args.tiny:
        wcfg = replace(wcfg, size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=20)
        args.num_envs = min(args.num_envs, 2)
    cfg = IndivConfig(mode=args.mode, lr=args.lr, sigma=args.sigma, gene_lr=args.gene_lr, steps=4 if args.tiny else 32)
    run_dir = Path(args.runs_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"world_name": args.world, "world": asdict(wcfg), "indiv": asdict(cfg), "train": {"memory": 0}, "args": vars(args)}, indent=1, default=str))

    net = make_net(world, wcfg, 0)
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    key = jax.random.PRNGKey(args.seed)
    key, k_prey, k_pred, k_env = jax.random.split(key, 4)
    if args.founders:
        prey0, pred0 = load_brains(Path(args.founders) / "checkpoints" / "latest.msgpack", template)[:2]
    else:
        prey0, pred0 = init_params(net, k_prey, wcfg.view, obs_channels(world, wcfg)), init_params(net, k_pred, wcfg.view, obs_channels(world, wcfg))
    E = args.num_envs
    state = jax.vmap(lambda k: world.reset(wcfg, k))(jax.random.split(k_env, E))
    prey_obs, pred_obs = jax.vmap(lambda s: world.observe(wcfg, s))(state)
    carry = Carry(state, tile(prey0, E, wcfg.max_prey, k_prey, args.founder_noise), tile(pred0, E, wcfg.max_pred, k_pred, args.founder_noise), prey_obs, pred_obs, key)
    step = make_step(net, world, wcfg, cfg)

    @jax.jit
    def update(c):
        return jax.lax.scan(step, c, None, length=cfg.steps)

    n_weights = sum(x.size for x in jax.tree.leaves(prey0))
    print(f"run {run_dir}  devices {jax.devices()}  {n_weights} weights a brain, mode {cfg.mode}", flush=True)
    t0, log, env_steps = time.time(), open(run_dir / "metrics.jsonl", "a"), 0
    for u in range(1, args.updates + 1):
        carry, info = update(carry)
        env_steps += cfg.steps * E
        steps = info.catches.size
        row = {"update": u, "env_steps": env_steps, "wall_s": time.time() - t0, "sps": env_steps / (time.time() - t0),
               "prey_alive": float(info.prey_alive.sum() / steps), "pred_alive": float(info.pred_alive.sum() / steps),
               "catches_per_step": float(info.catches.sum() / steps), "pred_births_per_step": float(info.pred_births.sum() / steps),
               "pred_deaths_per_step": float(info.pred_deaths.sum() / steps)}
        row.update({k: float(v) for k, v in world.extra_metrics(info).items() if jnp.ndim(v) == 0})
        row.update({f"pred_{k}": float(v) for k, v in spread(carry.pred, carry.state.pred.alive).items()})
        row.update({f"prey_{k}": float(v) for k, v in spread(carry.prey, carry.state.prey.alive).items()})
        log.write(json.dumps(row) + "\n"); log.flush()
        if u % args.ckpt_every == 0 or u == args.updates:
            save_population(run_dir / "checkpoints" / "pop_latest.npz", carry, world)
            print(f"update {u} sps {row['sps']:.0f} prey {row['prey_alive']:.1f} pred {row['pred_alive']:.1f} catches/step {row['catches_per_step']:.3f} spread {row['pred_weight_spread']:.4f} moved {row['pred_weight_moved']:.4f}", flush=True)
    (run_dir / "done.json").write_text(json.dumps({"updates": args.updates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
