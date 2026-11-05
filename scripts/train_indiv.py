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
from agents.indiv import CULTURE, Carry, IndivConfig, MODES, anchor, anchored_worlds, blank_brain, fresh_kin, from_pool, make_step, spread, tile
from agents.networks import init_params
from simulations.worlds import get_world, make_net, obs_channels, world_set


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


def load_pool(run_dir: Path, template):
    """Stage 19: every living grazer's brain in a run's pop_latest.npz, as one pytree with leading axis P."""
    leaves, tree = jax.tree.flatten(template)
    with np.load(Path(run_dir) / "checkpoints" / "pop_latest.npz") as d:
        return jax.tree.unflatten(tree, [jnp.asarray(d[f"prey_{i}"]) for i in range(len(leaves))])


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
    p.add_argument("--childhood", type=int, default=0, help="Stage 16: steps after birth with lessons")
    p.add_argument("--teach-cost", type=float, default=0.0, help="Stage 16: energy a parent pays per lesson")
    p.add_argument("--innate", action="store_true", help="Stage 16b: culture children are born with the parent's birth brain")
    p.add_argument("--subspace", type=int, default=0, help="Stage 17: mutate only along this many fixed random directions")
    p.add_argument("--anchor-frac", type=float, default=0.0, help="Stage 19: share of worlds whose grazers take brains from --anchor-pool")
    p.add_argument("--anchor-pool", default=None, help="Stage 19: a train_indiv run whose living grazers' brains are the frozen pool")
    p.add_argument("--world-set", action="append", default=[], metavar="KEY=VALUE", help="Stage 20: change a field of the world's config, repeatable (pred_base_cost=0.1, gene_sigma=0)")
    p.add_argument("--ckpt-every", type=int, default=15)
    p.add_argument("--tiny", action="store_true", help="a tiny world, for tests")
    args = p.parse_args(argv)

    world, cls = get_world(args.world)
    wcfg = cls()
    if args.tiny:
        wcfg = replace(wcfg, size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=20)
        args.num_envs = min(args.num_envs, 2)
    try:
        wcfg = world_set(wcfg, args.world_set)
    except ValueError as e:
        p.error(str(e))
    cfg = IndivConfig(mode=args.mode, lr=args.lr, sigma=args.sigma, gene_lr=args.gene_lr, steps=4 if args.tiny else 32, childhood=args.childhood, teach_cost=args.teach_cost, innate=args.innate, subspace=args.subspace, anchor_frac=args.anchor_frac)
    if (args.anchor_frac > 0) != bool(args.anchor_pool):
        p.error("--anchor-frac and --anchor-pool go together")
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
    pool, anch = None, None
    if cfg.anchor_frac > 0:
        pool = load_pool(Path(args.anchor_pool), template)
        anch = anchored_worlds(E, cfg.anchor_frac)
        mask = jnp.broadcast_to(anch[:, None], (E, wcfg.max_prey))
        carry = carry._replace(prey=anchor(carry.prey, mask, pool, jax.random.fold_in(k_prey, 19)))
        print(f"anchored: {int(anch.sum())} of {E} worlds, pool of {jax.tree.leaves(pool)[0].shape[0]} grazer brains from {args.anchor_pool}", flush=True)
    blank = None
    if cfg.mode in CULTURE:
        blank = blank_brain(net, wcfg.view, obs_channels(world, wcfg))
        carry = carry._replace(prey_kin=fresh_kin(E, wcfg.max_prey, cfg.childhood), pred_kin=fresh_kin(E, wcfg.max_pred, cfg.childhood))

    @jax.jit
    def update(c, pool):
        step = make_step(net, world, wcfg, cfg, blank=blank, anchor_pool=pool)
        return jax.lax.scan(step, c, None, length=cfg.steps)

    @jax.jit
    def pool_share(c, pool):
        """Share of living grazers whose brain is a pool brain, in anchored worlds and in the rest."""
        hit, alive = from_pool(c.prey.params, pool), c.state.prey.alive
        share = lambda w: (hit & alive & w[:, None]).sum() / jnp.maximum((alive & w[:, None]).sum(), 1)
        return share(anch), share(~anch)

    n_weights = sum(x.size for x in jax.tree.leaves(prey0))
    print(f"run {run_dir}  devices {jax.devices()}  {n_weights} weights a brain, mode {cfg.mode}", flush=True)
    t0, log, env_steps = time.time(), open(run_dir / "metrics.jsonl", "a"), 0
    for u in range(1, args.updates + 1):
        carry, info = update(carry, pool)
        env_steps += cfg.steps * E
        steps = info.catches.size
        row = {"update": u, "env_steps": env_steps, "wall_s": time.time() - t0, "sps": env_steps / (time.time() - t0),
               "prey_alive": float(info.prey_alive.sum() / steps), "pred_alive": float(info.pred_alive.sum() / steps),
               "catches_per_step": float(info.catches.sum() / steps), "pred_births_per_step": float(info.pred_births.sum() / steps),
               "pred_deaths_per_step": float(info.pred_deaths.sum() / steps)}
        row.update({k: float(v) for k, v in world.extra_metrics(info).items() if jnp.ndim(v) == 0})
        row.update({f"pred_{k}": float(v) for k, v in spread(carry.pred, carry.state.pred.alive).items()})
        row.update({f"prey_{k}": float(v) for k, v in spread(carry.prey, carry.state.prey.alive).items()})
        if anch is not None:
            check = u % args.ckpt_every == 0 or u == args.updates or u == 1
            shares = [float(x) for x in pool_share(carry, pool)] if check else None
            for i, (name, w) in enumerate((("anch", np.asarray(anch)), ("free", ~np.asarray(anch)))):
                if w.any():
                    row[f"{name}_pred_alive"] = float(np.asarray(info.pred_alive)[:, w].mean())
                    row[f"{name}_catches_per_step"] = float(np.asarray(info.catches)[:, w].mean())
                    if check:
                        row[f"{name}_prey_from_pool"] = shares[i]
        if cfg.mode in CULTURE:
            for name, kin in (("pred", carry.pred_kin), ("prey", carry.prey_kin)):
                row[f"{name}_child_steps"], row[f"{name}_lessons"], row[f"{name}_agree"] = float(kin.child_steps), float(kin.lessons), float(kin.agree)
        log.write(json.dumps(row) + "\n"); log.flush()
        if u % args.ckpt_every == 0 or u == args.updates:
            save_population(run_dir / "checkpoints" / "pop_latest.npz", carry, world)
            print(f"update {u} sps {row['sps']:.0f} prey {row['prey_alive']:.1f} pred {row['pred_alive']:.1f} catches/step {row['catches_per_step']:.3f} spread {row['pred_weight_spread']:.4f} moved {row['pred_weight_moved']:.4f}", flush=True)
    (run_dir / "done.json").write_text(json.dumps({"updates": args.updates}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
