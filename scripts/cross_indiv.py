"""Hunters from one run against grazers from another: python -m scripts.cross_indiv --hunters runs/A --grazers runs/B

Each side brings its own genes and brains, and brains are frozen: no learning, no mutation. Hunters of a culture arm (Stage 16) keep
their culture: children are born blank and take lessons. --both-live does the same for grazers from a culture run. --hunters may be a shared-brain run from
scripts.train (--founder), whose one brain goes to every hunter. Written to <grazers>/cross_<hunters>.json.
Stage 20: --world-set changes the test world (which is the grazers' run's), --bodies gives hunters another run's body genes,
--eval-seed reruns the test on other worlds, and the result adds total catches and what hunters pay per step for their bodies.
Stage 20b: --hunter-ckpt tests a population kept part way through training (train_indiv --keep-at).
Spec: docs/specs/2026-11-01-head-to-head-design.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from agents.checkpoint import load_brains
from agents.indiv import CULTURE, Carry, IndivConfig, Pop, blank_brain, fresh_kin, make_step, tile
from agents.networks import init_params
from scripts.eval import EVAL_SEED, load_world_config
from scripts.eval_indiv import draw
from scripts.league import evolved_pool
from simulations.worlds import make_net, obs_channels, world_set


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
    p.add_argument("--both-live", action="store_true", help="Stage 16: grazers keep their culture too")
    p.add_argument("--random", action="store_true", help="hunters act at random: the floor")
    p.add_argument("--frozen", action="store_true", help="Stage 16b check: culture hunters with lessons switched off, children copy the brain")
    p.add_argument("--births", action="store_true", help="Stage 16b check: hunters use the brain they were born with, before any lesson; implies --frozen")
    p.add_argument("--out", default=None, help="file name in place of cross_<hunters>.json")
    p.add_argument("--world-set", action="append", default=[], metavar="KEY=VALUE", help="Stage 20: change a field of the test world's config, repeatable")
    p.add_argument("--bodies", default=None, help="Stage 20: hunters take their body genes from this run instead of --hunters (brains still from --hunters)")
    p.add_argument("--eval-seed", type=int, default=EVAL_SEED, help="Stage 20: worlds to test on; the usual test uses EVAL_SEED")
    p.add_argument("--hunter-ckpt", default="pop_latest.npz", help="Stage 20b: which of --hunters' kept populations to test (pop_u0200.npz)")
    args = p.parse_args(argv)

    hunters, grazers = Path(args.hunters), Path(args.grazers)
    config = json.loads((grazers / "config.json").read_text())
    world, wcfg = load_world_config(config)
    try:
        wcfg = world_set(wcfg, args.world_set)
    except ValueError as e:
        p.error(str(e))
    net = make_net(world, wcfg, 0)
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    pool_h, _ = evolved_pool(Path(args.bodies) if args.bodies else hunters)
    pool_g, _ = evolved_pool(grazers)
    pool = pool_g._replace(pred_genes=pool_h.pred_genes, pred_gen=pool_h.pred_gen, pred_w=pool_h.pred_w)
    snap_g = dict(np.load(grazers / "checkpoints" / "pop_latest.npz"))
    E = args.episodes
    key, k_reset, k_prey, k_pred = jax.random.split(jax.random.PRNGKey(args.eval_seed), 4)
    if args.founder:
        one = load_brains(hunters / "checkpoints" / "latest.msgpack", template)[1]
        pred = tile(one, E, wcfg.max_pred, k_pred, 0.0)
    else:
        snap_h = dict(np.load(hunters / "checkpoints" / args.hunter_ckpt))
        if args.births:
            snap_h = {k.replace("pred_birth_", "pred_") if k.startswith("pred_birth_") else k: v for k, v in snap_h.items() if not (k.startswith("pred_") and not k.startswith("pred_birth_"))} | {k: v for k, v in snap_h.items() if k.startswith("pred_birth_")}
        pred = frozen(snap_h, "pred", template, E, wcfg.max_pred, k_pred)
    state = jax.vmap(lambda c: world.reset(wcfg, c, pool))(jax.random.split(k_reset, E))
    prey_obs, pred_obs = jax.vmap(lambda s: world.observe(wcfg, s))(state)
    carry = Carry(state, frozen(snap_g, "prey", template, E, wcfg.max_prey, k_prey), pred, prey_obs, pred_obs, key)
    frozen_cfg = IndivConfig(mode="darwin", sigma=0.0)
    hunter_cfg, grazer_cfg = frozen_cfg, frozen_cfg
    trained = json.loads((hunters / "config.json").read_text()).get("indiv", {})
    if trained.get("mode") in CULTURE and not args.founder and not args.frozen and not args.births:
        hunter_cfg = IndivConfig(**{k: v for k, v in trained.items() if k != "steps"})
        if args.both_live:
            grazer_cfg = hunter_cfg
    blank = None
    if hunter_cfg.mode in CULTURE:
        blank = blank_brain(net, wcfg.view, obs_channels(world, wcfg))
        carry = carry._replace(pred_kin=fresh_kin(E, wcfg.max_pred, hunter_cfg.childhood))
        if grazer_cfg.mode in CULTURE:
            carry = carry._replace(prey_kin=fresh_kin(E, wcfg.max_prey, grazer_cfg.childhood))
    fixed = None
    if args.random:
        from simulations.predprey import NUM_ACTIONS
        fixed = lambda k, o: jax.random.randint(k, o.shape[:2], 0, NUM_ACTIONS)
    step = make_step(net, world, wcfg, hunter_cfg, restart=False, prey_cfg=grazer_cfg, blank=blank, fixed_pred=fixed)

    upkeep_of = getattr(world, "hunter_upkeep", None)

    def counted(c, x):
        paid = (upkeep_of(wcfg, c.state) * c.state.pred.alive).sum() if upkeep_of else 0.0
        payers = c.state.pred.alive.sum()
        c, info = step(c, x)
        return c, (info, paid, payers)

    @jax.jit
    def chunk(c):
        c, (info, paid, alive) = jax.lax.scan(counted, c, None, length=100)
        return c, info.catches.sum(), info.pred_alive.sum(), paid.sum(), alive.sum()

    catches, hunter_steps, paid, alive_steps = 0.0, 0.0, 0.0, 0.0
    for _ in range(max(args.steps // 100, 1)):
        carry, c, h, u, a = chunk(carry)
        catches, hunter_steps, paid, alive_steps = catches + float(c), hunter_steps + float(h), paid + float(u), alive_steps + float(a)
    out = {"hunters": ("random " if args.random else "") + hunters.name, "grazers": grazers.name, "catches_per_hunter_step": catches / max(hunter_steps, 1.0),
           "hunters_at_end": float(carry.state.pred.alive.sum(-1).mean()), "grazers_at_end": float(carry.state.prey.alive.sum(-1).mean()),
           "both_alive": float((carry.state.pred.alive.any(-1) & carry.state.prey.alive.any(-1)).mean()),
           "catches": catches, "hunter_steps": hunter_steps, "hunters_alive_at_end_frac": float(carry.state.pred.alive.any(-1).mean()),
           "eval_seed": args.eval_seed, "world_set": args.world_set, "bodies": args.bodies, "hunter_ckpt": args.hunter_ckpt}
    if upkeep_of:  # body upkeep per hunter-step, from the hunters alive at the start of each step
        out["upkeep_per_hunter_step"] = paid / max(alive_steps, 1.0)
        out["break_even_catches_per_hunter_step"] = out["upkeep_per_hunter_step"] / wcfg.catch_energy
    (grazers / (args.out or f"cross_{'random_' if args.random else ''}{hunters.name}.json")).write_text(json.dumps(out, indent=1))
    print(f"{hunters.name} against {grazers.name}: {out['catches_per_hunter_step']:.4f} catches per hunter-step; both alive in {out['both_alive']:.3f} of worlds; hunters {out['hunters_at_end']:.1f}, grazers {out['grazers_at_end']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
