"""Stage 10a test: do hunters last? python -m scripts.persist --run-dir runs/NAME [--evolved]

Plays whole episodes with no restarts and counts the worlds that still hold both species. Bars are in the spec.
Writes persist.json in the run directory.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.policies import chase_policy, forget_after, pair_act, pair_memory
from scripts.eval import EVAL_SEED
from scripts.meals import load

BOTH_TARGET = 0.5


def curve(world, cfg, prey_pi, pred_pi, key, episodes: int, pool=None) -> np.ndarray:
    """(steps, 9): worlds with both species alive, worlds with hunters, hunters, grazers, before each step, then
    grazer energy (total), plant cover, unbitten plants, grazer births and grazer deaths."""
    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def body(carry, k):
            state, mem = carry
            hunters, grazers = state.pred.alive.sum(-1), state.prey.alive.sum(-1)
            row = [((hunters > 0) & (grazers > 0)).sum(), (hunters > 0).sum(), hunters.sum(), grazers.sum(), (state.prey.energy * state.prey.alive).sum(),
                   state.plant.mean(), (state.plant & ~state.bitten).mean()]
            k_act, k_step = jax.random.split(k)
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            row = jnp.stack(row + [res.info.prey_births.sum(), res.info.prey_deaths.sum()]).astype(jnp.float32)
            return (res.state, forget_after(mem, res)), row

        _, rows = jax.lax.scan(body, (state, pair_memory(prey_pi, pred_pi, state)), jax.random.split(k_steps, cfg.max_steps))
        return rows

    return np.asarray(run(key))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true")
    p.add_argument("--steps", type=int, help="play this long instead of the run's own max_steps")
    p.add_argument("--bot-grazers", action="store_true", help="probe only: grazers walk to the nearest plant in view; writes nothing")
    p.add_argument("--bot-hunters", action="store_true", help="probe only: hunters walk to the nearest grazer in view; writes nothing")
    p.add_argument("--seed-rate", type=float, help="probe only: override how fast plants spread; writes nothing")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    world, cfg, prey_pi, pred_pi, pool = load(run_dir, args.evolved)
    if args.steps:
        cfg = replace(cfg, max_steps=args.steps)
    if args.seed_rate is not None:
        cfg = replace(cfg, seed_rate=args.seed_rate)
    if args.bot_grazers:
        prey_pi = chase_policy(0)
    if args.bot_hunters:
        pred_pi = chase_policy(2)
    rows = curve(world, cfg, prey_pi, pred_pi, jax.random.PRNGKey(EVAL_SEED), args.episodes, pool)
    table = [{"step": int(t), "both_share": float(rows[t, 0] / args.episodes), "hunter_worlds": float(rows[t, 1] / args.episodes),
              "hunters": float(rows[t, 2] / args.episodes), "grazers": float(rows[t, 3] / args.episodes),
              "grazer_energy": float(rows[t, 4] / max(rows[t, 3], 1.0)), "plant_cover": float(rows[t, 5]), "unbitten": float(rows[t, 6]),
              "grazer_births": float(rows[t, 7] / args.episodes), "grazer_deaths": float(rows[t, 8] / args.episodes)}
             for t in [*range(0, cfg.max_steps, max(50, cfg.max_steps // 10)), cfg.max_steps - 1]]
    last = table[-1]
    out = {"curve": table, "both_at_end": last["both_share"], "pass": bool(last["both_share"] >= BOTH_TARGET)}
    if args.seed_rate is None and not args.bot_grazers and not args.bot_hunters:
        (run_dir / "persist.json").write_text(json.dumps(out, indent=1))
    for r in table:
        print(f"step {r['step']:3d}: both species in {r['both_share']:.3f} of worlds; hunters {r['hunters']:.2f}, grazers {r['grazers']:.1f} at {r['grazer_energy']:.1f} energy; plants {r['plant_cover']:.2f}, unbitten {r['unbitten']:.3f}; births {r['grazer_births']:.2f}, deaths {r['grazer_deaths']:.2f}")
    print(f"both alive at step {last['step']} in {last['both_share']:.3f} of worlds (bar {BOTH_TARGET}): {'PASS' if out['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
