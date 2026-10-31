"""Stage 10a: is there a world that lasts? python -m scripts.lasting [--steps 2000] [--episodes 64]

Hand-written animals, no training: grazers walk to the nearest plant in view, hunters to the nearest grazer. A grid of
settings, each on two sets of random seeds. The bar is in the spec. Writes lasting.json in --out.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.policies import chase_policy
from simulations import evo

BOTH_TARGET = 0.5
GRID = {"seed_rate": (0.03, 0.08, 0.2), "bite_regrow": (0.1, 0.3), "catch_energy": (25.0, 75.0, 150.0), "die_rate": (0.02, 0.005)}
PLANT, PREY = 0, 2  # observation channels


def play(cfg, key, episodes: int):
    """(steps, 5): worlds with both species, with grazers, then hunters, grazers and plant cover per world, every step."""
    graze, hunt = chase_policy(PLANT), chase_policy(PREY)

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(evo.reset, cfg))(jax.random.split(k_reset, episodes))

        def body(state, k):
            hunters, grazers = state.pred.alive.sum(-1), state.prey.alive.sum(-1)
            row = jnp.stack([((hunters > 0) & (grazers > 0)).sum(), (grazers > 0).sum(), hunters.sum(), grazers.sum(), state.plant.mean((-2, -1)).sum()]).astype(jnp.float32)
            kp, kq, ks = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state)
            prey_a = graze(kp, prey_obs); pred_a = hunt(kq, pred_obs)
            return jax.vmap(partial(evo.advance, cfg))(jax.random.split(ks, episodes), state, prey_a, pred_a).state, row

        _, rows = jax.lax.scan(body, state, jax.random.split(k_steps, cfg.max_steps))
        return rows

    return np.asarray(run(key)) / episodes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--out", default="runs")
    p.add_argument("--tiny", action="store_true", help="tests: a small map and one setting")
    args = p.parse_args(argv)
    base = evo.EvoConfig(max_steps=args.steps) if not args.tiny else evo.EvoConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=args.steps)
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    grid = grid[:1] if args.tiny else grid
    results = []
    for setting in grid:
        cfg = replace(base, **setting)
        ends = []
        for s in (0, 1):
            r = play(cfg, jax.random.PRNGKey(100 + s), args.episodes)
            ends.append(r)
        both = [float(r[-1, 0]) for r in ends]
        marks = [t for t in (100, 500, 1000, args.steps - 1) if t < args.steps]
        out = {**setting, "both_at_end": both, "pass": all(b >= BOTH_TARGET for b in both),
               "curve": [{"step": t, "both": float(ends[0][t, 0]), "grazers": float(ends[0][t, 3]), "hunters": float(ends[0][t, 2]), "plants": float(ends[0][t, 4])} for t in marks]}
        results.append(out)
        print(" ".join(f"{k}={v}" for k, v in setting.items()), f"both {both[0]:.2f} {both[1]:.2f}", "|",
              " ".join(f"t{c['step']}: {c['grazers']:.1f}g {c['hunters']:.1f}h {c['plants']:.2f}p" for c in out["curve"]), "PASS" if out["pass"] else "", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "lasting.json").write_text(json.dumps(results, indent=1))
    n = sum(r["pass"] for r in results)
    print(f"{n} of {len(results)} settings hold both species at the end in at least {BOTH_TARGET} of worlds, on both seed sets: {'PASS' if n else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
