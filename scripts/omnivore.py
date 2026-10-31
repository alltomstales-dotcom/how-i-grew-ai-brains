"""Stage 9a test: do hunters leave meat when they can eat plants? python -m scripts.omnivore --omni runs/A --tough runs/B --neutral runs/C [--evolved]
Stage 9b test: which gut keeps them hunting? python -m scripts.omnivore --ladder runs/D03 runs/D08 runs/D20 [--evolved]

Measures the diet gene of living hunters and where their energy comes from, in each world. Bars are in the spec.
Writes omnivore.json in the --omni run directory.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.policies import forget_after, pair_act, pair_memory
from scripts.eval import EVAL_SEED
from scripts.meals import load

LEAVE_TARGET, GUT_TARGET = 0.15, 0.10
HUNTER_TARGET, PLANT_TARGET = 0.50, 0.25  # stage 9b: diet at the worst gut, at the best


def feeding(world, cfg, prey_pi, pred_pi, key, episodes: int, pool=None) -> dict:
    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def body(carry, k):
            state, acc, mem = carry
            k_act, k_step = jax.random.split(k)
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            i = res.info
            diet = state.pred.genes[..., 5]
            acc = acc + jnp.stack([(diet * state.pred.alive).sum(), (diet ** 2 * state.pred.alive).sum(), state.pred.alive.sum(), state.prey.alive.sum(),
                                   i.catches.sum(), i.pred_bites.sum(), i.pred_plant_gain.sum(), i.pred_meat_gain.sum(), i.plant_cover.sum()]).astype(jnp.float32)
            return (res.state, acc, forget_after(mem, res)), None

        init = (state, jnp.zeros(9, jnp.float32), pair_memory(prey_pi, pred_pi, state))
        (_, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    d, d2, p, q, c, b, pg, mg, cover = (float(v) for v in np.asarray(run(key)))
    mean = d / max(p, 1.0); steps = episodes * cfg.max_steps
    return {"diet": mean, "diet_sd": max(d2 / max(p, 1.0) - mean ** 2, 0.0) ** 0.5, "catches_per_pred_step": c / max(p, 1.0), "bites_per_pred_step": b / max(p, 1.0),
            "plant_share": pg / max(pg + mg, 1e-6), "hunters": p / steps, "grazers": q / steps, "plant_cover": cover / steps}


def line(d: dict) -> str:
    return (f"diet {d['diet']:.3f} (sd {d['diet_sd']:.3f}), catches {d['catches_per_pred_step']:.4f}, bites {d['bites_per_pred_step']:.3f}, "
            f"plant share {d['plant_share']:.3f}, hunters {d['hunters']:.1f}, grazers {d['grazers']:.1f}, plant cover {d['plant_cover']:.2f}")


def verdict(omni: dict, tough: dict, neutral: dict) -> dict:
    out = {"left_meat": neutral["diet"] - omni["diet"], "gut_keeps": tough["diet"] - omni["diet"]}
    out["leave_pass"] = bool(out["left_meat"] >= LEAVE_TARGET)
    out["gut_pass"] = bool(out["gut_keeps"] >= GUT_TARGET)
    out["pass"] = out["leave_pass"] and out["gut_pass"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omni")
    p.add_argument("--tough")
    p.add_argument("--neutral")
    p.add_argument("--ladder", nargs="+", help="run directories, worst gut first; writes ladder.json in the first")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true")
    args = p.parse_args(argv)
    key = jax.random.PRNGKey(EVAL_SEED)
    res = {}
    if args.ladder:
        for path in args.ladder:
            world, cfg, prey_pi, pred_pi, pool = load(Path(path), args.evolved)
            res[str(cfg.plant_digest)] = d = feeding(world, cfg, prey_pi, pred_pi, key, args.episodes, pool)
            print(f"gut {cfg.plant_digest:<5} {line(d)}")
        low, high = res[next(iter(res))], res[list(res)[-1]]
        out = {"guts": res, "hunter_pass": bool(low["diet"] >= HUNTER_TARGET), "plant_pass": bool(high["diet"] <= PLANT_TARGET)}
        out["pass"] = out["hunter_pass"] and out["plant_pass"]
        (Path(args.ladder[0]) / "ladder.json").write_text(json.dumps(out, indent=1))
        print(f"diet {low['diet']:.3f} at the worst gut (bar at least {HUNTER_TARGET}), {high['diet']:.3f} at the best (bar at most {PLANT_TARGET}): {'PASS' if out['pass'] else 'FAIL'}")
        return 0
    for name, path in (("omni", args.omni), ("tough", args.tough), ("neutral", args.neutral)):
        world, cfg, prey_pi, pred_pi, pool = load(Path(path), args.evolved)
        res[name] = feeding(world, cfg, prey_pi, pred_pi, key, args.episodes, pool)
        print(f"{name:8s} {line(res[name])}")
    out = {**res, **verdict(res["omni"], res["tough"], res["neutral"])}
    (Path(args.omni) / "omnivore.json").write_text(json.dumps(out, indent=1))
    print(f"diet {out['left_meat']:.3f} below neutral (bar {LEAVE_TARGET}), tough gut {out['gut_keeps']:.3f} above omni (bar {GUT_TARGET}): {'PASS' if out['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
