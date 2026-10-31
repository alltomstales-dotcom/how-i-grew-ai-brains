"""Stage 9a test: do hunters leave meat when they can eat plants? python -m scripts.omnivore --omni runs/A --tough runs/B --neutral runs/C [--evolved]
Stage 9b test: which gut keeps them hunting? python -m scripts.omnivore --ladder runs/D03 runs/D08 runs/D20 [--evolved]
Stage 9c test: do winters make real omnivores? python -m scripts.omnivore --winter runs/W --control runs/C [--evolved]
Stage 9d test: do two kinds of hunter live together? python -m scripts.omnivore --camps runs/NAME [--evolved]
Over an episode, exploratory: python -m scripts.omnivore --camp-curve runs/NAME [--evolved]

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
MIX_DIET, MIX_PLANTS, MIX_SD = (0.30, 0.70), (0.20, 0.80), 0.20  # stage 9c: what counts as one mixed diet
PLANT_CAMP, MEAT_CAMP, TOGETHER_STEP, TOGETHER_TARGET = 0.4, 0.6, 400, 0.25  # stage 9d


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


def camps(world, cfg, prey_pi, pred_pi, key, episodes: int, pool=None) -> dict:
    """Of the worlds with hunters alive at TOGETHER_STEP, how many hold both a plant eater and a meat eater?"""
    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def counts(s):
            diet, alive = s.pred.genes[..., 5], s.pred.alive
            return (alive & (diet < PLANT_CAMP)).sum(-1), (alive & (diet > MEAT_CAMP)).sum(-1), alive.sum(-1), (s.pred.energy * alive * (diet < PLANT_CAMP)).sum(-1), (s.pred.energy * alive * (diet > MEAT_CAMP)).sum(-1)

        def body(carry, k):
            state, mem = carry
            k_act, k_step = jax.random.split(k)
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            return (res.state, forget_after(mem, res)), None

        start = counts(state)
        (state, _), _ = jax.lax.scan(body, (state, pair_memory(prey_pi, pred_pi, state)), jax.random.split(k_steps, TOGETHER_STEP))
        return start, counts(state)

    (p0, m0, _, _, _), (p, m, a, pe, me) = jax.device_get(run(key))
    living = a > 0
    together = (p > 0) & (m > 0)
    return {"worlds_with_hunters": int(living.sum()), "together_at_start": float(((p0 > 0) & (m0 > 0)).mean()),
            "together_share": float(together.sum() / max(living.sum(), 1)), "plant_eaters": float(p.sum() / episodes), "meat_eaters": float(m.sum() / episodes),
            "plant_eater_energy": float(pe.sum() / max(p.sum(), 1)), "meat_eater_energy": float(me.sum() / max(m.sum(), 1))}


def camp_curve(world, cfg, prey_pi, pred_pi, key, episodes: int, pool=None) -> list[dict]:
    """Every 25 steps: worlds with hunters, worlds holding both kinds, each kind's numbers, energy and share of births."""
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
            diet, alive = state.pred.genes[..., 5], state.pred.alive
            low, high = alive & (diet < PLANT_CAMP), alive & (diet > MEAT_CAMP)
            row = jnp.stack([(alive.sum(-1) > 0).sum(), ((low.sum(-1) > 0) & (high.sum(-1) > 0)).sum(), low.sum(), high.sum(),
                             (state.pred.energy * low).sum(), (state.pred.energy * high).sum(), (state.prey.alive.sum(-1)).sum()]).astype(jnp.float32)
            k_act, k_step = jax.random.split(k)
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            return (res.state, forget_after(mem, res)), row

        _, rows = jax.lax.scan(body, (state, pair_memory(prey_pi, pred_pi, state)), jax.random.split(k_steps, cfg.max_steps))
        return rows

    rows = np.asarray(run(key))
    out = []
    for t in range(0, cfg.max_steps, 25):
        w, both, lo, hi, le, he, gr = (float(v) for v in rows[t])
        out.append({"step": t, "worlds_with_hunters": int(w), "worlds_with_both": int(both), "plant_eaters": lo / episodes, "meat_eaters": hi / episodes,
                    "plant_eater_energy": le / max(lo, 1.0), "meat_eater_energy": he / max(hi, 1.0), "grazers": gr / episodes})
    return out


def line(d: dict) -> str:
    return (f"diet {d['diet']:.3f} (sd {d['diet_sd']:.3f}), catches {d['catches_per_pred_step']:.4f}, bites {d['bites_per_pred_step']:.3f}, "
            f"plant share {d['plant_share']:.3f}, hunters {d['hunters']:.1f}, grazers {d['grazers']:.1f}, plant cover {d['plant_cover']:.2f}")


def verdict(omni: dict, tough: dict, neutral: dict) -> dict:
    out = {"left_meat": neutral["diet"] - omni["diet"], "gut_keeps": tough["diet"] - omni["diet"]}
    out["leave_pass"] = bool(out["left_meat"] >= LEAVE_TARGET)
    out["gut_pass"] = bool(out["gut_keeps"] >= GUT_TARGET)
    out["pass"] = out["leave_pass"] and out["gut_pass"]
    return out


def winter_verdict(winter: dict, control: dict) -> dict:
    inside = lambda v, band: band[0] <= v <= band[1]
    out = {"mixed_pass": inside(winter["diet"], MIX_DIET) and inside(winter["plant_share"], MIX_PLANTS),
           "one_camp_pass": winter["diet_sd"] <= MIX_SD, "control_pass": not inside(control["diet"], MIX_DIET)}
    out["pass"] = out["mixed_pass"] and out["one_camp_pass"] and out["control_pass"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omni")
    p.add_argument("--tough")
    p.add_argument("--neutral")
    p.add_argument("--ladder", nargs="+", help="run directories, worst gut first; writes ladder.json in the first")
    p.add_argument("--camps")
    p.add_argument("--camp-curve")
    p.add_argument("--winter")
    p.add_argument("--control")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true")
    args = p.parse_args(argv)
    key = jax.random.PRNGKey(EVAL_SEED)
    res = {}
    if args.camp_curve:
        world, cfg, prey_pi, pred_pi, pool = load(Path(args.camp_curve), args.evolved)
        rows = camp_curve(world, cfg, prey_pi, pred_pi, key, args.episodes, pool)
        (Path(args.camp_curve) / "camp_curve.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(f"step {r['step']:3d}: hunters in {r['worlds_with_hunters']:3d} worlds, both kinds in {r['worlds_with_both']:3d}; per world {r['plant_eaters']:.2f} plant eaters at {r['plant_eater_energy']:.1f}, "
                  f"{r['meat_eaters']:.2f} meat eaters at {r['meat_eater_energy']:.1f}; grazers {r['grazers']:.1f}")
        return 0
    if args.camps:
        world, cfg, prey_pi, pred_pi, pool = load(Path(args.camps), args.evolved)
        d = camps(world, cfg, prey_pi, pred_pi, key, args.episodes, pool)
        d["pass"] = bool(d["together_share"] >= TOGETHER_TARGET)
        (Path(args.camps) / "camps.json").write_text(json.dumps(d, indent=1))
        print(f"both kinds alive at step {TOGETHER_STEP} in {d['together_share']:.3f} of {d['worlds_with_hunters']} worlds with hunters (bar {TOGETHER_TARGET}; {d['together_at_start']:.3f} of worlds at the start); "
              f"per world {d['plant_eaters']:.1f} plant eaters at {d['plant_eater_energy']:.1f} energy, {d['meat_eaters']:.1f} meat eaters at {d['meat_eater_energy']:.1f}: {'PASS' if d['pass'] else 'FAIL'}")
        return 0
    if args.winter:
        for name, path in (("winter", args.winter), ("control", args.control)):
            world, cfg, prey_pi, pred_pi, pool = load(Path(path), args.evolved)
            res[name] = feeding(world, cfg, prey_pi, pred_pi, key, args.episodes, pool)
            print(f"{name:8s} {line(res[name])}")
        out = {**res, **winter_verdict(res["winter"], res["control"])}
        (Path(args.winter) / "winter.json").write_text(json.dumps(out, indent=1))
        print(f"mixed diet {out['mixed_pass']}, one camp {out['one_camp_pass']}, control not mixed {out['control_pass']}: {'PASS' if out['pass'] else 'FAIL'}")
        return 0
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
