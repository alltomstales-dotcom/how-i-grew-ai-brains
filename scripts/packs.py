"""Stage 23 tests: packs with cheaters. Spec: docs/specs/2026-11-05-pack-cheaters-design.md

  python -m scripts.packs gate --run E=DIR --run P=DIR --run N=DIR [--lo 101 --hi 200] [--out FILE]
  python -m scripts.packs bar1 --run E=DIR --run P=DIR --run N=DIR [--lo 501 --hi 600] [--out FILE]
  python -m scripts.packs summary --run NAME=DIR [...] [--lo 501 --hi 600] [--out FILE]
  python -m scripts.packs eval --policy E=DIR --policy P=DIR --effort E=DIR --effort P=DIR --grazer own=DIR [--grazer held=DIR]
         --out DIR --prefix s0 [--seed 12345] [--episodes 256]
  python -m scripts.packs score --dir DIR --prefix s0 [--spread F] [--out FILE]
  python -m scripts.packs spread --dir DIR --prefix P --cell E_E --grazer own --seeds 12345 1 2

`gate`, `bar1` and `summary` read the runs' metrics.jsonl (one row per update). Intervals there come from resampling the
10-update blocks of the window, 2,000 times. `eval` plays every policy (a run's last hunter brain, under that run's sharing
rule) with every effort distribution (the effort genes of a run's last gene snapshot), against frozen grazers, in 256 fresh
worlds x 500 steps: no restarts, no learning, no mutation (children copy their parent's effort), founder bodies. Every cell
starts from the same reset keys, and each hunter slot takes the effort at the same quantile of each distribution, so cells
differ only in policy and effort. Files: DIR/<prefix>_<policy>_<effort>_<grazer>.npz with per-world counts.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_brains
from agents.networks import init_params
from simulations import evo
from simulations.worlds import make_net, obs_channels

EVAL_SEED = 12345
BOOT_BLOCKS = 2000
BOOT_WORLDS = 1000
BLOCK = 10
FIELDS = ("catches", "pred_steps", "pred_starved", "pred_births", "prey_steps", "prey_deaths", "attempts", "attempt_hunters",
          "catch_hunters", "effort_sum", "free_kept", "kept", "pred_extinct_step")
EFFORT_BAR, EFFORT_NEAR = 0.10, 0.08
CATCH_BAR = 0.80


def named(pairs):
    out = {}
    for p in pairs:
        name, sep, val = p.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=VALUE, got {p!r}")
        out[name] = val
    return out


def rows(run: str | Path) -> list[dict]:
    return [json.loads(line) for line in (Path(run) / "metrics.jsonl").read_text().splitlines() if line.strip()]


def window(rs, lo, hi):
    w = [r for r in rs if lo <= r["update"] <= hi]
    if len(w) < hi - lo + 1:
        raise SystemExit(f"window {lo}-{hi} has {len(w)} updates")
    return w


def blocks(w, keys):
    """{key: (n_blocks,) sums over each 10-update block}."""
    n = len(w) // BLOCK
    return {k: np.array([sum(r[k] for r in w[i * BLOCK:(i + 1) * BLOCK]) for i in range(n)], float) for k in keys}


def boot_idx(n, seed, reps=BOOT_BLOCKS):
    return np.random.default_rng(seed).integers(0, n, (reps, n))


def interval(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------------------------------------- gate (pilot)

def gate_arm(w, seed):
    b = blocks(w, ("birth_blocked", "birth_eligible", "pred_births", "cov_ee", "var_e", "var_energy", "breed_dev_sum"))
    idx = boot_idx(len(b["cov_ee"]), seed)
    corr = lambda i: b["cov_ee"][i].sum(-1) / np.sqrt(np.maximum(b["var_e"][i].sum(-1) * b["var_energy"][i].sum(-1), 1e-12))
    dev = lambda i: b["breed_dev_sum"][i].sum(-1) / np.maximum(b["pred_births"][i].sum(-1), 1e-9)
    all_ = np.arange(len(b["cov_ee"]))
    gen = [r["pred_gen"] for r in w]
    return {
        "blocked_share": float(b["birth_blocked"].sum() / max(b["birth_eligible"].sum(), 1e-9)),
        "eligible": float(b["birth_eligible"].sum()), "blocked": float(b["birth_blocked"].sum()),
        "gen_last10": float(np.mean(gen[-10:])), "gen_mid10": float(np.mean(gen[-60:-50])),
        "births_per_block_min": float(b["pred_births"].min()),
        "births_per_hunter_step": float(sum(r["pred_births"] for r in w) / max(sum(r["hunter_steps"] for r in w), 1)),
        "effort_mean": float(np.mean([r["pred_effort"] for r in w])), "effort_std": float(np.mean([r["pred_effort_std"] for r in w])),
        "corr": float(corr(all_)), "corr_ci": interval(corr(idx)),
        "breed_dev": float(dev(all_)), "breed_dev_ci": interval(dev(idx)),
        "hunters": float(np.mean([r["pred_alive"] for r in w])),
        "catches_per_hunter_step": float(sum(r["catches"] for r in w) / max(sum(r["hunter_steps"] for r in w), 1)),
    }


def excludes0(ci):
    return ci[0] > 0 or ci[1] < 0


def cmd_gate(args) -> int:
    runs = named(args.run)
    arms = {a: gate_arm(window(rows(d), args.lo, args.hi), 23 + i) for i, (a, d) in enumerate(runs.items())}
    c1 = all(v["blocked_share"] < 0.05 for v in arms.values())
    c2 = all(v["gen_last10"] >= 10 and v["gen_last10"] - v["gen_mid10"] >= 2 and v["births_per_block_min"] > 0 for v in arms.values())
    sel = [a for a in ("E", "P") if a in arms]
    c3_parts = {a: {"mixed": arms[a]["effort_std"] >= 0.03, "corr_nonzero": excludes0(arms[a]["corr_ci"]),
                    "same_sign": np.sign(arms[a]["breed_dev"]) == np.sign(arms[a]["corr"]),
                    "dev_nonzero": excludes0(arms[a]["breed_dev_ci"])} for a in sel}
    c3 = (all(p["mixed"] and p["corr_nonzero"] and p["same_sign"] for p in c3_parts.values())
          and any(p["dev_nonzero"] for p in c3_parts.values()))
    out = {"window": [args.lo, args.hi], "arms": arms, "criteria": {"1_free_slots": bool(c1), "2_turnover": bool(c2),
           "3_selection": bool(c3), "3_parts": {a: {k: bool(v) for k, v in p.items()} for a, p in c3_parts.items()}},
           "gate": "pass" if (c1 and c2 and c3) else "fail"}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------------------------------------- bar 1

def cmd_bar1(args) -> int:
    runs = named(args.run)
    per = {a: np.array([r["pred_effort"] for r in window(rows(d), args.lo, args.hi)]) for a, d in runs.items()}
    blk = {a: v[: len(v) // BLOCK * BLOCK].reshape(-1, BLOCK).mean(1) for a, v in per.items()}
    rng = np.random.default_rng(2301)
    boot = {a: b[rng.integers(0, len(b), (BOOT_BLOCKS, len(b)))].mean(1) for a, b in blk.items()}
    out = {"window": [args.lo, args.hi], "effort": {a: float(v.mean()) for a, v in per.items()},
           "effort_ci": {a: interval(boot[a]) for a in per}, "diffs": {}}
    verdicts = []
    for other in ("P", "N"):
        d = float(per[other].mean() - per["E"].mean())
        ci = interval(boot[other] - boot["E"])
        v = "pass" if d >= EFFORT_BAR and ci[0] > 0 else ("no result" if (EFFORT_NEAR <= d < EFFORT_BAR) or d >= EFFORT_BAR else "fail")
        out["diffs"][f"{other}-E"] = {"diff": d, "ci": ci, "verdict": v}
        verdicts.append(v)
    out["verdict"] = "pass" if all(v == "pass" for v in verdicts) else ("fail" if "fail" in verdicts else "no result")
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------ summary

def summarise(rs, lo, hi, world_steps):
    w = window(rs, lo, hi)
    s = lambda k: float(sum(r[k] for r in w))
    hs = max(s("hunter_steps"), 1.0)
    ws = world_steps * len(w)
    return {
        "effort": float(np.mean([r["pred_effort"] for r in w])), "effort_std": float(np.mean([r["pred_effort_std"] for r in w])),
        "hunters": float(np.mean([r["pred_alive"] for r in w])), "grazers": float(np.mean([r["prey_alive"] for r in w])),
        "hunter_steps": hs, "catches": s("catches"), "catches_per_hunter_step": s("catches") / hs,
        "attempts": s("attempts"), "catch_share_of_attempts": s("catches") / max(s("attempts"), 1),
        "group_size_catch": s("catch_hunters") / max(s("catches"), 1), "group_size_attempt": s("attempt_hunters") / max(s("attempts"), 1),
        "lone_assigned_per_hunter_step": s("lone_assigned") / hs,
        "starved_per_hunter_step": s("pred_starved") / hs, "births_per_hunter_step": s("pred_births") / hs,
        "hunter_extinct_restarts_per_1000_world_steps": 1000 * s("pred_extinct_restarts") / ws,
        "grazer_extinct_restarts_per_1000_world_steps": 1000 * s("prey_extinct_restarts") / ws,
        "free_kept_share": s("free_kept_share") / len(w), "clipped_share": 1 - s("kept_total") / max(s("meal_total"), 1e-9),
        "blocked_share": s("birth_blocked") / max(s("birth_eligible"), 1), "gen": float(np.mean([r["pred_gen"] for r in w])),
        "breed_dev": s("breed_dev_sum") / max(s("pred_births"), 1),
        "effort_energy_corr": s("cov_ee") / max(np.sqrt(s("var_e") * s("var_energy")), 1e-12),
        "wall_s": float(rs[-1]["wall_s"]), "updates": int(rs[-1]["update"]),
    }


def cmd_summary(args) -> int:
    out = {}
    for name, d in named(args.run).items():
        rs = rows(d)
        cfg = json.loads((Path(d) / "config.json").read_text())
        world_steps = cfg["train"]["num_envs"] * cfg["train"]["rollout_len"]
        out[name] = summarise(rs, args.lo, args.hi, world_steps)
        out[name]["trajectory"] = {r["update"]: [round(r["pred_effort"], 4), round(r["pred_alive"], 2), round(r["catches_per_hunter_step"], 5)]
                                   for r in rs if r["update"] in (1, 50, 100, 150, 200, 300, 400, 500, 600) or r["update"] == rs[-1]["update"]}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------------------------------------- eval

def run_config(run: str | Path):
    from scripts.eval import load_world_config
    world, cfg = load_world_config(json.loads((Path(run) / "config.json").read_text()))
    if not cfg.effort:
        raise SystemExit(f"{run}: not an effort world")
    return dataclasses.replace(cfg, gene_sigma=0.0)  # no mutation during the test


def effort_genes(run: str | Path) -> np.ndarray:
    with np.load(Path(run) / "checkpoints" / "genes_latest.npz") as d:
        return np.asarray(d["pred_genes"], np.float32)[:, evo.EFFORT]


def make_rollout(cfg, net, episodes):
    @jax.jit
    def run(prey_p, pred_p, effort, key):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(evo.reset, cfg))(jax.random.split(k_reset, episodes))
        state = state._replace(pred=state.pred._replace(genes=state.pred.genes.at[..., evo.EFFORT].set(effort)))
        z = jnp.zeros(episodes, jnp.float32)
        acc0 = {f: z for f in FIELDS[:-1]}
        acc0["pred_extinct_step"] = jnp.full(episodes, -1, jnp.int32)

        def body(carry, x):
            state, acc = carry
            k, t = x
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state)
            prey_a = jax.random.categorical(kp, net.apply(prey_p, prey_obs)[0])
            pred_a = jax.random.categorical(kq, net.apply(pred_p, pred_obs)[0])
            res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            i, x_ = res.info, res.info.effort
            gone = (acc["pred_extinct_step"] < 0) & (res.state.pred.alive.sum(-1) == 0) & (state.pred.alive.sum(-1) > 0)
            acc = {
                "catches": acc["catches"] + i.catches, "pred_steps": acc["pred_steps"] + state.pred.alive.sum(-1),
                "pred_starved": acc["pred_starved"] + i.pred_deaths, "pred_births": acc["pred_births"] + i.pred_births,
                "prey_steps": acc["prey_steps"] + state.prey.alive.sum(-1), "prey_deaths": acc["prey_deaths"] + i.prey_deaths,
                "attempts": acc["attempts"] + x_.attempts, "attempt_hunters": acc["attempt_hunters"] + x_.attempt_hunters,
                "catch_hunters": acc["catch_hunters"] + x_.catch_hunters,
                "effort_sum": acc["effort_sum"] + (state.pred.genes[..., evo.EFFORT] * state.pred.alive).sum(-1),
                "free_kept": acc["free_kept"] + x_.free_kept, "kept": acc["kept"] + x_.kept_total,
                "pred_extinct_step": jnp.where(gone, t + 1, acc["pred_extinct_step"]),
            }
            return (res.state, acc), None

        xs = (jax.random.split(k_steps, cfg.max_steps), jnp.arange(cfg.max_steps))
        (_, acc), _ = jax.lax.scan(body, (state, acc0), xs)
        return acc

    return run


def matched_effort(dist: np.ndarray, episodes: int, slots: int, seed: int) -> np.ndarray:
    """Each slot's effort at the same quantile of every distribution: one shared uniform draw per (world, slot)."""
    u = np.random.default_rng(seed + 2323).random((episodes, slots))
    return np.quantile(np.asarray(dist, np.float64), u).astype(np.float32)


def cmd_eval(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policies, efforts, grazers = named(args.policy), named(args.effort), named(args.grazer)
    dists = {k: effort_genes(v) for k, v in efforts.items()}
    for pname, prun in policies.items():
        cfg = run_config(prun)
        net = make_net(evo, cfg, 0)
        template = init_params(net, jax.random.PRNGKey(0), cfg.view, obs_channels(evo, cfg))
        roll = make_rollout(cfg, net, args.episodes)
        _, pred_p, pmeta = load_brains(Path(prun) / "checkpoints" / "latest.msgpack", template)
        for gname, grun in grazers.items():
            prey_p, _, gmeta = load_brains(Path(grun) / "checkpoints" / "latest.msgpack", template)
            for ename, dist in dists.items():
                eff = jnp.asarray(matched_effort(dist, args.episodes, cfg.max_pred, args.seed))
                acc = {k: np.asarray(v) for k, v in jax.device_get(roll(prey_p, pred_p, eff, jax.random.PRNGKey(args.seed))).items()}
                path = out / f"{args.prefix}_{pname}_{ename}_{gname}.npz"
                meta = {"policy": prun, "policy_update": pmeta.get("update"), "rule": cfg.effort, "effort_from": efforts[ename],
                        "effort_mean": float(dist.mean()), "effort_std": float(dist.std()), "grazer": grun, "seed": args.seed,
                        "episodes": args.episodes, "steps": cfg.max_steps}
                np.savez(path, meta=json.dumps(meta), **acc)
                rate = acc["catches"].sum() / max(acc["pred_steps"].sum(), 1)
                print(f"{path.name}: rule {cfg.effort} effort {dist.mean():.3f}  catches/hunter-step {rate:.4f}  catches "
                      f"{acc['catches'].sum():.0f}  hunter-steps {acc['pred_steps'].sum():.0f}  starved {acc['pred_starved'].sum():.0f}  "
                      f"extinct {(acc['pred_extinct_step'] >= 0).mean():.2f}", flush=True)
    return 0


# -------------------------------------------------------------------------------------------------------- score

def load(d, prefix, policy, effort, grazer):
    z = np.load(Path(d) / f"{prefix}_{policy}_{effort}_{grazer}.npz")
    return {k: z[k] for k in FIELDS}


def rate(a, idx=None):
    if idx is None:
        return a["catches"].sum() / max(a["pred_steps"].sum(), 1)
    return a["catches"][idx].sum(-1) / np.maximum(a["pred_steps"][idx].sum(-1), 1)


def ratio(num, den, idx):
    b = rate(num, idx) / np.maximum(rate(den, idx), 1e-12)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"ratio": float(rate(num) / max(rate(den), 1e-12)), "lo": float(lo), "hi": float(hi)}


def cell_summary(a):
    ext = a["pred_extinct_step"] >= 0
    hs = max(a["pred_steps"].sum(), 1)
    return {"catches_per_hunter_step": float(rate(a)), "catches": float(a["catches"].sum()), "hunter_steps": float(hs),
            "starved_per_hunter_step": float(a["pred_starved"].sum() / hs), "births": float(a["pred_births"].sum()),
            "extinct_share": float(ext.mean()), "extinct_median_step": float(np.median(a["pred_extinct_step"][ext])) if ext.any() else None,
            "attempts": float(a["attempts"].sum()), "catch_share_of_attempts": float(a["catches"].sum() / max(a["attempts"].sum(), 1)),
            "group_size_catch": float(a["catch_hunters"].sum() / max(a["catches"].sum(), 1)),
            "group_size_attempt": float(a["attempt_hunters"].sum() / max(a["attempts"].sum(), 1)),
            "effort_mean_living": float(a["effort_sum"].sum() / hs), "free_kept_share": float(a["free_kept"].sum() / max(a["kept"].sum(), 1e-9)),
            "grazer_steps": float(a["prey_steps"].sum()), "grazer_deaths": float(a["prey_deaths"].sum())}


def cmd_score(args) -> int:
    d, prefix = Path(args.dir), args.prefix
    grazers = sorted({p.stem.split("_")[-1] for p in d.glob(f"{prefix}_*_*_*.npz")})
    rng = np.random.default_rng(1023)
    out = {"prefix": prefix, "spread": args.spread, "grazers": {}}
    for g in grazers:
        C = {(p, e): load(d, prefix, p, e, g) for p in ("E", "P") for e in ("E", "P")}
        n = len(C[("E", "E")]["catches"])
        idx = rng.integers(0, n, (BOOT_WORLDS, n))
        main = ratio(C[("E", "E")], C[("P", "P")], idx)
        within_E = ratio(C[("E", "E")], C[("E", "P")], idx)
        within_P = ratio(C[("P", "E")], C[("P", "P")], idx)
        attribution = within_E["hi"] < 1 and within_P["hi"] < 1
        beyond, clear = main["ratio"] <= CATCH_BAR, main["hi"] < 1
        near = (main["ratio"] - CATCH_BAR) / CATCH_BAR <= 2 * args.spread
        if beyond and clear:
            v = "pass" if attribution else "attribution fails"
        elif (beyond and not clear) or near:
            v = "no result"
        else:
            v = "fail"
        out["grazers"][g] = {"bar2": {"E_E/P_P": main, "verdict": v, "attribution": bool(attribution)},
                             "within_policy_E: E_E/E_P": within_E, "within_policy_P: P_E/P_P": within_P,
                             "policy_effect_at_D_P: E_P/P_P": ratio(C[("E", "P")], C[("P", "P")], idx),
                             "policy_effect_at_D_E: E_E/P_E": ratio(C[("E", "E")], C[("P", "E")], idx),
                             "cells": {f"{p}_{e}": cell_summary(a) for (p, e), a in C.items()}}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_spread(args) -> int:
    rates = [rate(load(args.dir, f"{args.prefix}{s}", *args.cell.split("_"), args.grazer)) for s in args.seeds]
    rel = float(np.std(rates) / np.mean(rates))
    out = {"rates": [float(r) for r in rates], "rel_std": rel, "ratio_spread": float(np.sqrt(2) * rel)}
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--run", action="append", required=True)
    g.add_argument("--lo", type=int, default=101)
    g.add_argument("--hi", type=int, default=200)
    g.add_argument("--out")
    b = sub.add_parser("bar1")
    b.add_argument("--run", action="append", required=True)
    b.add_argument("--lo", type=int, default=501)
    b.add_argument("--hi", type=int, default=600)
    b.add_argument("--out")
    s = sub.add_parser("summary")
    s.add_argument("--run", action="append", required=True)
    s.add_argument("--lo", type=int, default=501)
    s.add_argument("--hi", type=int, default=600)
    s.add_argument("--out")
    e = sub.add_parser("eval")
    e.add_argument("--policy", action="append", required=True)
    e.add_argument("--effort", action="append", required=True)
    e.add_argument("--grazer", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--prefix", required=True)
    e.add_argument("--seed", type=int, default=EVAL_SEED)
    e.add_argument("--episodes", type=int, default=256)
    c = sub.add_parser("score")
    c.add_argument("--dir", required=True)
    c.add_argument("--prefix", required=True)
    c.add_argument("--spread", type=float, default=0.0)
    c.add_argument("--out")
    r = sub.add_parser("spread")
    r.add_argument("--dir", required=True)
    r.add_argument("--prefix", required=True, help="file prefix before the seed, e.g. pilot")
    r.add_argument("--cell", default="E_E")
    r.add_argument("--grazer", default="own")
    r.add_argument("--seeds", type=int, nargs="+", default=[EVAL_SEED, 1, 2])
    args = p.parse_args(argv)
    return {"gate": cmd_gate, "bar1": cmd_bar1, "summary": cmd_summary, "eval": cmd_eval, "score": cmd_score, "spread": cmd_spread}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
