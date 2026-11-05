"""Stage 22 tests: the office. Spec: docs/specs/2026-11-05-office-design.md

  python -m scripts.office eval --hunter NAME=RUN [...] --grazer NAME=RUN [...] --layout home|test|train [...] --out DIR
         [--prefix P] [--seed 12345] [--episodes 256] [--world-set KEY=VALUE ...]
  python -m scripts.office score --dir DIR --seed S [--spread F] [--out FILE]
  python -m scripts.office prey --dir DIR --seed S [--out FILE]
  python -m scripts.office spread --dir DIR --prefix P --hunter NAME --grazer NAME
  python -m scripts.office replay-run --hunter RUN --grazer RUN --out runs/of22replay/NAME [--layout-seed 1000]

`eval` plays each frozen hunter brain against each frozen grazer brain on each layout set, 256 worlds x 500 steps, no
restarts, founder bodies, nothing learning. A hunter given as NAME=random plays at random. Every combination with the same
--seed starts from the same reset keys and random streams, so the same world index has the same layout, founders and
placements for every arm. Layout sets: home is open ground (`office_home`), test the 16 held-out office layouts
(`office` at office_seed 1000), train the 16 training layouts (office_seed 0). Each writes
DIR/<prefix>_<hunter>_<grazer>_<layout>.npz with per-world counts, so scoring can pair arms world by world.

`score` judges the spec's two bars for one seed from the files eval wrote with --prefix s<seed>, hunters home (arm 1),
cont (arm 2), office (arm 3), and grazers own and held. `replay-run` makes a run folder that scripts.lineage can trace:
`office` at one held-out layout, with the grazer's and the hunter's brains in one checkpoint.
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

from agents.checkpoint import load_brains, save_checkpoint
from agents.networks import init_params
from simulations import evo
from simulations.office import TEST_SEED, TRAIN_SEED
from simulations.worlds import get_world, make_net, obs_channels, world_set

EVAL_SEED = 12345
BOOT = 1000
LAYOUTS = ("home", "test", "train")
FIELDS = ("catches", "pred_steps", "pred_starved", "pred_births", "prey_steps", "prey_deaths", "prey_births", "pred_extinct_step")
LOSS_BAR, RECOVERY_BAR = 0.80, 1.25


def layout_config(layout: str, sets=()):
    """sets: Stage 22b's --world-set KEY=VALUE changes, applied to every layout's config."""
    if layout == "home":
        return world_set(get_world("office_home")[1](), sets)
    cfg = get_world("office")[1]()
    return world_set(dataclasses.replace(cfg, office_seed=TEST_SEED if layout == "test" else TRAIN_SEED), sets)


def _net_and_template(cfg):
    net = make_net(evo, cfg, 0)
    return net, init_params(net, jax.random.PRNGKey(0), cfg.view, obs_channels(evo, cfg))


def brains(run: str | Path, template):
    prey, pred, meta = load_brains(Path(run) / "checkpoints" / "latest.msgpack", template)
    return prey, pred, meta.get("update")


def make_rollout(cfg, net, episodes: int, random_hunter: bool):
    """jit: (grazer params, hunter params, key) -> per-world counts, (E,) each."""

    @jax.jit
    def run(prey_p, pred_p, key):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(evo.reset, cfg))(jax.random.split(k_reset, episodes))
        z = jnp.zeros(episodes, jnp.float32)
        acc0 = {f: z for f in FIELDS[:-1]}
        acc0["pred_extinct_step"] = jnp.full(episodes, -1, jnp.int32)

        def body(carry, x):
            state, acc = carry
            k, t = x
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state)
            prey_a = jax.random.categorical(kp, net.apply(prey_p, prey_obs)[0])
            if random_hunter:
                pred_a = jax.random.randint(kq, pred_obs.shape[:-3], 0, evo.NUM_ACTIONS)
            else:
                pred_a = jax.random.categorical(kq, net.apply(pred_p, pred_obs)[0])
            res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            i = res.info
            gone = (acc["pred_extinct_step"] < 0) & (res.state.pred.alive.sum(-1) == 0) & (state.pred.alive.sum(-1) > 0)
            acc = {
                "catches": acc["catches"] + i.catches,
                "pred_steps": acc["pred_steps"] + state.pred.alive.sum(-1),
                "pred_starved": acc["pred_starved"] + i.pred_deaths,
                "pred_births": acc["pred_births"] + i.pred_births,
                "prey_steps": acc["prey_steps"] + state.prey.alive.sum(-1),
                "prey_deaths": acc["prey_deaths"] + i.prey_deaths,
                "prey_births": acc["prey_births"] + i.prey_births,
                "pred_extinct_step": jnp.where(gone, t + 1, acc["pred_extinct_step"]),
            }
            return (res.state, acc), None

        xs = (jax.random.split(k_steps, cfg.max_steps), jnp.arange(cfg.max_steps))
        (_, acc), _ = jax.lax.scan(body, (state, acc0), xs)
        return acc

    return run


def named(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs:
        name, sep, run = p.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=RUN, got {p!r}")
        out.append((name, run))
    return out


def cmd_eval(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hunters, grazers = named(args.hunter), named(args.grazer)
    for layout in args.layout:
        cfg = layout_config(layout, args.world_set)
        net, template = _net_and_template(cfg)
        runs = {}
        for random_hunter in (False, True):
            if any((r == "random") == random_hunter for _, r in hunters):
                runs[random_hunter] = make_rollout(cfg, net, args.episodes, random_hunter)
        key = jax.random.PRNGKey(args.seed)
        for gname, grun in grazers:
            prey_p, _, gupd = brains(grun, template)
            for hname, hrun in hunters:
                rnd = hrun == "random"
                pred_p, hupd = (prey_p, None) if rnd else (lambda b: (b[1], b[2]))(brains(hrun, template))
                acc = {k: np.asarray(v) for k, v in jax.device_get(runs[rnd](prey_p, pred_p, key)).items()}
                path = out / f"{args.prefix}_{hname}_{gname}_{layout}.npz"
                meta = {"hunter": hrun, "hunter_update": hupd, "grazer": grun, "grazer_update": gupd, "layout": layout,
                        "office_seed": int(cfg.office_seed), "office": int(cfg.office), "seed": args.seed, "episodes": args.episodes,
                        "steps": cfg.max_steps, "world_set": args.world_set}
                np.savez(path, meta=json.dumps(meta), **acc)
                rate = acc["catches"].sum() / max(acc["pred_steps"].sum(), 1)
                ext = (acc["pred_extinct_step"] >= 0).mean()
                print(f"{path.name}: catches/hunter-step {rate:.4f}  catches {acc['catches'].sum():.0f}  hunter-steps "
                      f"{acc['pred_steps'].sum():.0f}  starved {acc['pred_starved'].sum():.0f}  extinct {ext:.2f}", flush=True)
    return 0


# ---------------------------------------------------------------- scoring


def load(d: Path, prefix, hunter, grazer, layout) -> dict:
    z = np.load(d / f"{prefix}_{hunter}_{grazer}_{layout}.npz")
    return {k: z[k] for k in FIELDS}


def rate(a, idx=None):
    if idx is None:
        return a["catches"].sum() / max(a["pred_steps"].sum(), 1)
    return a["catches"][idx].sum(-1) / np.maximum(a["pred_steps"][idx].sum(-1), 1)


def ratio(num, den, boot_idx):
    point = rate(num) / max(rate(den), 1e-12)
    b = rate(num, boot_idx) / np.maximum(rate(den, boot_idx), 1e-12)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"ratio": float(point), "lo": float(lo), "hi": float(hi)}


def verdict(r: dict, bar: float, below: bool, spread: float) -> str:
    """below: the effect is the ratio falling to bar or under (loss); otherwise rising to bar or over (recovery)."""
    if below:
        beyond, clear = r["ratio"] <= bar, r["hi"] < 1
        miss = (r["ratio"] - bar) / bar
    else:
        beyond, clear = r["ratio"] >= bar, r["lo"] > 1
        miss = (bar - r["ratio"]) / bar
    if beyond and clear:
        return "pass"
    if (beyond and not clear) or miss <= 2 * spread:
        return "no result"
    return "fail"


def summary(a) -> dict:
    n = len(a["catches"])
    ext = a["pred_extinct_step"] >= 0
    return {"catches_per_hunter_step": float(rate(a)), "catches": float(a["catches"].sum()), "hunter_steps": float(a["pred_steps"].sum()),
            "starved": float(a["pred_starved"].sum()), "starved_per_hunter_step": float(a["pred_starved"].sum() / max(a["pred_steps"].sum(), 1)),
            "hunter_births": float(a["pred_births"].sum()), "extinct_share": float(ext.mean()),
            "extinct_median_step": float(np.median(a["pred_extinct_step"][ext])) if ext.any() else None,
            "grazer_steps": float(a["prey_steps"].sum()), "grazer_deaths": float(a["prey_deaths"].sum()),
            "grazer_births": float(a["prey_births"].sum()), "worlds": n}


def cmd_score(args) -> int:
    d, s = Path(args.dir), args.seed
    prefix = f"s{s}"
    rng = np.random.default_rng(1000 + s)
    out = {"seed": s, "spread": args.spread, "bars": {}, "reported": {}, "tables": {}}
    for g in ("own", "held"):
        A = {(h, lay): load(d, prefix, h, g, lay) for h in ("home", "cont", "office") for lay in LAYOUTS}
        n = len(A[("home", "home")]["catches"])
        idx = rng.integers(0, n, (BOOT, n))  # the same resampled worlds for every arm and layout compared
        loss1 = ratio(A[("home", "test")], A[("home", "home")], idx)
        rec = ratio(A[("office", "test")], A[("cont", "test")], idx)
        loss2 = ratio(A[("cont", "test")], A[("cont", "home")], idx)
        keep_cont = ratio(A[("office", "home")], A[("cont", "home")], idx)
        keep_home = ratio(A[("office", "home")], A[("home", "home")], idx)
        seen = ratio(A[("office", "train")], A[("office", "test")], idx)
        c_home, c_off, o_off = (float(rate(A[k])) for k in (("cont", "home"), ("cont", "test"), ("office", "test")))
        share = (o_off - c_off) / (c_home - c_off) if c_home > c_off else None
        bars = {"1_transfer_loss": {**loss1, "loss": 1 - loss1["ratio"], "verdict": verdict(loss1, LOSS_BAR, True, args.spread)},
                "2_recovery": {**rec, "verdict": verdict(rec, RECOVERY_BAR, False, args.spread)}}
        out["bars"][g] = bars
        out["reported"][g] = {"arm2_office_over_home": loss2, "home_retention_vs_cont": keep_cont, "home_retention_vs_home": keep_home,
                              "arm3_seen_over_heldout": seen, "recovery_share": share}
        out["tables"][g] = {f"{h}/{lay}": summary(A[(h, lay)]) for (h, lay) in A}
        for lay in ("home", "test"):
            f = d / f"{prefix}_random_{g}_{lay}.npz"
            if f.exists():
                out["tables"][g][f"random/{lay}"] = summary(load(d, prefix, "random", g, lay))
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    for g in ("own", "held"):
        b, r = out["bars"][g], out["reported"][g]
        print(f"seed {s} grazer {g}: bar 1 office/home {b['1_transfer_loss']['ratio']:.3f} [{b['1_transfer_loss']['lo']:.3f}, "
              f"{b['1_transfer_loss']['hi']:.3f}] loss {b['1_transfer_loss']['loss']:.1%} -> {b['1_transfer_loss']['verdict']}; "
              f"bar 2 office/cont {b['2_recovery']['ratio']:.3f} [{b['2_recovery']['lo']:.3f}, {b['2_recovery']['hi']:.3f}] -> {b['2_recovery']['verdict']}")
        print(f"   arm 2 office/home {r['arm2_office_over_home']['ratio']:.3f}; retention vs cont {r['home_retention_vs_cont']['ratio']:.3f}, "
              f"vs home {r['home_retention_vs_home']['ratio']:.3f}; seen/held-out {r['arm3_seen_over_heldout']['ratio']:.3f}; "
              f"recovery share {r['recovery_share']}")
        for k, v in out["tables"][g].items():
            print(f"   {k:14s} rate {v['catches_per_hunter_step']:.4f} catches {v['catches']:7.0f} hunter-steps {v['hunter_steps']:8.0f} "
                  f"starved/step {v['starved_per_hunter_step']:.4f} extinct {v['extinct_share']:.2f} (median {v['extinct_median_step']})")
    return 0


def dense_rate(a, idx=None, steps=500):
    """Stage 22b: catches per living hunter-step per mean living grazer (catches / (hunter-steps x grazer-steps / world-steps))."""
    if idx is None:
        return a["catches"].sum() / max(a["pred_steps"].sum() * a["prey_steps"].sum() / (len(a["catches"]) * steps), 1e-12)
    n = idx.shape[-1]
    return a["catches"][idx].sum(-1) / np.maximum(a["pred_steps"][idx].sum(-1) * a["prey_steps"][idx].sum(-1) / (n * steps), 1e-12)


def cmd_prey(args) -> int:
    """Stage 22b, reported only: living grazer-steps, grazer deaths against catches, and the density-adjusted office/home ratio."""
    d, s = Path(args.dir), args.seed
    rng = np.random.default_rng(2000 + s)
    out = {}
    for g in ("own", "held"):
        A = {(h, lay): load(d, f"s{s}", h, g, lay) for h in ("home", "cont", "office") for lay in ("home", "test")}
        n = len(A[("home", "home")]["catches"])
        idx = rng.integers(0, n, (BOOT, n))
        row = {}
        for h in ("home", "cont", "office"):
            hm, of = A[(h, "home")], A[(h, "test")]
            gs = lambda a, i=None: a["prey_steps"].sum() if i is None else a["prey_steps"][i].sum(-1)
            gb = gs(of, idx) / np.maximum(gs(hm, idx), 1)
            db = dense_rate(of, idx) / np.maximum(dense_rate(hm, idx), 1e-12)
            row[h] = {"grazer_steps_home": float(gs(hm)), "grazer_steps_office": float(gs(of)),
                      "grazer_steps_office_over_home": {"ratio": float(gs(of) / gs(hm)), "lo": float(np.percentile(gb, 2.5)), "hi": float(np.percentile(gb, 97.5))},
                      "grazer_deaths_minus_catches": {"home": float(hm["prey_deaths"].sum() - hm["catches"].sum()),
                                                      "office": float(of["prey_deaths"].sum() - of["catches"].sum())},
                      "dense_office_over_home": {"ratio": float(dense_rate(of) / dense_rate(hm)), "lo": float(np.percentile(db, 2.5)), "hi": float(np.percentile(db, 97.5))}}
            r = row[h]
            print(f"seed {s} grazer {g} {h:6s}: grazer-steps home {r['grazer_steps_home']/1e6:.2f}M office {r['grazer_steps_office']/1e6:.2f}M "
                  f"(ratio {r['grazer_steps_office_over_home']['ratio']:.3f}); starved grazers home {r['grazer_deaths_minus_catches']['home']:.0f} "
                  f"office {r['grazer_deaths_minus_catches']['office']:.0f}; density-adjusted office/home {r['dense_office_over_home']['ratio']:.3f} "
                  f"[{r['dense_office_over_home']['lo']:.3f}, {r['dense_office_over_home']['hi']:.3f}]")
        out[g] = row
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_spread(args) -> int:
    """Relative standard deviation of the catch rate over evaluation seeds: files <prefix><seed>_<hunter>_<grazer>_<layout>.npz."""
    d = Path(args.dir)
    out = {}
    for lay in ("home", "test"):
        rates = []
        for f in sorted(d.glob(f"{args.prefix}*_{args.hunter}_{args.grazer}_{lay}.npz")):
            rates.append(rate({k: np.load(f)[k] for k in FIELDS}))
        rates = np.array(rates)
        out[lay] = {"rates": rates.tolist(), "rel_sd": float(rates.std(ddof=1) / rates.mean())}
    out["ratio_spread"] = float(np.sqrt(2) * max(out["home"]["rel_sd"], out["test"]["rel_sd"]))
    print(json.dumps(out, indent=1))
    return 0


def cmd_replay_run(args) -> int:
    """A run folder for scripts.lineage: the office at one held-out layout, the grazer's and the hunter's brains together."""
    cfg = dataclasses.replace(get_world("office")[1](), office=1, office_seed=args.layout_seed)
    net, template = _net_and_template(cfg)
    prey_p, _, gupd = brains(args.grazer, template)
    _, pred_p, hupd = brains(args.hunter, template)
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "checkpoints" / "latest.msgpack", prey_p, pred_p,
                    {"update": hupd, "grazer": str(args.grazer), "grazer_update": gupd, "hunter": str(args.hunter)})
    config = {"world_name": "office", "world": dataclasses.asdict(cfg), "train": {"memory": 0},
              "args": {"world_set": [f"office=1", f"office_seed={args.layout_seed}"], "grazer": str(args.grazer), "hunter": str(args.hunter)}}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"{out}: grazer {args.grazer} (update {gupd}), hunter {args.hunter} (update {hupd}), layout {args.layout_seed}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--hunter", action="append", required=True, help="NAME=RUN, or NAME=random")
    e.add_argument("--grazer", action="append", required=True, help="NAME=RUN")
    e.add_argument("--layout", action="append", required=True, choices=LAYOUTS)
    e.add_argument("--out", required=True)
    e.add_argument("--prefix", default="x")
    e.add_argument("--seed", type=int, default=EVAL_SEED)
    e.add_argument("--episodes", type=int, default=256)
    e.add_argument("--world-set", action="append", default=[], metavar="KEY=VALUE",
                   help="Stage 22b: change a field of every layout's world config, repeatable (prey_no_upkeep=true)")
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--spread", type=float, default=0.0, help="the evaluation spread of a ratio, from the pilot")
    s.add_argument("--out")
    sp = sub.add_parser("spread")
    sp.add_argument("--dir", required=True)
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--hunter", required=True)
    sp.add_argument("--grazer", required=True)
    pr = sub.add_parser("prey")
    pr.add_argument("--dir", required=True)
    pr.add_argument("--seed", type=int, required=True)
    pr.add_argument("--out")
    r = sub.add_parser("replay-run")
    r.add_argument("--hunter", required=True)
    r.add_argument("--grazer", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--layout-seed", type=int, default=TEST_SEED)
    args = p.parse_args(argv)
    return {"eval": cmd_eval, "score": cmd_score, "spread": cmd_spread, "prey": cmd_prey, "replay-run": cmd_replay_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
