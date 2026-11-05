"""Stage 23 dynasties: who works and who free-rides. Spec: docs/specs/2026-11-05-pack-cheaters-design.md

  python -m scripts.pack_lineage --run-dir runs/pp-pk23-E-s0 [--run-dir ...] [--out-dir runs/lineage/pk23] [--episodes 16]
         [--steps 1000] [--seed 12345] [--replays 4]

Replays episodes of an effort-world run (stage 23) with the run's one hunter brain and grazer brain, starting from the run's
last gene pool, and traces every animal with scripts/lineage.py's uids and line tables. lineage.py itself is unchanged; this
adds, for every hunter: its effort (gene 5, labelled "effort", never "diet"), steps assigned to a grazer, hunts (groups of
at least pack_size), catches shared, summed contribution share (e / S), meal handed out and kept after the energy cap, its
partners, and its free-ride surplus: kept meal minus contribution share x the group's kept meal. In each catch the positive
surpluses are paid for by the negative ones in proportion, which gives a line-to-line table of who fed whom.

Writes <out-dir>/<run>/pack_lineage.json (every episode), replays in the 3-D viewer's format (episode3d_ep<k>.json, the
best again as episode3d.json) for the episodes with the largest worker -> cheater transfer, and <out-dir>/pk23_lines.md
ranking lines by free-riding, by working, worker -> cheater pairs by energy, and by lineage.py's demographic drama.
A "cheater" is a hunter whose share of a kill's kept meal is above its share of the effort, nothing more; lines are
within one episode, not training-wide dynasties, and mutating descendants need not share their founder's effort.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts import lineage as L
from simulations import evo

PK_GENES = ("speed", "sight", "breed", "size", "brain", "effort", "camo")


class EffortTracer(L.Tracer):
    """lineage.Tracer, with gene 5 labelled effort."""

    def _new(self, kind, parent, t, slot, genes=None, gen=None) -> int:
        uid = super()._new(kind, parent, t, slot, None, gen)
        if genes is not None:
            self.animals[uid]["genes"] = {g: round(float(v), 3) for g, v in zip(PK_GENES, genes)}
        return uid


def _blank():
    return {"alive_steps": 0, "assigned_steps": 0, "hunts": 0, "catches": 0, "contrib_sum": 0.0, "meal_frac_sum": 0.0,
            "meal": 0.0, "kept": 0.0, "surplus": 0.0, "given": 0.0, "received": 0.0, "free_rides": 0, "partners": defaultdict(int)}


def simulate(run_dir: Path, episodes: int, steps: int, seed: int, chunk: int = 50):
    from agents.checkpoint import load_brains
    from agents.networks import init_params
    from scripts.eval import load_world_config
    from scripts.league import evolved_pool
    from simulations.worlds import make_net, obs_channels

    config = json.loads((run_dir / "config.json").read_text())
    world, cfg = load_world_config(config)
    if not getattr(cfg, "effort", ""):
        raise SystemExit(f"{run_dir}: not an effort world (stage 23)")
    net = make_net(evo, cfg, 0)
    template = init_params(net, jax.random.PRNGKey(0), cfg.view, obs_channels(evo, cfg))
    prey_p, pred_p, meta_ck = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    pool, bodies = evolved_pool(run_dir)
    E = episodes
    key, k_reset = jax.random.split(jax.random.PRNGKey(seed))
    state = jax.vmap(lambda k: evo.reset(cfg, k, pool))(jax.random.split(k_reset, E))
    po, qo = jax.vmap(partial(evo.observe, cfg))(state)

    @jax.jit
    def run_chunk(c):
        def body(c, _):
            s, po, qo, key = c
            key, kp, kq, ks = jax.random.split(key, 4)
            pa = jax.random.categorical(kp, net.apply(prey_p, po)[0])
            qa = jax.random.categorical(kq, net.apply(pred_p, qo)[0])
            res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(ks, E), s, pa, qa)
            po2, qo2 = jax.vmap(partial(evo.observe, cfg))(res.state)
            n, i, x = res.state, res.info, res.info.effort
            out = {"prey_parent": i.prey_parent, "pred_parent": i.pred_parent, "prey_alive": n.prey.alive, "pred_alive": n.pred.alive,
                   "prey_pos": n.prey.pos, "pred_pos": n.pred.pos, "prey_energy": n.prey.energy, "pred_energy": n.pred.energy,
                   "prey_genes": n.prey.genes, "pred_genes": n.pred.genes, "prey_gen": n.prey.gen, "pred_gen": n.pred.gen,
                   "plant": n.plant, "catches": i.catches, "assigned": x.assigned, "group": x.group, "contrib": x.contrib,
                   "meal": x.meal, "kept": x.kept, "caught": x.caught, "was_pred_alive": s.pred.alive}
            return (n, po2, qo2, key), out
        return jax.lax.scan(body, c, None, length=chunk)

    s0 = jax.device_get(state)
    tracers, frames = [EffortTracer() for _ in range(E)], [[] for _ in range(E)]
    stats = [defaultdict(_blank) for _ in range(E)]
    transfers = [defaultdict(float) for _ in range(E)]  # (giver uid, receiver uid) -> energy
    effort_series = [[] for _ in range(E)]  # [t, uid-of-line-root -> [mean, n]] filled after, from births

    def rows(tr, kind, pos, energy):
        slots = tr.slots[kind]
        return [[int(slots[i]), int(pos[i, 0]), int(pos[i, 1]), round(float(energy[i]), 1)] for i in np.nonzero(slots >= 0)[0]]

    for e in range(E):
        for kind in L.SPECIES:
            sp = getattr(s0, kind)
            tracers[e].start(kind, sp.alive[e], sp.genes[e], sp.gen[e])
        frames[e].append({"prey": rows(tracers[e], "prey", s0.prey.pos[e], s0.prey.energy[e]),
                          "pred": rows(tracers[e], "pred", s0.pred.pos[e], s0.pred.energy[e]), "plant": L._bits(s0.plant[e]),
                          "tough": L._bytes(np.asarray(s0.plant_genes[e])[..., 1]), "ate": [], "hunts": []})
    was = {"prey": np.asarray(s0.prey.alive), "pred": np.asarray(s0.pred.alive)}
    carry, t, mismatched = (state, po, qo, key), 0, 0
    while t < steps:
        carry, out = run_chunk(carry)
        out = jax.device_get(out)
        for j in range(chunk):
            t += 1
            if t > steps:
                break
            for e in range(E):
                tr, st = tracers[e], stats[e]
                g = lambda k: out[k][j, e]
                pred_uid, prey_uid = tr.slots["pred"].copy(), tr.slots["prey"].copy()
                caught, assigned, group = g("caught").astype(bool), g("assigned"), g("group")
                eaten = np.zeros(prey_uid.shape[0], bool)
                eaten[assigned[caught]] = True
                if int(eaten.sum()) != int(g("catches")):
                    mismatched += 1
                hunts = []
                for i in np.nonzero(pred_uid >= 0)[0]:
                    a = st[int(pred_uid[i])]
                    a["alive_steps"] += 1
                    if assigned[i] >= 0:
                        a["assigned_steps"] += 1
                        if group[i] >= cfg.pack_size:
                            a["hunts"] += 1
                for prey_slot in np.unique(assigned[(assigned >= 0) & (group >= cfg.pack_size)]):
                    members = np.nonzero((assigned == prey_slot) & (pred_uid >= 0))[0]
                    got = bool(caught[members].any())
                    hunts.append([[int(pred_uid[m]) for m in members], int(got), int(prey_uid[prey_slot])])
                    if not got:
                        continue
                    kept = g("kept")[members].astype(float)
                    total = kept.sum()
                    contrib = g("contrib")[members].astype(float)
                    surplus = kept - contrib * total
                    pos_sum = surplus[surplus > 0].sum()
                    for k, m in enumerate(members):
                        a = st[int(pred_uid[m])]
                        a["catches"] += 1
                        a["contrib_sum"] += contrib[k]
                        a["meal_frac_sum"] += kept[k] / total if total > 0 else 0.0
                        a["meal"] += float(g("meal")[m])
                        a["kept"] += kept[k]
                        a["surplus"] += surplus[k]
                        if surplus[k] > 1e-6:
                            a["received"] += surplus[k]
                            a["free_rides"] += 1
                        elif surplus[k] < -1e-6:
                            a["given"] += -surplus[k]
                        for m2 in members:
                            if m2 != m:
                                a["partners"][int(pred_uid[m2])] += 1
                    if pos_sum > 0:
                        for k, m in enumerate(members):
                            if surplus[k] < -1e-6:
                                for k2, m2 in enumerate(members):
                                    if surplus[k2] > 1e-6:
                                        transfers[e][(int(pred_uid[m]), int(pred_uid[m2]))] += -surplus[k] * surplus[k2] / pos_sum
                tr.step(t, "prey", g("prey_parent"), was["prey"][e], g("prey_alive"), eaten, g("prey_genes"), g("prey_gen"))
                tr.step(t, "pred", g("pred_parent"), was["pred"][e], g("pred_alive"), None, g("pred_genes"), g("pred_gen"))
                ate = [int(pred_uid[i]) for i in np.nonzero(caught)[0] if pred_uid[i] >= 0]
                frames[e].append({"prey": rows(tr, "prey", g("prey_pos"), g("prey_energy")),
                                  "pred": rows(tr, "pred", g("pred_pos"), g("pred_energy")), "plant": L._bits(g("plant")),
                                  "ate": ate, "hunts": hunts})
            was = {"prey": np.asarray(out["prey_alive"][j]), "pred": np.asarray(out["pred_alive"][j])}
    for tr in tracers:
        for a in tr.animals:
            if a["species"] == "pred" and a["cause"] == "died":
                a["cause"] = "starved"
    final = jax.device_get(carry[0])
    meta = {"world": config.get("world_name"), "run_kind": "shared", "mode": "darwin", "frozen": False, "anchor_frac": 0.0,
            "bodies": bodies, "steps": steps, "episodes": E, "seed": seed, "brain_update": meta_ck.get("update"),
            "cause_of_death": "exact" if not mismatched else "inexact", "cause_mismatched_steps": mismatched,
            "size": cfg.size, "terrain": L._bytes(np.asarray(final.terrain[0]) / 2.0), "flank": bool(cfg.flank),
            "pack_size": int(cfg.pack_size), "stealth": int(cfg.stealth_radius), "climate": cfg.climate,
            "winter_len": int(cfg.winter_len), "effort_rule": cfg.effort, "effort_cost": cfg.effort_cost, "effort_k": cfg.effort_k,
            "world_set": []}
    return tracers, frames, stats, transfers, meta


def line_effort(animals, root, T, every):
    """{founder: [[t, mean effort of living members, n], ...]}"""
    out = defaultdict(list)
    hunters = [a for a in animals if a["species"] == "pred"]
    for t in sorted(set(range(0, T + 1, every)) | {T}):
        by = defaultdict(list)
        for a in hunters:
            if a["born"] <= t and (a["died"] is None or a["died"] > t):
                by[root[a["uid"]]].append(a["genes"]["effort"])
        for f, v in by.items():
            out[f].append([t, round(float(np.mean(v)), 3), len(v)])
    return out


def analyse_episode(e, tr, st, tf, steps, every):
    rows = L.analyse(tr.animals, steps, every, e)
    root = L._roots(tr.animals)
    eff = line_effort(tr.animals, root, steps, every)
    by_uid = {a["uid"]: a for a in tr.animals}
    hunters = []
    for a in tr.animals:
        if a["species"] != "pred":
            continue
        s = st.get(a["uid"], _blank())
        c = s["catches"]
        hunters.append({"uid": a["uid"], "parent_uid": a["parent_uid"], "line": root[a["uid"]], "born": a["born"], "died": a["died"],
                        "cause": a["cause"], "effort": a["genes"]["effort"], "gen": a.get("gen", 0),
                        "alive_steps": s["alive_steps"], "assigned_steps": s["assigned_steps"], "hunts": s["hunts"], "catches": c,
                        "contrib_share": round(s["contrib_sum"] / c, 3) if c else None,
                        "meal_share": round(s["meal_frac_sum"] / c, 3) if c else None,
                        "meal": round(s["meal"], 2), "kept": round(s["kept"], 2), "surplus": round(s["surplus"], 2),
                        "received": round(s["received"], 2), "given": round(s["given"], 2), "free_rides": s["free_rides"],
                        "free_rider": bool(c and s["meal_frac_sum"] > s["contrib_sum"] + 1e-6),
                        "partners": {str(k): v for k, v in sorted(s["partners"].items(), key=lambda kv: -kv[1])[:8]},
                        "children": sum(1 for b in tr.animals if b["parent_uid"] == a["uid"])})
    line_tf = defaultdict(float)
    for (gv, rc), v in tf.items():
        line_tf[(root[gv], root[rc])] += v
    for r in rows:
        if r["species"] != "pred":
            continue
        mem = [h for h in hunters if h["line"] == r["founder"]]
        r["effort_founder"] = r["genes"]["effort"] if r.get("genes") else None
        r["effort_series"] = eff.get(r["founder"], [])
        r["received"] = round(sum(h["received"] for h in mem), 2)
        r["given"] = round(sum(h["given"] for h in mem), 2)
        r["net_surplus"] = round(r["received"] - r["given"], 2)
        r["catches"] = sum(h["catches"] for h in mem)
        r["mean_effort_members"] = round(float(np.mean([h["effort"] for h in mem])), 3) if mem else None
        r["role"] = "cheater" if r["net_surplus"] > 0 else ("worker" if r["net_surplus"] < 0 else "none")
    names = {r["founder"]: r["name"] for r in rows if r["species"] == "pred"}
    # net flow between two lines: what a gave b minus what b gave a, kept in the direction it ran
    net = {}
    for (g, rc), v in line_tf.items():
        if g != rc and (rc, g) not in net:
            net[(g, rc)] = v - line_tf.get((rc, g), 0.0)
    pairs = sorted(([g, rc, round(v, 2)] if v >= 0 else [rc, g, round(-v, 2)] for (g, rc), v in net.items()), key=lambda x: -x[2])
    within = round(sum(v for (g, rc), v in line_tf.items() if g == rc), 2)
    return {"episode": e, "species_extinct": L.species_end(tr.animals, steps), "founders": rows, "hunters": hunters,
            "line_transfers": [{"from": g, "from_name": names.get(g), "to": rc, "to_name": names.get(rc), "energy": v} for g, rc, v in pairs],
            "within_line_transfer": within,
            "animals": [{k: a[k] for k in ("uid", "parent_uid", "species", "born", "died", "cause")} for a in tr.animals]}


def replay(run_name, ep, tr, frames, meta):
    out = L.replay_json(run_name, ep["episode"], tr, frames, meta, ep["founders"])
    hs = {h["uid"]: h for h in ep["hunters"]}
    for uid, a in out["animals"].items():
        a.pop("diet", None)
        a["effort"] = (tr.animals[uid].get("genes") or {}).get("effort", 0.0) if a["kind"] == "pred" else None
        h = hs.get(uid)
        if h:
            a.update({k: h[k] for k in ("hunts", "catches", "contrib_share", "meal_share", "kept", "surplus", "received", "given",
                                         "free_rider", "partners")})
    for r in ep["founders"]:
        if r["species"] == "pred" and r["founder"] in out["lines"]:
            out["lines"][r["founder"]].update({k: r[k] for k in ("role", "net_surplus", "received", "given", "effort_founder",
                                                                  "mean_effort_members", "effort_series", "catches")})
    out["transfers"] = ep["line_transfers"][:10]
    out["effort"] = {"rule": meta["effort_rule"], "cost": meta["effort_cost"], "k": meta["effort_k"], "gene": "effort (slot 5)",
                     "hunts": "each frame's hunts: [hunter uids, caught 0/1, grazer uid] for groups of at least pack_size"}
    return out


def trace_run(run_dir: Path, out_dir: Path, episodes, steps, seed, replays, every, origin=None):
    tracers, frames, stats, transfers, meta = simulate(run_dir, episodes, steps, seed)
    eps = [analyse_episode(e, tracers[e], stats[e], transfers[e], steps, every) for e in range(episodes)]
    score = lambda ep: (ep["line_transfers"][0]["energy"] if ep["line_transfers"] else 0.0)
    picked = sorted(range(episodes), key=lambda e: -score(eps[e]))[:replays]
    target = out_dir / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(picked):
        text = json.dumps(replay(run_dir.name, eps[e], tracers[e], frames[e], meta), separators=(",", ":"))
        eps[e]["replay"] = f"episode3d_ep{e}.json"
        (target / eps[e]["replay"]).write_text(text)
        if i == 0:
            (target / "episode3d.json").write_text(text)
    out = {"run": run_dir.name, "run_path": origin or str(run_dir), **{k: v for k, v in meta.items() if k != "terrain"}, "episodes_traced": eps}
    (target / "pack_lineage.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{target}: {episodes} episodes x {steps} steps, cause of death {meta['cause_of_death']}, replays "
          f"{[eps[e]['replay'] for e in picked]}", flush=True)
    return out


def rank(out_dir: Path, top: int = 10) -> Path:
    lines, pairs = [], []
    for f in sorted(out_dir.glob("*/pack_lineage.json")):
        d = json.loads(f.read_text())
        for ep in d["episodes_traced"]:
            fr = {r["founder"]: r for r in ep["founders"] if r["species"] == "pred"}
            for r in fr.values():
                lines.append((d, ep, r))
            for p in ep["line_transfers"]:
                if p["from"] in fr and p["to"] in fr:
                    pairs.append((d, ep, fr[p["from"]], fr[p["to"]], p["energy"]))
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    rep = lambda d, ep: f"`{d['run']}/{ep['replay']}`" if ep.get("replay") else "no replay"
    out = ["# Stage 23 dynasties: workers and cheaters", "",
           "A line is a founder hunter alive at step 0 and its descendants, within one episode. A hunter **free-rides** on a kill",
           "when its share of the kill's kept meal is above its share of the effort (e / S). Its **surplus** is kept meal minus",
           "effort share x the group's kept meal: positive is taken from partners, negative is given. In each kill, positive",
           "surpluses are paid for by negative ones in proportion, which gives the worker -> cheater transfers (net, per pair of lines).",
           "Effort is gene 5 of these runs, labelled effort; mutating descendants need not share their founder's effort.",
           f"Rebuild with `python -m scripts.pack_lineage --rank-only --out-dir {out_dir}`.", "",
           "## Worker -> cheater pairs (net energy transferred between two lines: what the worker line gave minus what it got back)", "",
           "| # | worker line (effort) | cheater line (effort) | energy | run | episode | replay |", "|---|---|---|---|---|---|---|"]
    for i, (d, ep, g, r, v) in enumerate(sorted(pairs, key=lambda x: -x[4])[:top], 1):
        out.append(f"| {i} | {g['name']} (#{g['founder']}, {fmt(g['mean_effort_members'])}) | {r['name']} (#{r['founder']}, "
                   f"{fmt(r['mean_effort_members'])}) | {v:.1f} | {d['run']} | {ep['episode']} | {rep(d, ep)} |")
    for title, key in (("Cheater lines (surplus received)", "received"), ("Worker lines (surplus given)", "given")):
        out += ["", f"## {title}", "", "| # | line | founder effort | members' mean effort | received | given | catches | descendants | peak | run | episode |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, (d, ep, r) in enumerate(sorted(lines, key=lambda x: -x[2][key])[:top], 1):
            out.append(f"| {i} | {r['name']} (#{r['founder']}) | {fmt(r['effort_founder'])} | {fmt(r['mean_effort_members'])} | "
                       f"{r['received']:.1f} | {r['given']:.1f} | {r['catches']} | {r['descendants']} | {r['peak_pct']:.0f}% @ {r['peak_step']} | "
                       f"{d['run']} | {ep['episode']} |")
    out += ["", "## Hunter lines by demographic drama (lineage.py's score)", "",
            "| # | line | role | drama | arc | net surplus | founder effort | descendants | line extinct | run | episode |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (d, ep, r) in enumerate(sorted(lines, key=lambda x: -x[2]["drama"])[:top], 1):
        out.append(f"| {i} | {r['name']} (#{r['founder']}) | {r['role']} | {r['drama']:.1f} | {r['arc']} | {r['net_surplus']:.1f} | "
                   f"{fmt(r['effort_founder'])} | {r['descendants']} | {r['extinct_step'] if r['extinct_step'] is not None else '-'} | {d['run']} | {ep['episode']} |")
    out += ["", "## Stories (best pair per replayed episode)", ""]
    seen = set()
    for d, ep, g, r, v in sorted(pairs, key=lambda x: -x[4]):
        k = (d["run"], ep["episode"])
        if k in seen or not ep.get("replay"):
            continue
        seen.add(k)
        out += [f"### {g['name']} worked, {r['name']} free-rode: {d['run']}, episode {ep['episode']}", "",
                f"- {g['name']} line (founder #{g['founder']}, effort {fmt(g['effort_founder'])}, members' mean {fmt(g['mean_effort_members'])}): "
                f"{g['catches']} shared kills, gave {g['given']:.1f}, received {g['received']:.1f}; {g['descendants']} descendants, "
                f"peak {g['peak_pct']:.0f}% at step {g['peak_step']}" + (f", extinct at step {g['extinct_step']}" if g["extinct_step"] is not None else ""),
                f"- {r['name']} line (founder #{r['founder']}, effort {fmt(r['effort_founder'])}, members' mean {fmt(r['mean_effort_members'])}): "
                f"{r['catches']} shared kills, received {r['received']:.1f}, gave {r['given']:.1f}; {r['descendants']} descendants, "
                f"peak {r['peak_pct']:.0f}% at step {r['peak_step']}" + (f", extinct at step {r['extinct_step']}" if r["extinct_step"] is not None else ""),
                f"- {g['name']} -> {r['name']}: {v:.1f} energy over {d['steps']} steps. Replay {rep(d, ep)}; highlight `line == {g['founder']}` and `line == {r['founder']}`.",
                f"- {r['name']} moments: " + "; ".join(f"{m['text']} (step {m['step']})" for m in r["moments"][:6]), ""]
    path = out_dir / "pk23_lines.md"
    path.write_text("\n".join(out))
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", default=[])
    p.add_argument("--out-dir", default="runs/lineage/pk23")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--replays", type=int, default=4)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--origin", action="append", default=[])
    p.add_argument("--rank-only", action="store_true")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    if not args.rank_only:
        if not args.run_dir:
            p.error("--run-dir is required unless --rank-only")
        for i, r in enumerate(args.run_dir):
            trace_run(Path(r), out_dir, args.episodes, args.steps, args.seed, args.replays, args.every,
                      args.origin[i] if i < len(args.origin) else None)
    print(f"ranked: {rank(out_dir, args.top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
