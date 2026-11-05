"""Stage 21b studio clip: one test episode of oracle-call grazers, with lineage, every grazer's call bit, and the same
episode with grazer reception muted. Spec docs/specs/2026-11-05-oracle-call-design.md

  python -m scripts.oracle_clip --grazers runs/pp-or-O-s0 --hunters runs/pp-st-0 --out runs/lineage/or21b [--seeds 0-15]

Each seed is one world: evo.reset from PRNGKey(seed)'s first split, then one key per step, as `scripts.alarm clip` does.
The real and muted versions share that start and that stream; they differ only in what grazers hear (muted: nothing).
Hunters hear the real calls in both (pp-st-0's hunters have zero weights on the channel). The simulation makes every
call: a living grazer's bit is on exactly when a living hunter is within 1 cell of it before it moves.

The seed kept is the one with the most strong moments in the first 150 steps (see `moments`). Writes, in --out:
  episode3d.json         the real episode in the 3-D viewer's format, lineage fields as scripts.lineage writes them,
                         plus steps[t].prey_call
  episode3d_muted.json   the same world with grazer reception muted
  moments.json           candidate shots, ranked, at most 30
  search.json            every searched seed's counts
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts import lineage
from scripts.alarm import brain, eval_world
from simulations import evo
from simulations import predprey as pp
from simulations.predprey import NUM_ACTIONS
from simulations.worlds import make_net

HORIZON = 50  # steps a listener must live after hearing, to count as survived
EARLY = 150  # moments are calls made before this step
TOP = 30


# ---------------------------------------------------------------- playing

def make_play(cfg, net, prey_p, pred_p, steps: int):
    """jitted play(seeds (S,), prey_mode): S worlds for `steps` steps. prey_mode 0 = real reception, 1 = muted.
    Returns (start state, per-step outputs stacked (steps, S, ...))."""
    shim, exact = lineage._extended(evo)

    def one(seed, prey_mode):
        k_reset, key = jax.random.split(jax.random.PRNGKey(seed))
        s0 = evo.reset(cfg, k_reset)

        def body(state, k):
            kp, kq, kw = jax.random.split(k, 3)
            real = state.prey_signal
            heard = jnp.where(prey_mode == 1, jnp.zeros_like(real), real)
            prey_obs, pred_obs = evo.observe(cfg, state, heard, real)
            pa = jax.random.categorical(kp, net.apply(prey_p, prey_obs)[0])
            qa = jax.random.categorical(kq, net.apply(pred_p, pred_obs)[0])
            call = evo.hunter_adjacent(cfg, state.prey, state.pred) & state.prey.alive  # the oracle's bit, as advance sets it
            res = shim.advance(cfg, kw, state, pa, qa)
            return res.state, {"prey_act": pa, "call": call, "info": res.info}

        _, out = jax.lax.scan(body, s0, jax.random.split(key, steps))
        return s0, out

    return jax.jit(jax.vmap(one, in_axes=(0, None))), exact


def trace(s0, out, e: int, steps: int, exact: bool):
    """World e of a play: (Tracer, viewer frames, numpy frames for the moment finder). Frame t is the state before step
    t's actions; frames has steps + 1 entries."""
    s0, out = jax.tree.map(lambda x: np.asarray(x)[e], (s0, out))  # vmapped over worlds: axis 0 is the world
    info = out["info"]
    tr = lineage.Tracer()
    for kind in lineage.SPECIES:
        sp = getattr(s0, kind)
        tr.start(kind, sp.alive, sp.genes, sp.gen)

    def rows(kind, pos, energy):
        slots = tr.slots[kind]
        return [[int(slots[i]), int(pos[i, 0]), int(pos[i, 1]), round(float(energy[i]), 1)] for i in np.nonzero(slots >= 0)[0]]

    def sim_frame(prey_pos, prey_alive, pred_pos, pred_alive):
        return {"uid": tr.slots["prey"].copy(), "pos": np.asarray(prey_pos), "alive": np.asarray(prey_alive, bool),
                "pred_pos": np.asarray(pred_pos), "pred_alive": np.asarray(pred_alive, bool)}

    view = [{"prey": rows("prey", s0.prey.pos, s0.prey.energy), "pred": rows("pred", s0.pred.pos, s0.pred.energy),
             "plant": lineage._bits(s0.plant), "tough": lineage._bytes(np.asarray(s0.plant_genes)[..., 1]), "ate": []}]
    sim = [sim_frame(s0.prey.pos, s0.prey.alive, s0.pred.pos, s0.pred.alive)]
    was = {k: np.asarray(getattr(s0, k).alive) for k in lineage.SPECIES}
    mismatched = 0
    for t in range(steps):
        # step t's actions and calls belong to frame t
        sim[t]["act"] = np.asarray(out["prey_act"][t])
        sim[t]["call"] = np.asarray(out["call"][t], bool)
        view[t]["prey_call"] = [int(sim[t]["call"][s]) for s in np.nonzero(sim[t]["uid"] >= 0)[0]]
        eaten = np.asarray(info.x_prey_eaten[t]) if exact else None
        if exact and int(eaten.sum()) != int(info.catches[t]):
            mismatched += 1
        tr.step(t + 1, "prey", info.prey_parent[t], was["prey"], info.x_prey_alive[t], eaten,
                info.x_prey_genes[t], info.x_prey_gen[t])
        tr.step(t + 1, "pred", info.pred_parent[t], was["pred"], info.x_pred_alive[t], None,
                info.x_pred_genes[t], info.x_pred_gen[t])
        ate = [int(tr.slots["pred"][i]) for i in np.nonzero(info.x_pred_caught[t])[0] if tr.slots["pred"][i] >= 0]
        f = {"prey": rows("prey", info.x_prey_pos[t], info.x_prey_energy[t]),
             "pred": rows("pred", info.x_pred_pos[t], info.x_pred_energy[t]), "plant": lineage._bits(info.x_plant[t]),
             "ate": ate}
        view.append(f)  # plant toughness is frozen in this world, so frame 0's map holds (the viewer keeps the last one)
        sim.append(sim_frame(info.x_prey_pos[t], info.x_prey_alive[t], info.x_pred_pos[t], info.x_pred_alive[t]))
        was = {"prey": np.asarray(info.x_prey_alive[t]), "pred": np.asarray(info.x_pred_alive[t])}
    view[steps]["prey_call"] = [0] * len(view[steps]["prey"])  # no step is taken from the last frame
    sim[steps]["act"] = np.zeros_like(sim[steps]["alive"], np.int32)
    sim[steps]["call"] = np.zeros_like(sim[steps]["alive"])
    for a in tr.animals:  # starvation is the only death a hunter has
        if a["species"] == "pred" and a["cause"] == "died":
            a["cause"] = "starved"
    return tr, view, sim, mismatched


# ---------------------------------------------------------------- moments

def _wrap(v, size):
    return (v + size // 2) % size - size // 2


def _away(act, off) -> bool:
    """The move `act` takes the mover away from a point `off` (point minus mover, wrapped): negative dot product."""
    return int((np.asarray(pp.DELTAS[int(act) % NUM_ACTIONS]) * off).sum()) < 0


def _move(act, off) -> str:
    """away, toward or sideways relative to the point `off`, or still."""
    mv = np.asarray(pp.DELTAS[int(act) % NUM_ACTIONS])
    if not mv.any():
        return "still"
    dot = int((mv * off).sum())
    return "away" if dot < 0 else ("toward" if dot > 0 else "sideways")


def heard_calls(sim, size: int, radius: int, early: int = EARLY) -> list[dict]:
    """Every forced call made at step t < early that a quiet grazer hears and steps away from at t + 1.
    A call made at t is heard at t + 1 (killed callers are silent: the caller must be the same living animal at t + 1).
    A listener at t + 1: alive, the same animal as at t, 2 to `radius` cells from the caller, no hunter within 1 cell
    (grazers see hunters only that close), and its move at t + 1 goes away from the caller. Returns one record per
    (step, caller) with its listeners."""
    out = []
    for t in range(min(early, len(sim) - 2)):
        f, g = sim[t], sim[t + 1]
        for i in np.nonzero(f["call"] & f["alive"])[0]:
            u = int(f["uid"][i])
            if not (g["alive"][i] and int(g["uid"][i]) == u):
                continue
            listeners = []
            for j in np.nonzero(g["alive"])[0]:
                v = int(g["uid"][j])
                if j == i or int(f["uid"][j]) != v:
                    continue
                off = _wrap(g["pos"][i] - g["pos"][j], size)
                dist = int(np.abs(off).max())
                if not 2 <= dist <= radius:
                    continue
                if (g["pred_alive"] & (np.abs(_wrap(g["pred_pos"] - g["pos"][j], size)).max(-1) <= 1)).any():
                    continue
                if _away(g["act"][j], off):
                    listeners.append({"uid": v, "distance": dist})
            if listeners:
                out.append({"step": t, "heard_step": t + 1, "caller": u, "listeners": listeners})
    return out


def identities(animals) -> dict:
    """uid -> a key that names the same animal in a diverged replay of the same world: founders by slot, newborns by
    (slot, birth step, parent's key). The real and muted versions number newborns differently once they diverge."""
    key = {}
    for a in animals:
        p = a["parent_uid"]
        key[a["uid"]] = ("f", a["species"], a["slot"]) if p is None else (a["slot"], a["born"], key[p])
    return key


def fate(a, heard_step: int, horizon: int = HORIZON) -> dict:
    died = a["died"]
    return {"survived_50": died is None or died > heard_step + horizon, "died_step": died, "cause": a["cause"]}


def compare(moments, real_tr, muted_tr, muted_sim, size: int) -> list[dict]:
    """Each listener's outcome in the real episode and in the muted one, for the same animal (by identity)."""
    real_key = identities(real_tr.animals)
    muted_uid = {k: u for u, k in identities(muted_tr.animals).items()}
    ranked = []
    for m in moments:
        t1 = m["heard_step"]
        g = muted_sim[t1]
        slot_of = {int(u): s for s, u in enumerate(g["uid"]) if u >= 0}
        cu = muted_uid.get(real_key[m["caller"]])
        caller_muted = {"uid": cu, "alive": cu is not None and cu in slot_of,
                        "called": cu is not None and bool(muted_sim[m["step"]]["call"][np.nonzero(muted_sim[m["step"]]["uid"] == cu)[0]].any())}
        rows, strong = [], 0
        for L in m["listeners"]:
            real = {"moved_away": True, **fate(real_tr.animals[L["uid"]], t1)}
            mu = muted_uid.get(real_key[L["uid"]])
            if mu is None or mu not in slot_of:
                muted = {"uid": mu, "alive_at_heard_step": False, "moved_away": None, "move": None, "survived_50": None, "died_step": None,
                         "cause": None if mu is None else muted_tr.animals[mu]["cause"]}
            else:
                j = slot_of[mu]
                where = g["pos"][slot_of[cu]] if caller_muted["alive"] else None
                move = None if where is None else _move(g["act"][j], _wrap(where - g["pos"][j], size))
                muted = {"uid": mu, "alive_at_heard_step": True, "moved_away": None if move is None else move == "away",
                         "move": move, **fate(muted_tr.animals[mu], t1)}
            # strong: away and survived for real; in the muted world alive to hear, then died or didn't step away
            is_strong = bool(real["survived_50"] and muted["alive_at_heard_step"]
                             and (not muted["survived_50"] or muted["moved_away"] is False))
            strong += int(is_strong)
            rows.append({"uid": L["uid"], "distance": L["distance"], "real": real, "muted": muted, "strong": bool(is_strong)})
        died = sum(bool(r["strong"] and not r["muted"]["survived_50"]) for r in rows)
        ranked.append({**m, "caller_muted": caller_muted, "listeners": rows, "n_listeners": len(rows), "n_strong": strong,
                       "n_strong_muted_died": died, "n_real_survived": sum(r["real"]["survived_50"] for r in rows)})
    ranked.sort(key=lambda r: (-r["n_strong"], -r["n_strong_muted_died"], -int(r["caller_muted"]["called"]),
                               -r["n_real_survived"], r["step"]))
    return ranked


# ---------------------------------------------------------------- main

def episode_json(run_name, seed, tr, view, meta, label, hunters):
    lines = lineage.analyse(tr.animals, meta["steps"], 5, 0)
    ep = lineage.replay_json(run_name, 0, tr, view, meta, lines)
    ep.update(condition=label, hunters=hunters, seed=seed, oracle_call=True,
              prey_call="steps[t].prey_call[k] is 1 when the grazer in steps[t].prey[k] has the forced call bit on at step t "
                        "(a living hunter within 1 cell before it moves), else 0. The simulation sets it; the grazer's own "
                        "call output is ignored. It is heard from step t + 1, up to 4 cells away.",
              reception="muted: grazers hear no calls; calls are still made and paid for" if label == "muted" else "real")
    return ep, lines


def seed_list(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grazers", required=True)
    p.add_argument("--hunters", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="0-15")
    p.add_argument("--steps", type=int, default=500)
    a = p.parse_args(argv)
    grazers, hunters, out_dir = Path(a.grazers), Path(a.hunters), Path(a.out)
    world, cfg = eval_world(grazers, False)
    assert cfg.oracle_call, f"{grazers} isn't an oracle-call run"
    net = make_net(world, cfg)
    prey_p, prey_u = brain(grazers, "prey", net, cfg)
    pred_p, pred_u = brain(hunters, "pred", net, cfg)
    seeds = seed_list(a.seeds)
    play, exact = make_play(cfg, net, prey_p, pred_p, a.steps)
    S = jnp.asarray(seeds, jnp.uint32)
    played = {label: jax.device_get(play(S, jnp.int32(mode))) for label, mode in (("real", 0), ("muted", 1))}
    radius = cfg.view // 2
    search, best = [], None
    for e, seed in enumerate(seeds):
        traced = {label: trace(*played[label], e, a.steps, exact) for label in played}
        moments = compare(heard_calls(traced["real"][2], cfg.size, radius), traced["real"][0], traced["muted"][0],
                          traced["muted"][2], cfg.size)
        n_strong = sum(m["n_strong"] > 0 for m in moments)
        row = {"seed": seed, "strong_moments": n_strong, "strong_listeners": sum(m["n_strong"] for m in moments),
               "moments": len(moments), "calls_before_150": int(sum(f["call"].sum() for f in traced["real"][2][:EARLY]))}
        search.append(row)
        print(json.dumps(row), flush=True)
        if best is None or (row["strong_moments"], row["strong_listeners"]) > (best[0]["strong_moments"], best[0]["strong_listeners"]):
            best = (row, traced, moments)
    row, traced, moments = best
    seed = row["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for label in ("real", "muted"):
        tr, view, _, mismatched = traced[label]
        meta = {"world": "alarm", "run_kind": "shared", "mode": "darwin", "frozen": True, "bodies": None, "steps": a.steps,
                "episodes": 1, "seed": seed, "size": cfg.size, "terrain": lineage._bytes(np.zeros((cfg.size, cfg.size))),
                "flank": False, "pack_size": 0, "stealth": int(cfg.stealth_radius), "climate": "", "winter_len": 0,
                "cause_of_death": "exact" if exact and not mismatched else ("inexact" if exact else "unknown")}
        ep, lines = episode_json(grazers.name, seed, tr, view, meta, label, hunters.name)
        name = "episode3d.json" if label == "real" else "episode3d_muted.json"
        (out_dir / name).write_text(json.dumps(ep, separators=(",", ":")))
        files[label] = {"file": name, "animals": len(tr.animals), "cause_of_death": meta["cause_of_death"],
                        "grazer_lines": sum(r["species"] == "prey" for r in lines)}
    name_of = {label: {r["founder"]: r["name"] for r in lineage.analyse(traced[label][0].animals, a.steps, 5, 0)}
               for label in traced}
    roots = {label: lineage._roots(traced[label][0].animals) for label in traced}
    for m in moments:
        m["caller_family"] = name_of["real"][roots["real"][m["caller"]]]
        for L in m["listeners"]:
            L["family"] = name_of["real"][roots["real"][L["uid"]]]
    doc = {"grazers": grazers.name, "grazer_update": prey_u, "hunters": hunters.name, "hunter_update": pred_u, "seed": seed,
           "steps": a.steps, "files": files,
           "definition": f"A forced call made at step t < {EARLY} by a grazer with a living hunter within 1 cell, heard at "
                         f"t + 1 by grazers 2 to {radius} cells away with no hunter within 1 cell (all they can see), that "
                         f"step away from the caller at t + 1. survived_50: alive {HORIZON} steps after t + 1. muted: the "
                         "same animal (by slot, birth and parent) in the muted episode; moved_away is measured against "
                         "the same caller's position there (null if that caller isn't alive there). strong: survived in "
                         "the real episode, and in the muted one was alive at t + 1 but died within 50 steps or moved "
                         "toward, sideways or not at all (move). Ranked by strong listeners, "
                         "then those of them that died muted, then whether the same caller also called in the muted "
                         "world (a closer scene), then real survivors, then step.",
           "caution": "Every call is made by the simulation, not chosen by the grazer. The muted episode diverges from the "
                      "real one after the first call is heard, so by step t its scene can differ; a single moment is "
                      "illustrative. Aggregate (spec Result, fixed cohort, 3 seeds): listener 50-step risk with real "
                      "calls was .37 to .38 of muted and .47 to .49 of scrambled; listeners stepped away from the nearest "
                      "call 85 to 87% of the time.",
           "total_moments": len(moments), "strong_moments": row["strong_moments"], "moments": moments[:TOP]}
    (out_dir / "moments.json").write_text(json.dumps(doc, indent=1))
    (out_dir / "search.json").write_text(json.dumps({"picked": seed, "seeds": search}, indent=1))
    print(f"seed {seed}: {row['strong_moments']} strong moments of {len(moments)}; wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
