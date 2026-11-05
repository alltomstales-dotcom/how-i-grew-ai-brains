"""Dynasties: python -m scripts.lineage --run-dir runs/A [--run-dir runs/B ...] [--out-dir runs/lineage]

Replays episodes of a trained run and gives every animal a uid at birth, so a slot reused by a newborn gets a new uid. Each
animal records its parent's uid, species, birth and death steps, and how it died: "eaten" or "starved" (null if alive at the
end). Every founder alive at step 0 starts a family line. For each line: descendants, peak share of its species and when, the
extinction step, a share-over-time series, and story moments (rise past 25/50/80%, takeover at 100%, fall to half its peak,
near extinction and comeback, last survivor, extinction, longest-lived member).

Writes <out-dir>/<run>/lineage.json (every episode), replays of the episodes holding the run's most dramatic lines in the 3-D
viewer's format (episode3d_ep<k>.json, and the best again as episode3d.json) with lineage fields on every animal, and
<out-dir>/lines.md ranking the most dramatic lines across every run in <out-dir>.

Individual-brain runs (scripts.train_indiv) replay under the run's own rule (mutation, learning, culture, anchoring) unless
--frozen. Shared-brain runs (scripts.train) give every animal the run's one brain, so their lines are lines of bodies.
Cause of death is exact: the world's predation step is watched as it runs, and checked against the step's catch count.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import types
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SPECIES = ("prey", "pred")
WORD = {"prey": "grazers", "pred": "hunters"}
GENES = ("speed", "sight", "breed", "size", "brain", "diet", "camo")
GRAZER_NAMES = ("Clover", "Fern", "Sorrel", "Yarrow", "Bramble", "Heather", "Sedge", "Thistle", "Nettle", "Moss", "Rowan",
                "Willow", "Aspen", "Alder", "Hazel", "Juniper", "Laurel", "Myrtle", "Briar", "Cress", "Dock", "Flax", "Gorse",
                "Ivy", "Lupin", "Mallow", "Oat", "Reed", "Rush", "Tansy", "Vetch", "Sage")
HUNTER_NAMES = ("Ash", "Flint", "Slate", "Ember", "Cinder", "Basalt", "Onyx", "Jasper", "Garnet", "Quartz", "Obsidian",
                "Shale", "Granite", "Ochre", "Pumice", "Iron")
COLORS = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
          "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080")
RISES = (25, 50, 80)
NEAR = 2  # a line down to this many living members, after it had at least GROWN, is near extinction
GROWN = 3
TOUGH_EVERY = 50


# ---------------------------------------------------------------------------------------------------------------- uids

class Tracer:
    """uids for one world's animals. A newborn is a slot whose parent isn't itself (StepInfo.prey_parent/pred_parent);
    births come after deaths in a step, so a slot alive before a step that holds a newborn after it lost its animal."""

    def __init__(self):
        self.animals: list[dict] = []
        self.slots: dict[str, np.ndarray] = {}

    def _new(self, kind, parent, t, slot, genes=None, gen=None) -> int:
        uid = len(self.animals)
        a = {"uid": uid, "parent_uid": parent, "species": kind, "born": int(t), "died": None, "cause": None, "slot": int(slot)}
        if genes is not None:
            a["genes"] = {g: round(float(v), 3) for g, v in zip(GENES, genes)}
        if gen is not None:
            a["gen"] = int(gen)
        self.animals.append(a)
        return uid

    def start(self, kind, alive, genes=None, gen=None) -> None:
        alive = np.asarray(alive, bool)
        slots = np.full(alive.shape[0], -1, np.int64)
        for s in np.nonzero(alive)[0]:
            slots[s] = self._new(kind, None, 0, s, None if genes is None else genes[s], None if gen is None else gen[s])
        self.slots[kind] = slots

    def step(self, t, kind, parent, was_alive, alive, eaten=None, genes=None, gen=None) -> dict:
        """After step t. Returns {"born": [uid], "died": [uid]}."""
        parent, was_alive, alive = np.asarray(parent), np.asarray(was_alive, bool), np.asarray(alive, bool)
        slots = self.slots[kind]
        n = parent.shape[0]
        born = parent != np.arange(n)
        died = was_alive & (~alive | born)
        gone = []
        for s in np.nonzero(died)[0]:
            a = self.animals[slots[s]]
            a["died"] = int(t)
            a["cause"] = "died" if eaten is None else ("eaten" if eaten[s] else "starved")
            gone.append(int(slots[s]))
        new = np.where(died, -1, slots)
        came = []
        for s in np.nonzero(born)[0]:
            p = int(slots[parent[s]])
            if p < 0 or died[parent[s]]:
                raise ValueError(f"step {t}: {kind} slot {s} was born to slot {parent[s]}, which holds no living animal")
            new[s] = self._new(kind, p, t, s, None if genes is None else genes[s], None if gen is None else gen[s])
            came.append(int(new[s]))
        if not ((new >= 0) == alive).all():
            raise ValueError(f"step {t}: {kind} uids and living slots disagree")
        self.slots[kind] = new
        return {"born": came, "died": gone}


# ------------------------------------------------------------------------------------------------------------ analysis

def _roots(animals) -> dict:
    root = {}
    for a in animals:  # uids grow with time, so a parent comes before its children
        root[a["uid"]] = a["uid"] if a["parent_uid"] is None else root[a["parent_uid"]]
    return root


def _depths(animals) -> dict:
    d = {}
    for a in animals:
        d[a["uid"]] = 0 if a["parent_uid"] is None else d[a["parent_uid"]] + 1
    return d


def counts(animals, T: int, kind: str):
    """(founder uids, (T+1, F) living members of each line at each step, (T+1,) living animals of the species)."""
    founders = [a["uid"] for a in animals if a["species"] == kind and a["parent_uid"] is None]
    col = {u: i for i, u in enumerate(founders)}
    root = _roots(animals)
    diff = np.zeros((T + 2, max(len(founders), 1)), np.int64)
    for a in animals:
        if a["species"] != kind:
            continue
        end = T + 1 if a["died"] is None else a["died"]
        diff[a["born"], col[root[a["uid"]]]] += 1
        diff[end, col[root[a["uid"]]]] -= 1
    n = np.cumsum(diff, 0)[: T + 1, : len(founders)]
    return founders, n, n.sum(1)


def _word(kind):
    return WORD[kind]


TAKEOVER_WEIGHT = 0.5  # in 1000 steps one grazer line nearly always takes over, so a takeover alone counts half


def line_moments(name, kind, share, n, members, T, species_end=None, valid=None) -> tuple[list, dict]:
    """Story moments of one line, and the parts of its drama score (percentage points of its species).
    species_end: the step the whole species died out, if it did. valid: steps where the species is still a population
    (at least half its founders' number); shares only count there, so a line isn't "100% of the hunters" when two are
    left, and a species dying out isn't the line's fall. Extinction and last survivor use the raw counts."""
    out, w = [], _word(kind)
    valid = np.ones(len(share), bool) if valid is None else np.asarray(valid, bool)
    up = np.where(valid, share, -1.0)  # for peaks and thresholds
    start = float(share[0])
    peak_step = int(up.argmax())
    peak = max(float(up[peak_step]), start)
    for th in RISES:
        if start < th:
            hit = np.nonzero(up >= th)[0]
            if len(hit):
                out.append({"kind": "rise", "pct": th, "step": int(hit[0]), "text": f"{name} pass {th}% of the {w}"})
    full = np.nonzero((up >= 100.0) & (n >= 2))[0]
    if len(full) and start < 100:
        out.append({"kind": "takeover", "step": int(full[0]), "text": f"every living {w[:-1]} is a {name}"})
    down = np.where(valid, share, np.inf)[peak_step:]  # for troughs after the peak
    half = np.nonzero(down <= peak / 2)[0]
    if peak > start and len(half):
        out.append({"kind": "fall", "step": peak_step + int(half[0]), "pct": round(float(down[half[0]]), 1),
                    "from_pct": round(peak, 1), "text": f"{name} fall to half their peak of {peak:.0f}%"})
    alive = np.nonzero(n > 0)[0]
    extinct = None if n[-1] > 0 else int(alive[-1]) + 1
    # near extinction and comeback: down to NEAR members after having GROWN, then the share regained with at least
    # GROWN members again (a share that rises only because rivals die, with the line still at one or two, is no comeback)
    comeback, best = 0.0, None
    grown = np.nonzero(n >= GROWN)[0]
    if len(grown):
        g = int(grown[0])
        big = np.where((n[g:] >= GROWN) & valid[g:], share[g:], -1.0)
        run_max = np.maximum.accumulate(up[g:])
        later_max = np.maximum.accumulate(big[::-1])[::-1]
        low = (n[g:] > 0) & (n[g:] <= NEAR) & valid[g:]
        score = np.where(low, np.maximum(0.0, np.minimum(run_max - share[g:], later_max - share[g:])), 0.0)
        if score.max() > 0:
            b = g + int(score.argmax())
            rec = b + int(big[b - g:].argmax())
            comeback, best = float(score.max()), (b, rec)
    if best and share[best[1]] > share[best[0]]:
        b, rec = best
        out.append({"kind": "near_extinction", "step": b, "members": int(n[b]), "pct": round(float(share[b]), 1),
                    "text": f"{name} down to {int(n[b])} living"})
        out.append({"kind": "comeback", "step": rec, "pct": round(float(share[rec]), 1), "members": int(n[rec]),
                    "text": f"{name} back to {share[rec]:.0f}% of the {w}"})
    if extinct is not None:
        lasts = np.nonzero(n[:extinct] >= 2)[0]
        if len(members) > 1:
            since = int(lasts[-1]) + 1 if len(lasts) else 0
            last = [a for a in members if a["born"] <= since and (a["died"] is None or a["died"] > since)]
            if last:
                out.append({"kind": "last_survivor", "step": since, "uid": last[0]["uid"],
                            "text": f"#{last[0]['uid']} is the last {name}"})
        with_species = species_end is not None and extinct == species_end
        out.append({"kind": "extinction", "step": extinct, "with_species": with_species,
                    "text": f"the {name} line dies out" + (f", and with it the last of the {w}" if with_species else "")})
    life = lambda a: (T + 1 if a["died"] is None else a["died"]) - a["born"]
    oldest = max(members, key=lambda a: (life(a), -a["uid"]))
    out.append({"kind": "longest_lived", "step": oldest["born"], "uid": oldest["uid"], "age": life(oldest),
                "alive_at_end": oldest["died"] is None,
                "text": f"#{oldest['uid']}, born step {oldest['born']}, lives {life(oldest)} steps" + (" and is alive at the end" if oldest["died"] is None else "")})
    out.sort(key=lambda m: (m["step"], m["kind"]))
    trough = float(down.min()) if np.isfinite(down.min()) else peak
    end = float(share[-1]) if valid[-1] else 0.0
    parts = {"rise_fall": max(0.0, min(peak - start, peak - trough)), "comeback": comeback,
             "takeover": TAKEOVER_WEIGHT * max(0.0, end - start) if end >= 50 else 0.0}
    fell_with_species = extinct is not None and species_end is not None and extinct == species_end
    return out, {"extinct": extinct, "peak": peak, "peak_step": peak_step, "start": start, "parts": parts,
                 "fell_with_species": fell_with_species}


def species_end(animals, T: int) -> dict:
    """The step each species died out, or None."""
    out = {}
    for kind in SPECIES:
        _, _, total = counts(animals, T, kind)
        gone = np.nonzero(total == 0)[0]
        out[kind] = int(gone[0]) if len(gone) and not total[gone[0]:].any() else None
    return out


def analyse(animals, T: int, every: int = 5, episode: int = 0) -> list[dict]:
    """The founder table: one row per founder line, both species, with its series and moments. Names rotate by episode,
    so the same slot in two episodes doesn't share a name."""
    ends = species_end(animals, T)
    depth = _depths(animals)
    root = _roots(animals)
    rows = []
    for kind in SPECIES:
        founders, n, total = counts(animals, T, kind)
        share = np.where(total[:, None] > 0, 100.0 * n / np.maximum(total[:, None], 1), 0.0)
        floor = max(2, -(-len(founders) // 2))  # a population: at least half the founders' number
        valid = total >= floor
        names = GRAZER_NAMES if kind == "prey" else HUNTER_NAMES
        by_line: dict[int, list] = {u: [] for u in founders}
        for a in animals:
            if a["species"] == kind:
                by_line[root[a["uid"]]].append(a)
        for i, u in enumerate(founders):
            j = i + 5 * episode
            name = names[j % len(names)] if len(founders) <= len(names) else f"{names[i % len(names)]} {i // len(names) + 1}"
            members = by_line[u]
            moments, s = line_moments(name, kind, share[:, i], n[:, i], members, T, ends[kind], valid)
            parts = s["parts"]
            arc = max(parts, key=parts.get)
            f = next(a for a in members if a["uid"] == u)
            ts = sorted(set(range(0, T + 1, every)) | {T} | {m["step"] for m in moments if m["step"] <= T})
            rows.append({
                "founder": u, "name": name, "species": kind, "color": COLORS[len(rows) % len(COLORS)], "born": f["born"],
                "slot": f["slot"], "genes": f.get("genes"), "descendants": len(members) - 1,
                "children": sum(1 for a in members if a["parent_uid"] == u),
                "generations": max(depth[a["uid"]] for a in members), "peak_pct": round(s["peak"], 1), "peak_step": s["peak_step"],
                "start_pct": round(s["start"], 1), "final_pct": round(float(share[-1, i]), 1), "final_members": int(n[-1, i]),
                "extinct_step": s["extinct"], "drama": round(sum(parts.values()), 1), "arc": arc if max(parts.values()) > 0 else "none",
                "drama_parts": {k: round(v, 1) for k, v in parts.items()}, "fell_with_species": s["fell_with_species"],
                "share_floor": floor, "moments": moments,
                "series": [[t, round(float(share[t, i]), 1), int(n[t, i]), int(total[t])] for t in ts],
            })
    return rows


# --------------------------------------------------------------------------------------------------------------- replay

def _bits(mask) -> str:
    return base64.b64encode(np.packbits(np.asarray(mask).reshape(-1).astype(np.uint8)).tobytes()).decode()


def _bytes(values) -> str:
    return base64.b64encode(np.clip(np.asarray(values).reshape(-1) * 255, 0, 255).astype(np.uint8).tobytes()).decode()


def _extended(world):
    """The world's StepInfo plus what the tracer needs, and an advance() that fills it in."""
    extra = ("x_prey_eaten", "x_pred_caught", "x_prey_pos", "x_prey_alive", "x_prey_energy", "x_prey_genes", "x_prey_gen",
             "x_pred_pos", "x_pred_alive", "x_pred_energy", "x_pred_genes", "x_pred_gen", "x_plant")
    Info = NamedTuple("LineageInfo", [(f, Any) for f in world.StepInfo._fields + extra])

    def advance(cfg, key, state, prey_act, pred_act):
        box = {}
        orig = getattr(world, "_predation", None)
        if orig is not None:
            def spy(*a, **k):
                out = orig(*a, **k)
                box["caught"], box["eaten"] = out[0], out[1]
                return out
            world._predation = spy
        try:
            res = world.advance(cfg, key, state, prey_act, pred_act)
        finally:
            if orig is not None:
                world._predation = orig
        s = res.state
        eaten = box.get("eaten", jnp.zeros(s.prey.alive.shape, bool)).astype(bool)
        caught = box.get("caught", jnp.zeros(s.pred.alive.shape, bool)).astype(bool)
        info = Info(*res.info, eaten, caught, s.prey.pos, s.prey.alive, s.prey.energy, s.prey.genes, s.prey.gen,
                    s.pred.pos, s.pred.alive, s.pred.energy, s.pred.genes, s.pred.gen, s.plant)
        return res._replace(info=info)

    return types.SimpleNamespace(advance=advance, observe=world.observe, batch_step=None), hasattr(world, "_predation")


def simulate(run_dir: Path, episodes: int, steps: int, seed: int, frozen: bool = False, chunk: int = TOUGH_EVERY):
    """Play `episodes` worlds of the run for `steps` steps. Returns (tracers, frames, meta): frames[e] is a list of
    per-step dicts for the replay (rows of [uid, x, y, energy]), built for every world."""
    from agents.checkpoint import load_brains
    from agents.indiv import CULTURE, Carry, IndivConfig, anchor, anchored_worlds, blank_brain, fresh_kin, make_step, Pop
    from agents.networks import init_params
    from scripts.eval import load_world_config
    from scripts.eval_indiv import draw
    from scripts.league import evolved_pool
    from simulations.worlds import make_net, obs_channels

    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    if not hasattr(world, "StepInfo") or "prey_parent" not in world.StepInfo._fields:
        raise SystemExit(f"{run_dir}: world {config.get('world_name')} records no parents")
    if config.get("train", {}).get("memory", 0):
        raise SystemExit(f"{run_dir}: brains with memory aren't supported")
    net = make_net(world, wcfg, 0)
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    E = episodes
    key, k_reset, k_prey, k_pred = jax.random.split(jax.random.PRNGKey(seed), 4)
    pool, bodies = evolved_pool(run_dir)
    state = jax.vmap(lambda k: world.reset(wcfg, k) if pool is None else world.reset(wcfg, k, pool))(jax.random.split(k_reset, E))
    prey_obs, pred_obs = jax.vmap(lambda s: world.observe(wcfg, s))(state)
    snap_file = run_dir / "checkpoints" / "pop_latest.npz"
    names = {f.name for f in dataclasses.fields(IndivConfig)}
    anchor_pool, blank = None, None
    if snap_file.exists():
        kind_of_run = "individual"
        snap = dict(np.load(snap_file))
        prey_pop, pred_pop = draw(snap, "prey", template, E, wcfg.max_prey, k_prey), draw(snap, "pred", template, E, wcfg.max_pred, k_pred)
        cfg = IndivConfig(**{k: v for k, v in config.get("indiv", {}).items() if k in names})
    else:
        kind_of_run = "shared"
        prey0, pred0 = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)[:2]
        prey_pop, pred_pop = Pop(prey0, prey0), Pop(pred0, pred0)  # one brain each, applied to every animal
        cfg = IndivConfig(mode="darwin", sigma=0.0)
    if frozen and kind_of_run == "individual":
        cfg = IndivConfig(mode="darwin", sigma=0.0)
        prey_pop, pred_pop = Pop(prey_pop.params, prey_pop.params), Pop(pred_pop.params, pred_pop.params)
    carry = Carry(state, prey_pop, pred_pop, prey_obs, pred_obs, key)
    if cfg.anchor_frac > 0:
        from scripts.train_indiv import load_pool
        given = Path(config["args"]["anchor_pool"])
        where = next((p for p in (given, run_dir.parent / given.name) if (p / "checkpoints" / "pop_latest.npz").exists()), None)
        if where is None:
            raise SystemExit(f"{run_dir}: anchored, and its pool run {given} isn't here (or beside the run)")
        anchor_pool = load_pool(where, template)
        mask = jnp.broadcast_to(anchored_worlds(E, cfg.anchor_frac)[:, None], (E, wcfg.max_prey))
        carry = carry._replace(prey=anchor(carry.prey, mask, anchor_pool, jax.random.fold_in(k_prey, 19)))
    if cfg.mode in CULTURE:
        blank = blank_brain(net, wcfg.view, obs_channels(world, wcfg))
        carry = carry._replace(prey_kin=fresh_kin(E, wcfg.max_prey, cfg.childhood), pred_kin=fresh_kin(E, wcfg.max_pred, cfg.childhood))
    shim, exact = _extended(world)

    def shared_step(c, _):
        """Shared-brain runs: the run's one brain per species acts for every animal; nothing learns or is inherited."""
        every_animal = lambda p, obs: jax.vmap(jax.vmap(lambda o: net.apply(p, o)[0]))(obs)
        key_, kp, kq, ks = jax.random.split(c.key, 4)
        pa = jax.random.categorical(kp, every_animal(c.prey.params, c.prey_obs))
        qa = jax.random.categorical(kq, every_animal(c.pred.params, c.pred_obs))
        res = jax.vmap(partial(shim.advance, wcfg))(jax.random.split(ks, E), c.state, pa, qa)
        po, qo = jax.vmap(partial(world.observe, wcfg))(res.state)
        return c._replace(state=res.state, prey_obs=po, pred_obs=qo, key=key_), res.info

    @jax.jit
    def run_chunk(c, pool_):
        step = shared_step if kind_of_run == "shared" else make_step(net, shim, wcfg, cfg, restart=False, blank=blank, anchor_pool=pool_)
        return jax.lax.scan(step, c, None, length=chunk)

    s0 = jax.device_get(carry.state)
    tracers, frames = [Tracer() for _ in range(E)], [[] for _ in range(E)]
    has_genes = hasattr(s0.prey, "genes")

    def rows(tr, kind, pos, energy):
        slots = tr.slots[kind]
        return [[int(slots[i]), int(pos[i, 0]), int(pos[i, 1]), round(float(energy[i]), 1)] for i in np.nonzero(slots >= 0)[0]]

    for e in range(E):
        for kind in SPECIES:
            sp = getattr(s0, kind)
            tracers[e].start(kind, sp.alive[e], sp.genes[e] if has_genes else None, sp.gen[e] if has_genes else None)
        frames[e].append({"prey": rows(tracers[e], "prey", s0.prey.pos[e], s0.prey.energy[e]),
                          "pred": rows(tracers[e], "pred", s0.pred.pos[e], s0.pred.energy[e]), "plant": _bits(s0.plant[e]),
                          "tough": _bytes(np.asarray(s0.plant_genes[e])[..., 1]), "ate": []})
    was = {k: np.asarray(getattr(s0, k).alive) for k in SPECIES}
    mismatched, t = 0, 0
    while t < steps:
        carry, info = run_chunk(carry, anchor_pool)
        info = jax.device_get(info)
        tough = np.asarray(jax.device_get(carry.state.plant_genes))[..., 1]
        for j in range(chunk):
            t += 1
            if t > steps:
                break
            for e in range(E):
                tr = tracers[e]
                eaten = info.x_prey_eaten[j, e] if exact else None
                if exact and int(eaten.sum()) != int(info.catches[j, e]):
                    mismatched += 1
                tr.step(t, "prey", info.prey_parent[j, e], was["prey"][e], info.x_prey_alive[j, e], eaten,
                        info.x_prey_genes[j, e], info.x_prey_gen[j, e])
                tr.step(t, "pred", info.pred_parent[j, e], was["pred"][e], info.x_pred_alive[j, e], None,
                        info.x_pred_genes[j, e], info.x_pred_gen[j, e])
                ate = [int(tr.slots["pred"][i]) for i in np.nonzero(info.x_pred_caught[j, e])[0] if tr.slots["pred"][i] >= 0]
                f = {"prey": rows(tr, "prey", info.x_prey_pos[j, e], info.x_prey_energy[j, e]),
                     "pred": rows(tr, "pred", info.x_pred_pos[j, e], info.x_pred_energy[j, e]), "plant": _bits(info.x_plant[j, e]),
                     "ate": ate}
                if t % TOUGH_EVERY == 0 and j == chunk - 1:
                    f["tough"] = _bytes(tough[e])
                frames[e].append(f)
            was = {"prey": np.asarray(info.x_prey_alive[j]), "pred": np.asarray(info.x_pred_alive[j])}
    for tr in tracers:  # starvation is the only death a hunter has
        for a in tr.animals:
            if a["species"] == "pred" and a["cause"] == "died":
                a["cause"] = "starved"
    final = jax.device_get(carry.state)
    terrain = np.asarray(final.terrain[0]) if hasattr(final, "terrain") else np.zeros((wcfg.size, wcfg.size))
    meta = {"world": config.get("world_name"), "run_kind": kind_of_run, "mode": cfg.mode, "frozen": frozen,
            "anchor_frac": cfg.anchor_frac, "bodies": bodies, "steps": steps, "episodes": E, "seed": seed,
            "cause_of_death": "exact" if exact and not mismatched else ("inexact" if exact else "unknown"),
            "cause_mismatched_steps": mismatched, "size": wcfg.size, "terrain": _bytes(terrain / 2.0),
            "flank": bool(getattr(wcfg, "flank", False)), "pack_size": int(getattr(wcfg, "pack_size", 0)),
            "stealth": int(getattr(wcfg, "stealth_radius", 0)), "climate": getattr(wcfg, "climate", ""),
            "winter_len": int(getattr(wcfg, "winter_len", 0)), "world_set": config.get("args", {}).get("world_set", [])}
    if getattr(wcfg, "hills", False):
        from simulations.evo import height_map
        meta["height"] = _bytes(height_map(wcfg))
    if getattr(wcfg, "thickets", 0) > 0:
        from simulations.evo import thicket_map
        meta["cover"] = _bits(thicket_map(wcfg))
    return tracers, frames, meta


def replay_json(run_name, episode, tracer, frames, meta, lines) -> dict:
    """The 3-D viewer's episode3d format, with each animal's lineage and the episode's line table."""
    T = len(frames)
    line_of = _roots(tracer.animals)
    animals = {}
    for a in tracer.animals:
        g = a.get("genes") or {}
        animals[a["uid"]] = {"kind": a["species"], "born": a["born"], "died": T if a["died"] is None else a["died"],
                             "gen": a.get("gen", 0), **{k: g.get(k, 0.0) for k in GENES},
                             "parent": a["parent_uid"], "line": line_of[a["uid"]], "cause": a["cause"],
                             "alive_at_end": a["died"] is None}
    keep = ("size", "terrain", "flank", "pack_size", "stealth", "climate", "winter_len", "height", "cover")
    out = {k: meta[k] for k in keep if k in meta}
    out.update(steps=frames, animals=animals, run=run_name, world=meta["world"], update=None, bodies=meta["bodies"],
               seed=meta["seed"], episode=episode,
               lines={r["founder"]: {k: r[k] for k in ("name", "species", "color", "founder", "born", "descendants",
                                                      "peak_pct", "peak_step", "extinct_step", "drama", "arc", "moments", "series")}
                      for r in lines},
               lineage={"episodes": meta["episodes"], "steps": meta["steps"], "mode": meta["mode"], "frozen": meta["frozen"],
                        "cause_of_death": meta["cause_of_death"]})
    return out


# ----------------------------------------------------------------------------------------------------------------- main

def trace_run(run_dir: Path, out_dir: Path, episodes: int, steps: int, seed: int, frozen: bool, replays: int, every: int,
              origin: str | None) -> dict:
    tracers, frames, meta = simulate(run_dir, episodes, steps, seed, frozen)
    eps = []
    for e, tr in enumerate(tracers):
        rows = analyse(tr.animals, steps, every, e)
        eps.append({"episode": e, "species_extinct": species_end(tr.animals, steps), "animals": [{k: a[k] for k in ("uid", "parent_uid", "species", "born", "died", "cause")}
                                              for a in tr.animals], "founders": rows})
    best = sorted(((r["drama"], e, r["founder"]) for e, ep in enumerate(eps) for r in ep["founders"]), reverse=True)
    picked = []
    for _, e, _ in best:
        if e not in picked:
            picked.append(e)
        if len(picked) >= replays:
            break
    target = out_dir / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, e in enumerate(picked):  # every replay under its episode's name, so a rerun overwrites them all; the best also as episode3d.json
        text = json.dumps(replay_json(run_dir.name, e, tracers[e], frames[e], meta, eps[e]["founders"]), separators=(",", ":"))
        files[e] = f"episode3d_ep{e}.json"
        (target / files[e]).write_text(text)
        if i == 0:
            (target / "episode3d.json").write_text(text)
    for ep in eps:
        ep["replay"] = files.get(ep["episode"])
    out = {"run": run_dir.name, "run_path": origin or str(run_dir), **{k: v for k, v in meta.items() if k not in ("terrain", "height", "cover")},
           "episodes_traced": eps}
    (target / "lineage.json").write_text(json.dumps(out, separators=(",", ":")))
    n = sum(len(ep["animals"]) for ep in eps)
    print(f"{target}: {episodes} episodes x {steps} steps, {n} animals, cause of death {meta['cause_of_death']}, replays {sorted(files.values())}", flush=True)
    return out


def _moment_line(m) -> str:
    return f"{m['kind'].replace('_', ' ')} @ {m['step']}"


ARCS = ("rise_fall", "comeback", "takeover")


def _detail(i, s, d, ep, r) -> list[str]:
    replay = (f"`{d['run']}/{ep['replay']}`" if ep.get("replay")
              else f"none written; rerun `--run-dir {d['run_path']} --episodes {d['episodes']} --seed {d['seed']} --steps {d['steps']} --replays {d['episodes']}`")
    gone = [f"{WORD[k]} at step {v}" for k, v in (ep.get("species_extinct") or {}).items() if v is not None]
    parts = ", ".join(f"{k} {v:.1f}" for k, v in r["drama_parts"].items() if v > 0) or "none"
    return [f"## {i}. {r['name']} {WORD[r['species']]}: {d['run']}, episode {ep['episode']} ({r['arc']}, drama {s:.1f})", "",
            f"- Run: `{d['run_path']}` ({d['world']}, {d['run_kind']} brains, mode {d['mode']}{', frozen' if d['frozen'] else ''}"
            f"{', grazers anchored in ' + format(d['anchor_frac'], '.0%') + ' of worlds' if d['anchor_frac'] else ''}); "
            f"seed {d['seed']}, episode {ep['episode']} of {d['episodes']}, {d['steps']} steps",
            f"- Founder #{r['founder']}, born step {r['born']} (slot {r['slot']}); {r['descendants']} descendants over {r['generations']} generations",
            f"- Share of the {WORD[r['species']]}: starts {r['start_pct']:.1f}%, peaks {r['peak_pct']:.1f}% at step {r['peak_step']}, "
            f"ends {r['final_pct']:.1f}% ({r['final_members']} living)" + (f"; line extinct at step {r['extinct_step']}" if r["extinct_step"] is not None else ""),
            f"- Drama parts: {parts}" + (f". Species died out: {', '.join(gone)}" if gone else ""),
            "- Moments: " + "; ".join(f"{m['text']} (step {m['step']})" for m in r["moments"]),
            f"- Replay: {replay}; highlight animals with `line == {r['founder']}`", ""]


def rank(out_dir: Path, top: int = 10, per_run: int = 3) -> Path:
    """lines.md: the most dramatic lines across every lineage.json under out_dir, at most per_run from one run in the
    overall table, then the best three of each arc."""
    entries = []
    for f in sorted(out_dir.glob("*/lineage.json")):
        d = json.loads(f.read_text())
        for ep in d["episodes_traced"]:
            for r in ep["founders"]:
                entries.append((r["drama"], d, ep, r))
    entries.sort(key=lambda x: (-x[0], x[1]["run"], x[2]["episode"], x[3]["founder"]))
    overall, used = [], {}
    for x in entries:
        if used.get(x[1]["run"], 0) < per_run:
            overall.append(x)
            used[x[1]["run"]] = used.get(x[1]["run"], 0) + 1
        if len(overall) >= top:
            break
    runs = sorted({x[1]["run"] for x in entries})
    out = ["# Dynasties: the most dramatic family lines", "",
           f"Runs: {', '.join(runs)}. A line is a founder alive at step 0 and all its descendants.", "",
           "Drama is in percentage points of the line's own species, the sum of three parts:",
           "- **rise_fall**: the smaller of its climb from its starting share to its peak and its fall from the peak afterwards;",
           f"- **comeback**: the smaller of its drop to at most {NEAR} living members (after it had at least {GROWN}) and its recovery after;",
           f"- **takeover**: its final share above where it began, when it ends with at least half its species, times {TAKEOVER_WEIGHT}",
           "  (in 1,000 steps one grazer line nearly always takes over, so a takeover alone ranks below a reversal).",
           "Shares count only while the species has at least half its founders' number alive, so a line is not",
           "\"100% of the hunters\" when two are left, and a species dying out is not a line's fall.",
           f"The arc is the largest part. At most {per_run} lines per run in the overall table.",
           f"Rebuild with `python -m scripts.lineage --rank-only --out-dir {out_dir}`.", "",
           "## Overall", "",
           "| # | line | species | run | episode | arc | drama | peak | descendants | line extinct |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (s, d, ep, r) in enumerate(overall, 1):
        out.append(f"| {i} | {r['name']} (founder #{r['founder']}) | {WORD[r['species']]} | {d['run']} | {ep['episode']} | {r['arc']}{' (with the species)' if r.get('fell_with_species') else ''} | "
                   f"{s:.1f} | {r['peak_pct']:.0f}% @ {r['peak_step']} | {r['descendants']} | {r['extinct_step'] if r['extinct_step'] is not None else '-'} |")
    out.append("")
    shown = [(x[1]["run"], x[2]["episode"], x[3]["founder"]) for x in overall]
    for arc in ARCS:
        best = sorted(entries, key=lambda x: (-x[3]["drama_parts"][arc], -x[0]))[:3]
        out += [f"## Best {arc}", ""]
        for s, d, ep, r in best:
            out.append(f"- {r['name']} {WORD[r['species']]} (founder #{r['founder']}), {d['run']} episode {ep['episode']}: "
                       f"{arc} {r['drama_parts'][arc]:.1f}, drama {s:.1f}")
            k = (d["run"], ep["episode"], r["founder"])
            if k not in shown:
                shown.append(k)
                overall.append((s, d, ep, r))
        out.append("")
    out += ["## Details", ""]
    for i, x in enumerate(overall, 1):
        out += _detail(i, *x)
    path = out_dir / "lines.md"
    path.write_text("\n".join(out))
    return path


def main(argv=None) -> int:
    from scripts.eval import EVAL_SEED
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", action="append", default=[], help="a run to trace, repeatable")
    p.add_argument("--out-dir", default="runs/lineage")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--frozen", action="store_true", help="brains frozen: no learning, children copy the brain unmutated")
    p.add_argument("--replays", type=int, default=8, help="replays per run, for the episodes of its most dramatic lines first")
    p.add_argument("--every", type=int, default=5, help="steps between points of each line's share series")
    p.add_argument("--top", type=int, default=10, help="lines in lines.md")
    p.add_argument("--origin", action="append", default=[], help="where each --run-dir lives, for lines.md (<gpu-host>:~/autolearn/runs/X)")
    p.add_argument("--rank-only", action="store_true", help="only rebuild lines.md from the lineage.json files in --out-dir")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    if not args.rank_only:
        if not args.run_dir:
            p.error("--run-dir is required unless --rank-only")
        for i, r in enumerate(args.run_dir):
            trace_run(Path(r), out_dir, args.episodes, args.steps, args.seed, args.frozen, args.replays, args.every,
                      args.origin[i] if i < len(args.origin) else None)
    if out_dir.exists():
        print(f"ranked: {rank(out_dir, args.top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
