"""Stage 11a: animals crossing between continents, each continent its own trainer, on any machine.

After each training update, some animals standing at the east edge leave for the next continent. They go as a small file in
<dir>/to-<next>/, and arrive at the west edge when that continent next looks. Nothing waits on anything, so a slow link only
delays arrivals. A migrant carries its genes, energy and lineage, and not its brain: it uses the brain of wherever it lands.

An animal's tag marks its lineage: 0 is native here, and n + 1 means it descends from an arrival from continent n. Children
inherit the tag, so the share of non-zero tags says how much of a continent descends from immigrants.
"""
from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

KINDS = ("prey", "pred")


def _write_atomic(path: Path, **arrays) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.chmod(tmp, 0o666)  # the relay between machines runs as another user
    os.replace(tmp, path)


class Transit:
    def __init__(self, directory, me: int, to: int, prob: float, strip: int, wcfg, world, seed: int = 0):
        self.dir, self.me, self.to, self.prob, self.strip = Path(directory), me, to, prob, strip
        self.cfg, self.world = wcfg, world
        self.rng = np.random.default_rng(seed + 977 * me)
        self.inbox, self.outbox = self.dir / f"to-{me}", self.dir / f"to-{to}"
        for d in (self.inbox, self.outbox):
            d.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(d, 0o777)
            except PermissionError:
                pass
        self.observe = jax.jit(jax.vmap(partial(world.observe, wcfg)))

    def exchange(self, runner, update: int):
        """Send and receive between updates, when no trajectory is open. Returns (runner, stats)."""
        state = jax.device_get(runner.env_state)
        species = {k: {f: np.array(getattr(getattr(state, k), f)) for f in ("pos", "energy", "alive", "genes", "gen", "tag")} for k in KINDS}
        stats, cleared = {}, {}
        out = {}
        for k in KINDS:
            s = species[k]
            leaving = s["alive"] & (s["pos"][..., 0] >= self.cfg.size - self.strip) & (self.rng.random(s["alive"].shape) < self.prob)
            out.update({f"{k}_{f}": s[f][leaving] for f in ("pos", "energy", "genes", "gen", "tag")})
            s["alive"][leaving] = False
            s["energy"][leaving] = 0.0
            stats[f"{k}_sent"] = int(leaving.sum())
            cleared[k] = leaving
        if stats["prey_sent"] or stats["pred_sent"]:
            _write_atomic(self.outbox / f"{self.me}-{update:07d}.npz", origin=np.array(self.me), **out)
        for k in KINDS:
            stats[f"{k}_received"] = 0
        arrivals = {k: np.zeros_like(species[k]["alive"]) for k in KINDS}
        for path in sorted(self.inbox.glob("*.npz")):
            try:
                with np.load(path) as d:
                    origin = int(d["origin"])
                    for k in KINDS:
                        stats[f"{k}_received"] += self._arrive(species[k], arrivals[k], d, k, origin)
                path.unlink()
            except (OSError, ValueError, EOFError):
                continue  # half-copied by the relay, or being moved: it will still be there next time
        for k in KINDS:
            s = species[k]
            n = max(int(s["alive"].sum()), 1)
            stats[f"{k}_alive"] = int(s["alive"].sum())
            stats[f"{k}_immigrant_share"] = float((s["alive"] & (s["tag"] != 0)).sum() / n)
            for g, name in enumerate(("speed", "sight", "breed")):
                stats[f"{k}_{name}"] = float(s["genes"][..., g][s["alive"]].mean()) if s["alive"].any() else 0.0
        new = {k: getattr(state, k)._replace(**{f: jnp.asarray(species[k][f]) for f in species[k]}) for k in KINDS}
        env_state = runner.env_state._replace(**new)
        prey_obs, pred_obs = self.observe(env_state)
        forget = {}
        for k, h in (("prey", runner.prey_h), ("pred", runner.pred_h)):
            if h is not None:  # whoever now holds a changed slot starts with an empty memory
                forget[f"{k}_h"] = jnp.where(jnp.asarray((cleared[k] | arrivals[k])[..., None]), 0.0, h)
        return runner._replace(env_state=env_state, prey_obs=prey_obs, pred_obs=pred_obs, **forget), stats

    def _arrive(self, s: dict, arrived: np.ndarray, d, k: str, origin: int) -> int:
        count = 0
        worlds = s["alive"].shape[0]
        for i in range(len(d[f"{k}_energy"])):
            e = int(self.rng.integers(worlds))
            free = np.nonzero(~s["alive"][e] & ~arrived[e])[0]
            if free.size == 0:
                continue  # that world is full: this migrant is lost, as it would be on arrival at a crowded shore
            slot = int(free[0])
            s["pos"][e, slot] = (int(self.rng.integers(self.strip)), int(d[f"{k}_pos"][i, 1]) % self.cfg.size)
            s["energy"][e, slot] = d[f"{k}_energy"][i]
            s["genes"][e, slot] = d[f"{k}_genes"][i]
            s["gen"][e, slot] = d[f"{k}_gen"][i]
            s["tag"][e, slot] = d[f"{k}_tag"][i] if d[f"{k}_tag"][i] != 0 else origin + 1
            s["alive"][e, slot] = True
            arrived[e, slot] = True
            count += 1
        return count


def log_row(log, update: int, stats: dict) -> None:
    log.write(json.dumps({"update": update, **stats}) + "\n")
    log.flush()
