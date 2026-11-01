"""Live feed for the phone viewer (viz/dashboard/live.html): short clips published while a run trains.

Every few updates, the trainer takes its first world as it stands and plays a few steps forward on a copy, with the brains of that
moment. So a clip is the real world and real brains, a few seconds old, and training itself is not disturbed. To stay light on a
phone, a clip holds only the animals inside one window of the map, plus a density map of the whole world.

Files: <run>/live/index.json names the latest clips, each <run>/live/clip_NNNNNNN.json. Old clips are deleted.
"""
from __future__ import annotations

import base64
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.policies import forget_after, network_policy, pair_act, pair_memory

KEEP = 12  # clips kept on disk
GRID = 32  # cells of the density map per side


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(a.tobytes()).decode()


class LiveFeed:
    def __init__(self, run_dir, wcfg, world, net, every: int = 3, steps: int = 48, window: int = 96, seed: int = 0):
        self.dir, self.cfg, self.world, self.net = Path(run_dir) / "live", wcfg, world, net
        self.every, self.steps, self.window = every, steps, min(window, wcfg.size)
        self.dir.mkdir(exist_ok=True)
        self.key = jax.random.PRNGKey(seed + 4242)
        self.history: list = []
        self._play = jax.jit(self._play_fn)

    def _play_fn(self, state, prey_params, pred_params, key):
        prey_pi, pred_pi = network_policy(self.net, prey_params), network_policy(self.net, pred_params)
        cfg, world = self.cfg, self.world
        mem = pair_memory(prey_pi, pred_pi, jax.tree.map(lambda x: x[None], state))
        mem = jax.tree.map(lambda x: x[0] if x is not None else None, mem)

        def body(carry, k):
            s, m = carry
            kp, kq, ks = jax.random.split(k, 3)
            prey_obs, pred_obs = world.observe(cfg, s)
            prey_a, pred_a, m = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, s, m)
            res = world.advance(cfg, ks, s, prey_a, pred_a)
            row = (s.prey.pos, s.prey.energy, s.prey.alive, s.pred.pos, s.pred.energy, s.pred.alive, s.plant & ~s.bitten, res.info.catches)
            return (res.state, forget_after(m, res)), row

        return jax.lax.scan(body, (state, mem), jax.random.split(key, self.steps))[1]

    def publish(self, runner, update: int, extra: dict | None = None) -> None:
        if update % self.every:
            return
        self.key, k = jax.random.split(self.key)
        state = jax.tree.map(lambda x: x[0], runner.env_state)
        rows = jax.device_get(self._play(state, runner.prey.params, runner.pred.params, k))
        (prey_pos, prey_e, prey_alive, pred_pos, pred_e, pred_alive, plant, catches) = rows
        N, W = self.cfg.size, self.window
        ox, oy = self._busiest(prey_pos[0], prey_alive[0], pred_pos[0], pred_alive[0])
        clip = {"update": update, "steps": self.steps, "size": N, "window": W, "ox": ox, "oy": oy, "t0": int(state.t),
                "climate": getattr(self.cfg, "climate", ""), "winter_len": int(getattr(self.cfg, "winter_len", 0)),
                "stealth": int(getattr(self.cfg, "stealth_radius", 0)), "catches": int(np.sum(catches))}
        genes = jax.device_get((state.prey.genes, state.pred.genes))
        for name, pos, energy, alive, g in (("prey", prey_pos, prey_e, prey_alive, genes[0]), ("pred", pred_pos, pred_e, pred_alive, genes[1])):
            clip[name] = self._kind(pos, energy, alive, g, ox, oy, N, W)
        rx = (np.arange(W) + ox) % N
        ry = (np.arange(W) + oy) % N
        clip["plant"] = [_b64(np.packbits(plant[t][np.ix_(ry, rx)])) for t in range(0, self.steps, 16)]
        dens = {}
        for name, pos, alive in (("prey", prey_pos[0], prey_alive[0]), ("pred", pred_pos[0], pred_alive[0])):
            cell = np.clip((pos[alive] * GRID) // N, 0, GRID - 1)
            d = np.zeros((GRID, GRID), np.uint16)
            np.add.at(d, (cell[:, 1], cell[:, 0]), 1)
            dens[name] = _b64(d.astype(np.uint16))
        clip["density"], clip["grid"] = dens, GRID
        g0, g1 = genes[0][prey_alive[0]], genes[1][pred_alive[0]]
        stats = {"update": update, "grazers": int(prey_alive[0].sum()), "hunters": int(pred_alive[0].sum()), "plants": round(float(plant[0].mean()), 3),
                 "catches": clip["catches"],
                 **{f"grazer_{n}": round(float(g0[:, i].mean()), 3) if len(g0) else 0.0 for i, n in enumerate(("speed", "sight", "breed"))},
                 **{f"hunter_{n}": round(float(g1[:, i].mean()), 3) if len(g1) else 0.0 for i, n in enumerate(("speed", "sight", "breed"))}, **(extra or {})}
        clip["stats"] = stats
        self.history = (self.history + [stats])[-200:]
        name = f"clip_{update:07d}.json"
        self._write(name, clip)
        clips = sorted(p.name for p in self.dir.glob("clip_*.json"))
        for old in clips[:-KEEP]:
            (self.dir / old).unlink(missing_ok=True)
        self._write("index.json", {"latest": name, "clips": clips[-KEEP:], "history": self.history, "climate": clip["climate"], "update": update})

    def _busiest(self, prey_pos, prey_alive, pred_pos, pred_alive) -> tuple[int, int]:
        """Where to look: the corner of the window that holds the most hunters, so there is something to watch."""
        N, W = self.cfg.size, self.window
        if W >= N:
            return 0, 0
        pos = pred_pos[pred_alive] if pred_alive.any() else prey_pos[prey_alive]
        if not len(pos):
            return 0, 0
        cells = (pos * 8) // N
        counts = np.zeros((8, 8), int)
        np.add.at(counts, (cells[:, 1], cells[:, 0]), 1)
        y, x = np.unravel_index(counts.argmax(), counts.shape)
        return int((x * N // 8 + N // 16 - W // 2) % N), int((y * N // 8 + N // 16 - W // 2) % N)

    def _write(self, name: str, data: dict) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        os.chmod(tmp, 0o666)
        os.replace(tmp, self.dir / name)

    @staticmethod
    def _kind(pos, energy, alive, genes, ox, oy, N, W) -> dict:
        rel = (pos - np.array([ox, oy])) % N
        inside = (rel[..., 0] < W) & (rel[..., 1] < W) & alive
        slots = np.nonzero(inside.any(0))[0]
        T = pos.shape[0]
        out = {"n": int(len(slots)), "slots": [int(s) for s in slots]}
        out["x"] = _b64(rel[:, slots, 0].astype(np.int16))  # (T, n): cells from the window's corner
        out["y"] = _b64(rel[:, slots, 1].astype(np.int16))
        out["e"] = _b64(np.clip(energy[:, slots], 0, 255).astype(np.uint8))
        out["on"] = _b64(np.packbits(alive[:, slots]))
        out["genes"] = _b64((np.clip(genes[slots][:, :4], 0, 1) * 255).astype(np.uint8))
        return out
