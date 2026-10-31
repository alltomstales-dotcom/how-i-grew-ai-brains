"""Record one episode for the 3-D viewer: python -m scripts.episode3d --run-dir runs/NAME [--evolved]

Plays the run's latest prey and predator brains for one episode (fixed seed) and writes
<run>/episode3d.json, which viz/dashboard/world3d.html replays. Each animal gets an id that
lasts its lifetime; a slot reused by a newborn gets a new id. Plants are packed as a bitmask per
step, and plant toughness every TOUGH_EVERY steps.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import jax
import numpy as np

from agents.checkpoint import load_brains
from agents.networks import init_params
from agents.policies import forget_after, network_policy, pair_act, pair_memory
from scripts.eval import EVAL_SEED, load_world_config
from simulations.worlds import make_net, obs_channels

TOUGH_EVERY = 50
GENES = ("speed", "sight", "breed", "size", "brain")


def _bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes()).decode()


def _bytes(values: np.ndarray) -> str:
    return base64.b64encode(np.clip(values.reshape(-1) * 255, 0, 255).astype(np.uint8).tobytes()).decode()


class Tracker:
    """Gives each animal an id for its lifetime, from slot, alive and generation."""

    def __init__(self, kind: str, animals: dict):
        self.kind, self.animals = kind, animals
        self.slot_id: dict[int, int] = {}
        self.prev = {}

    def frame(self, t: int, sp) -> list:
        alive, pos, energy = np.asarray(sp.alive), np.asarray(sp.pos), np.asarray(sp.energy)
        # Stage 1 (predprey) animals have no genes: show founder-like bodies, and treat a slot that was
        # empty last step as a newborn.
        genes = np.asarray(sp.genes) if hasattr(sp, "genes") else np.tile([[0.5, 0.5, 0.5, 0.5, 0.0]], (len(alive), 1))
        gen = np.asarray(sp.gen) if hasattr(sp, "gen") else np.where(alive, 0, -1)
        rows, seen = [], set()
        for s in np.nonzero(alive)[0]:
            s = int(s)
            uid = self.slot_id.get(s)
            if uid is None or self.prev.get(s) != int(gen[s]):  # a newborn in this slot
                uid = len(self.animals)
                self.animals[uid] = {"kind": self.kind, "born": t, "died": None, "gen": int(gen[s]),
                                     **{g: round(float(v), 3) for g, v in zip(GENES, genes[s])}}
                self.slot_id[s] = uid
            self.prev[s] = int(gen[s])
            seen.add(uid)
            rows.append([uid, int(pos[s, 0]), int(pos[s, 1]), round(float(energy[s]), 1)])
        for s, uid in list(self.slot_id.items()):
            if uid not in seen:
                if self.animals[uid]["died"] is None:
                    self.animals[uid]["died"] = t
                del self.slot_id[s]
                self.prev.pop(s, None)
        return rows


def record(world, wcfg, prey_pi, pred_pi, key, pool=None, max_steps=None) -> dict:
    @jax.jit
    def advance(state, mem, k):
        kp, kq = jax.random.split(k)
        prey_obs, pred_obs = world.observe(wcfg, state)
        prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
        res = world.advance(wcfg, jax.random.fold_in(k, 1), state, prey_a, pred_a)
        return res.state, forget_after(mem, res), res.done

    k_reset, key = jax.random.split(key)
    state = world.reset(wcfg, k_reset) if pool is None else world.reset(wcfg, k_reset, pool)
    mem = pair_memory(prey_pi, pred_pi, state)
    animals: dict = {}
    prey_t, pred_t = Tracker("prey", animals), Tracker("pred", animals)
    steps, meals = [], []
    last_pred_energy = None
    for t, k in enumerate(jax.random.split(key, max_steps or wcfg.max_steps)):
        s = jax.device_get(state)
        pred_rows = pred_t.frame(t, s.pred)
        plant = np.asarray(s.plant) if hasattr(s, "plant") else np.asarray(s.grass) == 0  # predprey: grass regrow timer
        step = {"prey": prey_t.frame(t, s.prey), "pred": pred_rows, "plant": _bits(plant)}
        if getattr(wcfg, "fickle", False):  # Stage 7c: which plant kind feeds right now (hidden from the animals)
            step["good"] = int(s.good)
        if t % TOUGH_EVERY == 0:
            step["tough"] = _bytes(np.asarray(s.plant_genes)[..., 1] if hasattr(s, "plant_genes") else np.zeros(plant.shape))
        # A predator whose energy jumped by more than a step's upkeep ate this step.
        energy = {r[0]: r[3] for r in pred_rows}
        if last_pred_energy is not None:
            step["ate"] = [u for u, e in energy.items() if e - last_pred_energy.get(u, e) > 5]
        last_pred_energy = energy
        steps.append(step)
        state, mem, done = advance(state, mem, k)
        if bool(done):
            break
    end = len(steps)
    for a in animals.values():
        a["died"] = end if a["died"] is None else a["died"]
    final = jax.device_get(state)
    terrain = np.asarray(final.terrain) if hasattr(final, "terrain") else np.zeros((wcfg.size, wcfg.size))
    out = {"size": wcfg.size, "steps": steps, "animals": animals, "terrain": _bytes(terrain / 2.0),
            "flank": bool(getattr(wcfg, "flank", False)), "pack_size": int(getattr(wcfg, "pack_size", 0))}
    if getattr(wcfg, "hills", False):  # Stage 8a: the real height map, which the viewer draws
        from simulations.evo import height_map
        out["height"] = _bytes(height_map(wcfg))
    if getattr(wcfg, "thickets", 0) > 0:  # Stage 8b: where the thickets are, which the viewer draws
        from simulations.evo import thicket_map
        out["cover"] = _bits(thicket_map(wcfg))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--evolved", action="store_true", help="start from the run's latest evolved bodies")
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--untrained", action="store_true", help="play fresh random brains (writes episode3d_untrained.json)")
    p.add_argument("--tries", type=int, default=1, help="record this many seeds and keep the longest episode")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    pool, bodies = None, "founders"
    if args.evolved:
        from scripts.league import evolved_pool
        pool, bodies = evolved_pool(run_dir)
        if pool is None:
            print(f"no gene snapshots in {run_dir}")
            return 1
    if args.untrained:  # the same network, freshly initialised: behaviour before any learning
        prey_params = init_params(net, jax.random.PRNGKey(101), wcfg.view, obs_channels(world, wcfg))
        pred_params = init_params(net, jax.random.PRNGKey(202), wcfg.view, obs_channels(world, wcfg))
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    best, seed = None, args.seed
    for s in range(args.seed, args.seed + max(args.tries, 1)):  # episodes end when a species dies out
        out = record(world, wcfg, prey_pi, pred_pi, jax.random.PRNGKey(s), pool)
        # Fickle worlds: an episode with a flip in it beats any without, since the flip is what the replay is for.
        flips = lambda o: sum(a.get("good") != b.get("good") for a, b in zip(o["steps"], o["steps"][1:]))
        score = lambda o: (min(flips(o), 1), len(o["steps"]))
        if best is None or score(out) > score(best):
            best, seed = out, s
        if len(best["steps"]) >= wcfg.max_steps:
            break
    out = best
    out.update(run=run_dir.name, world=config.get("world_name"), update=0 if args.untrained else meta.get("update"),
               bodies=bodies, seed=seed)
    path = run_dir / ("episode3d_untrained.json" if args.untrained else "episode3d.json")
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path}: {len(out['steps'])} steps, {len(out['animals'])} animals, {path.stat().st_size // 1024} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
