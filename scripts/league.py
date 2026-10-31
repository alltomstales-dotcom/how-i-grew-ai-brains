"""Elo ratings for a run's saved brains: python -m scripts.league --run-dir runs/NAME

Every saved predator brain plays every saved prey brain (fixed seed). In an evolving world each
side wears the bodies it trained with: predator brain i gets the predator genes saved with it,
prey brain j the prey (and plant) genes saved with it. Otherwise all play in founder bodies.
A match's score is the share of prey eaten: the predator's "win" share. Ratings are
fitted so that score = 1 / (1 + 10 ** ((prey - pred) / 400)), centred on 1000.
Within a species, a rising rating means later brains beat the opponents earlier
brains could not; a flat or falling one means the species is going in circles.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_brains
from agents.networks import init_params
from simulations.worlds import make_net, obs_channels
from agents.policies import forget_after, network_policy, pair_act, pair_memory
from scripts.eval import load_world_config

POOL_ROWS = 4096  # gene pools are padded to this many rows, so one compiled match serves all
PROGRESS_TARGET = 100.0  # Elo gained from the first saved brain to the last
LEAGUE_SEED = 777


def fit_elo(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scores[i, j]: predator i's score against prey j. Least squares in logit space."""
    s = np.clip(np.asarray(scores, np.float64), 1e-3, 1 - 1e-3)
    n_pred, n_prey = s.shape
    rows, target = [], []
    for i in range(n_pred):
        for j in range(n_prey):
            row = np.zeros(n_pred + n_prey)
            row[i], row[n_pred + j] = 1.0, -1.0
            rows.append(row)
            target.append(400 * np.log10(s[i, j] / (1 - s[i, j])))
    r = np.linalg.lstsq(np.array(rows), np.array(target), rcond=None)[0]
    r += 1000 - r.mean()  # the minimum-norm solution fixes the free offset; centre it
    return r[:n_pred], r[n_pred:]


def progress_verdict(elo: np.ndarray) -> bool:
    return bool(len(elo) >= 2 and elo[-1] - elo[0] >= PROGRESS_TARGET)


def evolved_pool(run_dir: Path):
    """(gene pool, snapshot file name) from a run's latest gene snapshot, or (None, None) if it has none."""
    ckpts = Path(run_dir) / "checkpoints"
    latest = ckpts / "genes_latest.npz"
    snaps = sorted(ckpts.glob("genes_*.npz"))
    snap = latest if latest.exists() else (snaps[-1] if snaps else None)
    if snap is None:
        return None, None
    with np.load(snap) as d:
        era = {k: d[k] for k in d.files}
    return match_pool(era, era), snap.name


def match_pool(prey_era: dict, pred_era: dict):
    """Prey and plant genes from the prey brain's snapshot, predator genes from the predator's."""
    from simulations import evo

    def pad(genes, width, gen=None):
        genes = evo.pad_genes(genes, width)[:POOL_ROWS]
        out, w, g = np.zeros((POOL_ROWS, width), np.float32), np.zeros(POOL_ROWS, np.float32), np.zeros(POOL_ROWS, np.int32)
        out[:len(genes)], w[:len(genes)] = genes, 1.0
        if gen is not None:
            g[:len(genes)] = np.asarray(gen)[:POOL_ROWS]
        return jnp.asarray(out), jnp.asarray(g), jnp.asarray(w)

    prey_g, prey_gen, prey_w = pad(prey_era["prey_genes"], evo.ANIMAL_GENES, prey_era["prey_gen"])
    pred_g, pred_gen, pred_w = pad(pred_era["pred_genes"], evo.ANIMAL_GENES, pred_era["pred_gen"])
    plant_g, _, plant_w = pad(prey_era["plant_genes"], evo.PLANT_GENES)
    return evo.GenePool(prey_g, prey_gen, prey_w, pred_g, pred_gen, pred_w, plant_g, plant_w)


def make_match(world, cfg, net, episodes: int):
    @jax.jit
    def play(prey_params, pred_params, key, pool=None):
        prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def body(carry, k):
            state, eaten, born, mem = carry
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            mem = forget_after(mem, res)
            return (res.state, eaten + res.info.catches.sum(), born + res.info.prey_births.sum(), mem), None

        zero = jnp.zeros((), jnp.float32)
        init = (state, zero, zero, pair_memory(prey_pi, pred_pi, state))
        (_, eaten, born, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return eaten / (cfg.init_prey * episodes + born)

    return play


def pick(paths: list[Path], n: int) -> list[Path]:
    if len(paths) <= n:
        return paths
    return [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, n).round().astype(int))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--max-brains", type=int, default=6)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    world, wcfg = load_world_config(json.loads((run_dir / "config.json").read_text()))
    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.msgpack"),
                   key=lambda q: int(re.search(r"(\d+)", q.name).group(1)))
    ckpts = pick(ckpts, args.max_brains)
    if not ckpts:
        print(f"no checkpoints in {run_dir}")
        return 1
    net = make_net(world, wcfg, json.loads((run_dir / "config.json").read_text())["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    brains = [load_brains(c, template) for c in ckpts]
    play = make_match(world, wcfg, net, args.episodes)
    key = jax.random.PRNGKey(LEAGUE_SEED)
    genes = [c.parent / c.name.replace("ckpt_", "genes_").replace(".msgpack", ".npz") for c in ckpts]
    evolved = hasattr(world, "gene_pool") and all(g.exists() for g in genes)
    eras = []
    if evolved:
        for g in genes:
            with np.load(g) as d:
                eras.append({k: d[k] for k in d.files})
    # scores[i, j]: predator brain i against prey brain j (brains[k] = (prey, pred, meta)).
    scores = np.array([[float(play(brains[j][0], brains[i][1], key, match_pool(eras[j], eras[i]) if evolved else None))
                        for j in range(len(brains))] for i in range(len(brains))])
    pred_elo, prey_elo = fit_elo(scores)
    updates = [int(b[2].get("update", 0)) for b in brains]
    out = {
        "updates": updates, "episodes": args.episodes, "bodies": "evolved" if evolved else "founder",
        "scores": scores.round(4).tolist(),
        "pred_elo": pred_elo.round(1).tolist(), "prey_elo": prey_elo.round(1).tolist(),
        "pred_progress": progress_verdict(pred_elo), "prey_progress": progress_verdict(prey_elo),
    }
    (run_dir / "league.json").write_text(json.dumps(out, indent=1))
    for name in ("pred", "prey"):
        elo = out[f"{name}_elo"]
        trail = "  ".join(f"u{u}:{e:.0f}" for u, e in zip(updates, elo))
        print(f"{name} Elo  {trail}  -> {'PROGRESS' if out[f'{name}_progress'] else 'NO PROGRESS'} "
              f"(target +{PROGRESS_TARGET:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
