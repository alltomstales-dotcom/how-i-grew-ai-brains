"""Stage 8b test: do hunters ambush? python -m scripts.ambush --run-dir runs/NAME [--evolved]

Three questions, all of which must come out yes (bars fixed before any run):
  * Do trained hunters sit in thickets more than the map offers (at least 1.3x the map's share)?
  * Do they stand still more than random hunters (at least 1.5x)?
  * Does hiding pay? The same brains play the same seeds with thickets no longer hiding predators.
    They must catch at least 10% less.
Writes ambush.json (ambush_evolved.json) in the run directory.
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
from agents.evaluate import evaluate
from agents.networks import init_params
from agents.policies import chase_policy, forget_after, network_policy, pair_act, pair_memory, random_policy
from scripts.eval import EVAL_SEED, load_world_config
from simulations import evo
from simulations.worlds import make_net, obs_channels

COVER_TARGET, STILL_TARGET, DROP_TARGET = 1.3, 1.5, 0.10


def hunter_habits(world, cfg, prey_policy, pred_policy, key, episodes: int, pool=None) -> dict:
    """Over living predator-steps: share spent in a thicket, share standing still, standing still while
    in a thicket, and energy burned per step (upkeep, speed and sight genes, walking)."""
    cover = jnp.asarray(evo.thicket_map(cfg))

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
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            before, after = state.pred, res.state.pred
            both = before.alive & after.alive
            still = both & (before.pos == after.pos).all(-1)
            inside = cover[before.pos[..., 1], before.pos[..., 0]] & before.alive
            d = jnp.abs(after.pos - before.pos)
            cells = jnp.minimum(d, cfg.size - d).sum(-1)
            burn = (cfg.pred_base_cost + cfg.speed_cost * before.genes[..., 0] ** 2
                    + cfg.sight_cost * (evo.sight_radius(before.genes[..., 1]) - 1) / 3 + cfg.move_cost * cells)
            acc = acc + jnp.stack([before.alive.sum(), inside.sum(), both.sum(), still.sum(),
                                   (still & inside).sum(), (both & inside).sum(), (burn * both).sum()]).astype(jnp.float32)
            return (res.state, acc, forget_after(mem, res)), None

        init = (state, jnp.zeros(7, jnp.float32), pair_memory(prey_policy, pred_policy, state))
        (_, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    alive, inside, both, still, still_in, both_in, burned = (float(v) for v in np.asarray(run(key)))
    return {"in_cover": inside / max(alive, 1), "still": still / max(both, 1),
            "still_in_cover": still_in / max(both_in, 1), "pred_steps": alive, "burn_per_step": burned / max(both, 1)}


def verdict(trained: dict, random: dict, map_share: float, full: float, exposed: float) -> dict:
    out = {"cover_ratio": trained["in_cover"] / max(map_share, 1e-9),
           "still_ratio": trained["still"] / max(random["still"], 1e-9),
           "exposed_drop": (full - exposed) / full if full > 0 else 0.0}
    out["cover_pass"] = bool(out["cover_ratio"] >= COVER_TARGET)
    out["still_pass"] = bool(out["still_ratio"] >= STILL_TARGET)
    out["hiding_pass"] = bool(out["exposed_drop"] >= DROP_TARGET)
    out["pass"] = out["cover_pass"] and out["still_pass"] and out["hiding_pass"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true", help="start from the run's latest evolved bodies")
    p.add_argument("--chaser", action="store_true", help="compare energy per catch with the one-line chaser (writes ambush_chaser.json)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    if getattr(wcfg, "thickets", 0) <= 0:
        print(f"{run_dir} has no thickets")
        return 1
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    pool, bodies = None, "founders"
    if args.evolved:
        from scripts.league import evolved_pool
        pool, bodies = evolved_pool(run_dir)
        if pool is None:
            print(f"no gene snapshots in {run_dir}")
            return 1
    key = jax.random.PRNGKey(EVAL_SEED)
    if args.chaser:
        chaser = chase_policy(world.PREY_CHANNEL)
        out = {}
        for name, pi in (("trained", pred_pi), ("chaser", chaser)):
            ev = evaluate(wcfg, prey_pi, pi, key, args.episodes, world, pool=pool)
            habits = hunter_habits(world, wcfg, prey_pi, pi, key, args.episodes, pool)
            out[name] = {**ev, **habits, "energy_per_catch": habits["burn_per_step"] / max(ev["catches_per_pred_step"], 1e-9)}
            print(f"{name}: catches {ev['catches_per_pred_step']:.4f} per step, burns {habits['burn_per_step']:.3f} energy per step, "
                  f"{out[name]['energy_per_catch']:.0f} energy per catch, still {100 * habits['still']:.0f}%, episode {ev['mean_episode_len']:.0f} steps")
        (run_dir / "ambush_chaser.json").write_text(json.dumps(out, indent=1))
        return 0
    exposed_cfg = dataclasses.replace(wcfg, cover_hides_pred=False)
    full = evaluate(wcfg, prey_pi, pred_pi, key, args.episodes, world, pool=pool)
    exposed = evaluate(exposed_cfg, prey_pi, pred_pi, key, args.episodes, world, pool=pool)
    trained = hunter_habits(world, wcfg, prey_pi, pred_pi, key, args.episodes, pool)
    rand = hunter_habits(world, wcfg, random_policy, random_policy, key, args.episodes, pool)
    map_share = float(evo.thicket_map(wcfg).mean())
    out = {"update": meta.get("update"), "bodies": bodies, "map_share": map_share, "trained": trained, "random": rand,
           "full": full, "exposed": exposed,
           **verdict(trained, rand, map_share, full["catches_per_pred_step"], exposed["catches_per_pred_step"])}
    (run_dir / ("ambush_evolved.json" if args.evolved else "ambush.json")).write_text(json.dumps(out, indent=1))
    print(f"bodies: {bodies}")
    print(f"in a thicket {100 * trained['in_cover']:.0f}% of the time (map {100 * map_share:.0f}%, random {100 * rand['in_cover']:.0f}%): "
          f"{out['cover_ratio']:.2f}x, bar {COVER_TARGET}x")
    print(f"standing still {100 * trained['still']:.0f}% (random {100 * rand['still']:.0f}%): {out['still_ratio']:.2f}x, bar {STILL_TARGET}x")
    print(f"catches {full['catches_per_pred_step']:.4f}, seen in thickets {exposed['catches_per_pred_step']:.4f}: "
          f"{100 * out['exposed_drop']:.1f}% less, bar {100 * DROP_TARGET:.0f}%")
    print("PASS" if out["pass"] else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
