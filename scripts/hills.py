"""Stage 8a test: do animals use the terrain? python -m scripts.hills --run-dir runs/NAME [--evolved]

Evaluates the run's latest brains on the same seeds three ways: as trained, with the predators'
height channel zeroed, and with the prey's height channel zeroed. Hills still hide and still cost
energy when the channel is zeroed; only the animal's knowledge of the slopes is taken away.
Pass: predators catch at least 10% less, or prey live at least 10% less long, without it.
Also reports how much of each species' time is spent on high ground (above mid height), for the
trained brains and for random ones. Writes hills.json in the run directory.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_brains
from agents.evaluate import evaluate
from agents.networks import init_params
from agents.policies import forget_after, network_policy, pair_act, pair_memory, random_policy
from scripts.eval import EVAL_SEED, load_world_config
from simulations import evo
from simulations.worlds import make_net, obs_channels

DROP_TARGET = 0.10


class flat_eyes:
    """A policy that sees everything except the height channel (after the signal channel, before
    the brain gene)."""

    def __init__(self, policy, channel: int):
        self.policy, self.channel, self.memory = policy, channel, getattr(policy, "memory", 0)

    def __call__(self, key, obs):
        return self.policy(key, obs.at[..., self.channel].set(0))

    def step(self, key, obs, h):
        return self.policy.step(key, obs.at[..., self.channel].set(0), h)


def height_channel(cfg) -> int:
    return evo.OBS_CHANNELS + int(cfg.signals)


def high_ground(world, cfg, prey_policy, pred_policy, key, episodes: int, pool=None) -> dict:
    """Share of living animal-steps spent above mid height, per species."""
    hmap = jnp.asarray(evo.height_map(cfg)) > 0.5

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def count(sp):
            on = hmap[sp.pos[..., 1], sp.pos[..., 0]] & sp.alive
            return jnp.stack([on.sum(), sp.alive.sum()]).astype(jnp.float32)

        def body(carry, k):
            state, acc, mem = carry
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            acc = (acc[0] + count(state.prey), acc[1] + count(state.pred))
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            return (res.state, acc, forget_after(mem, res)), None

        mem = pair_memory(prey_policy, pred_policy, state)
        zero = jnp.zeros(2, jnp.float32)
        (_, acc, _), _ = jax.lax.scan(body, (state, (zero, zero), mem), jax.random.split(k_steps, cfg.max_steps))
        return acc

    prey, pred = (np.asarray(a) for a in run(key))
    return {"prey": float(prey[0] / max(prey[1], 1)), "pred": float(pred[0] / max(pred[1], 1)),
            "map": float(np.asarray(hmap).mean())}


def verdict(trained: dict, pred_blind: dict, prey_blind: dict) -> dict:
    def drop(full, less):
        return (full - less) / full if full > 0 else 0.0
    out = {"pred_drop": drop(trained["catches_per_pred_step"], pred_blind["catches_per_pred_step"]),
           "prey_drop": drop(trained["prey_lifetime"], prey_blind["prey_lifetime"])}
    out["pass"] = bool(out["pred_drop"] >= DROP_TARGET or out["prey_drop"] >= DROP_TARGET)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true", help="start from the run's latest evolved bodies")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    if not getattr(wcfg, "hills", False):
        print(f"{run_dir} is not a hills world")
        return 1
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    ch = height_channel(wcfg)
    pool, bodies = None, "founders"
    if args.evolved:
        from scripts.league import evolved_pool
        pool, bodies = evolved_pool(run_dir)
        if pool is None:
            print(f"no gene snapshots in {run_dir}")
            return 1
    key = jax.random.PRNGKey(EVAL_SEED)

    def ev(prey, pred):
        return evaluate(wcfg, prey, pred, key, args.episodes, world, pool=pool)

    trained = ev(prey_pi, pred_pi)
    pred_blind = ev(prey_pi, flat_eyes(pred_pi, ch))
    prey_blind = ev(flat_eyes(prey_pi, ch), pred_pi)
    out = {"update": meta.get("update"), "bodies": bodies, "channel": ch,
           "trained": trained, "pred_blind": pred_blind, "prey_blind": prey_blind,
           **verdict(trained, pred_blind, prey_blind),
           "high_ground": {"trained": high_ground(world, wcfg, prey_pi, pred_pi, key, args.episodes, pool),
                           "random": high_ground(world, wcfg, random_policy, random_policy, key, args.episodes, pool)}}
    name = "hills_evolved.json" if args.evolved else "hills.json"
    (run_dir / name).write_text(json.dumps(out, indent=1))
    print(f"bodies: {bodies}")
    print(f"predators without the height channel catch {100 * out['pred_drop']:.1f}% less; "
          f"prey without it live {100 * out['prey_drop']:.1f}% less long (bar {100 * DROP_TARGET:.0f}%): "
          f"{'PASS' if out['pass'] else 'FAIL'}")
    hg = out["high_ground"]
    print(f"time on high ground (map {100 * hg['trained']['map']:.0f}%): trained prey {100 * hg['trained']['prey']:.0f}%, "
          f"predators {100 * hg['trained']['pred']:.0f}%; random prey {100 * hg['random']['prey']:.0f}%, "
          f"predators {100 * hg['random']['pred']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
