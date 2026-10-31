"""Stage 6b test: do animals use the signal channel, does it mean anything, and do listeners act on it?

Reports, for each species of a trained run, how often animals signal and the mutual information
between signalling and whether the other species is within 2 cells. Then evaluates the predators
twice on the same seeds, once as trained and once with the signal channel zeroed in what they see.
Pass: predators catch at least 10% less without it. Writes signals.json in the run directory.
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
from agents.policies import forget_after, network_policy, pair_act, pair_memory
from scripts.eval import EVAL_SEED, load_world_config
from simulations.predprey import NUM_ACTIONS
from simulations import evo
from simulations.worlds import make_net, obs_channels

NEAR = 2  # cells, wrapped Chebyshev
DROP_TARGET = 0.10
RATE_RANGE = (0.02, 0.98)
MI_TARGET = 0.05  # bits
SIG = evo.OBS_CHANNELS  # the signal channel


def mutual_information(table) -> float:
    """Mutual information in bits of a 2-D contingency table of counts."""
    t = np.asarray(table, np.float64)
    n = t.sum()
    if n == 0:
        return 0.0
    p = t / n
    outer = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / outer[nz])).sum())


class muted:
    """A policy that sees everything except the signal channel (right after the base channels;
    the brain gene, when on, comes after it)."""

    def __init__(self, policy):
        self.policy, self.memory = policy, getattr(policy, "memory", 0)

    def __call__(self, key, obs):
        return self.policy(key, obs.at[..., SIG].set(0))

    def step(self, key, obs, h):
        return self.policy.step(key, obs.at[..., SIG].set(0), h)


class shuffled(muted):
    """A policy that hears the signals of the next world over (episode axis rolled by one): the same
    signal statistics, none of the meaning. A fairer control than silence, which the brain never met."""

    def _swap(self, obs):
        return obs.at[..., SIG].set(jnp.roll(obs[..., SIG], 1, axis=0))

    def __call__(self, key, obs):
        return self.policy(key, self._swap(obs))

    def step(self, key, obs, h):
        return self.policy.step(key, self._swap(obs), h)


def _near(a, b, size):
    """(Na,) bool: some living animal of `b` is within NEAR cells of each animal of `a`."""
    d = jnp.abs(a.pos[:, None, :] - b.pos[None, :, :])
    d = jnp.minimum(d, size - d).max(-1)
    return ((d <= NEAR) & b.alive[None, :]).any(-1)


def signal_stats(world, cfg, prey_policy, pred_policy, key, episodes: int, pool=None) -> dict:
    """Per species: signal rate over living animal-steps, the 2x2 table
    [other species near (no, yes)] x [signal (no, yes)], and its mutual information."""

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def table(sp, near, a):
            sig = a >= NUM_ACTIONS
            cell = near.astype(jnp.int32) * 2 + sig.astype(jnp.int32)
            return jnp.zeros(4, jnp.float32).at[cell.reshape(-1)].add(sp.alive.reshape(-1).astype(jnp.float32))

        def body(carry, k):
            state, acc, mem = carry
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            prey_near = jax.vmap(lambda s: _near(s.prey, s.pred, cfg.size))(state)
            pred_near = jax.vmap(lambda s: _near(s.pred, s.prey, cfg.size))(state)
            acc = (acc[0] + table(state.prey, prey_near, prey_a), acc[1] + table(state.pred, pred_near, pred_a))
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            return (res.state, acc, forget_after(mem, res)), None

        acc0 = (jnp.zeros(4, jnp.float32), jnp.zeros(4, jnp.float32))
        (_, acc, _), _ = jax.lax.scan(body, (state, acc0, pair_memory(prey_policy, pred_policy, state)),
                                      jax.random.split(k_steps, cfg.max_steps))
        return acc

    out = {}
    for who, flat in zip(("prey", "pred"), jax.device_get(run(key))):
        t = np.asarray(flat, np.float64).reshape(2, 2)
        out[who] = {"rate": float(t[:, 1].sum() / max(t.sum(), 1.0)), "table": t,
                    "mi_bits": max(mutual_information(t), 0.0)}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true",
                   help="start from the run's latest evolved bodies (genes snapshot) instead of founders")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    if not getattr(wcfg, "signals", False):
        print(f"{run_dir.name}: this world has no signals")
        return 2
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    key = jax.random.PRNGKey(EVAL_SEED)
    pool, bodies = None, "founders"
    if args.evolved:
        from scripts.league import evolved_pool
        pool, bodies = evolved_pool(run_dir)
        if pool is None:
            print(f"no gene snapshots in {run_dir}")
            return 1

    stats = signal_stats(world, wcfg, prey_pi, pred_pi, key, args.episodes, pool)
    catches = evaluate(wcfg, prey_pi, pred_pi, key, args.episodes, world, pool=pool)["catches_per_pred_step"]
    catches_muted = evaluate(wcfg, prey_pi, muted(pred_pi), key, args.episodes, world, pool=pool)["catches_per_pred_step"]
    catches_shuffled = evaluate(wcfg, prey_pi, shuffled(pred_pi), key, args.episodes, world, pool=pool)["catches_per_pred_step"]
    drop = 1 - catches_muted / catches if catches > 0 else 0.0
    drop_shuffled = 1 - catches_shuffled / catches if catches > 0 else 0.0
    out = {"update": meta.get("update"), "bodies": bodies, "catches": catches, "catches_muted": catches_muted, "drop": drop,
           "catches_shuffled": catches_shuffled, "drop_shuffled": drop_shuffled,
           "pass_shuffled": bool(drop_shuffled >= DROP_TARGET)}
    for who in ("prey", "pred"):
        s = stats[who]
        out[who] = {"rate": s["rate"], "mi_bits": s["mi_bits"], "table": s["table"].tolist(),
                    "used": RATE_RANGE[0] <= s["rate"] <= RATE_RANGE[1], "meaningful": s["mi_bits"] >= MI_TARGET}
    out["pass"] = bool(drop >= DROP_TARGET)
    (run_dir / ("signals_evolved.json" if args.evolved else "signals.json")).write_text(json.dumps(out, indent=1))
    print(f"bodies: {bodies}")
    for who in ("prey", "pred"):
        print(f"{who}: signal rate {out[who]['rate']:.3f}, information about the other species within {NEAR} cells "
              f"{out[who]['mi_bits']:.3f} bits")
    print(f"predator catches per step {catches:.4f}, with the signal channel blanked {catches_muted:.4f} "
          f"(drop {drop:.1%}; pass at {DROP_TARGET:.0%}) -> {'PASS' if out['pass'] else 'FAIL'}")
    print(f"with signals from another world {catches_shuffled:.4f} (drop {drop_shuffled:.1%}) -> "
          f"{'PASS' if out['pass_shuffled'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
