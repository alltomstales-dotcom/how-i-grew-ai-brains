"""Stage 7c test: do animals learn within one life? python -m scripts.lifetime --run-dir runs/GRU --baseline runs/FF [--evolved]

In a fickle world, one plant kind feeds and the other poisons. Nothing shows which, and it flips at random.
Reports, for each run, the share of bites taken from the kind that feeds, by steps since the last flip.
Bars fixed before any run, for the memory run against its feed-forward baseline:
  * Soon after a flip has had time to be noticed (10-40 steps), the memory brain's good-bite share beats the
    feed-forward brain's by at least 0.10.
  * The memory brain recovers: its share at 10-40 steps beats its own share at 0-10 steps by at least 0.05.
Writes lifetime.json in the run directory, with the verdict when --baseline is given.
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
from simulations.worlds import make_net, obs_channels

class amnesiac:
    """A memory brain whose memory is wiped every step: the control for 'it is the memory that does it'."""

    def __init__(self, policy):
        self.policy, self.memory = policy, policy.memory

    def __call__(self, key, obs):
        return self.step(key, obs, jnp.zeros(obs.shape[:-3] + (self.memory,), jnp.float32))[0]

    def step(self, key, obs, h):
        return self.policy.step(key, obs, jnp.zeros_like(h))


EDGES = (0, 10, 40)  # bins of steps since a flip: 0-10, 10-40, 40+
BEAT_TARGET, RECOVER_TARGET = 0.10, 0.05


def bite_shares(world, cfg, prey_policy, pred_policy, key, episodes: int, pool=None) -> dict:
    edges = jnp.asarray(EDGES)

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
            bin_ = jnp.searchsorted(edges, state.since_flip, side="right") - 1
            acc = acc.at[bin_, 0].add(res.info.good_bites.astype(jnp.float32))
            acc = acc.at[bin_, 1].add(res.info.bad_bites.astype(jnp.float32))
            return (res.state, acc, forget_after(mem, res)), None

        init = (state, jnp.zeros((len(EDGES), 2), jnp.float32), pair_memory(prey_policy, pred_policy, state))
        (_, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    acc = np.asarray(run(key))
    names = ("0-10", "10-40", "40+")
    return {n: {"good": float(g), "bad": float(b), "share": float(g / max(g + b, 1.0))} for n, (g, b) in zip(names, acc)}


def next_bites(world, cfg, prey_policy, pred_policy, key, episodes: int, pool=None) -> dict:
    """Within one animal's life, with no flip in between: the share of next bites that are good, after a
    good bite and after a bad one. A brain that learns by tasting switches after a bad bite. A bite shows in
    the animal's energy over the step: a good one gains, a bad one loses more than upkeep ever costs."""

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)

        def body(carry, k):
            state, table, prev, mem = carry
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            a, b = state.prey, res.state.prey
            same = a.alive & b.alive & (a.gen == b.gen) & (res.state.since_flip[:, None] > 0)  # the same animal, no flip
            delta = b.energy - a.energy
            good, bad = same & (delta > 1.0), same & (delta < -cfg.poison)
            table = table.at[0, 0].add((good & (prev == 1)).sum()).at[0, 1].add((bad & (prev == 1)).sum())
            table = table.at[1, 0].add((good & (prev == 2)).sum()).at[1, 1].add((bad & (prev == 2)).sum())
            prev = jnp.where(same, prev, 0)
            prev = jnp.where(good, 1, jnp.where(bad, 2, prev))
            return (res.state, table, prev, forget_after(mem, res)), None

        init = (state, jnp.zeros((2, 2), jnp.float32), jnp.zeros(state.prey.alive.shape, jnp.int8),
                pair_memory(prey_policy, pred_policy, state))
        (_, table, _, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return table

    t = np.asarray(run(key))
    return {"after_good": float(t[0, 0] / max(t[0].sum(), 1.0)), "after_bad": float(t[1, 0] / max(t[1].sum(), 1.0)),
            "pairs_after_good": float(t[0].sum()), "pairs_after_bad": float(t[1].sum())}


def verdict(memory: dict, baseline: dict) -> dict:
    out = {"beat": memory["10-40"]["share"] - baseline["10-40"]["share"],
           "recover": memory["10-40"]["share"] - memory["0-10"]["share"]}
    out["beat_pass"] = bool(out["beat"] >= BEAT_TARGET)
    out["recover_pass"] = bool(out["recover"] >= RECOVER_TARGET)
    out["pass"] = out["beat_pass"] and out["recover_pass"]
    return out


def measure(run_dir: Path, episodes: int, evolved: bool) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    pool, bodies = None, "founders"
    if evolved:
        from scripts.league import evolved_pool
        pool, bodies = evolved_pool(run_dir)
    key = jax.random.PRNGKey(EVAL_SEED)
    return {"update": meta.get("update"), "bodies": bodies, "memory": config["train"].get("memory", 0),
            "bites": bite_shares(world, wcfg, prey_pi, pred_pi, key, episodes, pool),
            "eval": evaluate(wcfg, prey_pi, pred_pi, key, episodes, world, pool=pool)}


def controls(run_dir: Path, baseline: Path, episodes: int, evolved: bool) -> dict:
    """Not part of the pass bars: is it the memory, and does the same animal change after a bad bite?"""
    out = {}
    for name, path, wipe in (("memory", run_dir, False), ("wiped", run_dir, True), ("feed_forward", baseline, False)):
        config = json.loads((path / "config.json").read_text())
        world, wcfg = load_world_config(config)
        net = make_net(world, wcfg, config["train"].get("memory", 0))
        template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
        prey_params, pred_params, _ = load_brains(path / "checkpoints" / "latest.msgpack", template)
        prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
        if wipe:
            prey_pi = amnesiac(prey_pi)
        pool = None
        if evolved:
            from scripts.league import evolved_pool
            pool, _ = evolved_pool(path)
        key = jax.random.PRNGKey(EVAL_SEED)
        out[name] = {"bites": bite_shares(world, wcfg, prey_pi, pred_pi, key, episodes, pool),
                     "next": next_bites(world, wcfg, prey_pi, pred_pi, key, episodes, pool),
                     "eval": evaluate(wcfg, prey_pi, pred_pi, key, episodes, world, pool=pool)}
        b, n = out[name]["bites"], out[name]["next"]
        print(f"{name}: good-bite share {b['0-10']['share']:.3f} {b['10-40']['share']:.3f} {b['40+']['share']:.3f} | next bite good after a "
              f"good one {n['after_good']:.3f}, after a bad one {n['after_bad']:.3f} ({int(n['pairs_after_bad'])} pairs) | "
              f"prey lifetime {out[name]['eval']['prey_lifetime']:.1f}")
    (run_dir / "lifetime_controls.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--baseline", default=None, help="a feed-forward run of the same world")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true", help="start from each run's latest evolved bodies")
    p.add_argument("--controls", action="store_true", help="memory wiped, and next bite after a bad bite (lifetime_controls.json)")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    if args.controls:
        controls(run_dir, Path(args.baseline), args.episodes, args.evolved)
        return 0
    out = measure(run_dir, args.episodes, args.evolved)
    show = lambda m: " ".join(f"{k}: {v['share']:.3f}" for k, v in m["bites"].items())
    print(f"{run_dir.name} good-bite share by steps since flip  {show(out)}  prey lifetime {out['eval']['prey_lifetime']:.1f}")
    if args.baseline:
        base = measure(Path(args.baseline), args.episodes, args.evolved)
        print(f"{Path(args.baseline).name} good-bite share by steps since flip  {show(base)}  prey lifetime {base['eval']['prey_lifetime']:.1f}")
        out["baseline"] = {"run": Path(args.baseline).name, **base}
        out.update(verdict(out["bites"], base["bites"]))
        print(f"beats baseline by {out['beat']:+.3f} (bar +{BEAT_TARGET}), recovers by {out['recover']:+.3f} (bar +{RECOVER_TARGET}): "
              f"{'PASS' if out['pass'] else 'FAIL'}")
    (run_dir / "lifetime.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
