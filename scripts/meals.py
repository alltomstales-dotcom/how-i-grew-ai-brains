"""Stage 5b test: do hunters go for fat prey? python -m scripts.meals --run-dir runs/NAME --control runs/CONTROL [--evolved]

Selectivity: the energy held by the grazers eaten, over what they would hold at the living population's mean. Bars are in the spec.
Writes meals.json in the run directory.
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
from agents.networks import init_params
from agents.policies import forget_after, network_policy, pair_act, pair_memory
from scripts.eval import EVAL_SEED, load_world_config
from simulations import evo
from simulations.worlds import make_net, obs_channels

SELECT_TARGET, DROP_TARGET = 1.15, 0.10


class no_fat_sense:
    def __init__(self, policy, channel: int):
        self.policy, self.channel, self.memory = policy, channel, getattr(policy, "memory", 0)

    def __call__(self, key, obs):
        return self.policy(key, obs.at[..., self.channel].set(0))

    def step(self, key, obs, h):
        return self.policy.step(key, obs.at[..., self.channel].set(0), h)


def fat_channel(cfg) -> int:
    return evo.OBS_CHANNELS + int(cfg.signals) + int(cfg.hills) + int(cfg.thickets > 0)


def hunting(world, cfg, prey_pi, pred_pi, key, episodes: int, pool=None) -> dict:
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
            k_act, k_step = jax.random.split(k)
            kp, kq = jax.random.split(k_act)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_pi, pred_pi, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(jax.random.split(k_step, episodes), state, prey_a, pred_a)
            i = res.info
            acc = acc + jnp.stack([i.eaten_energy.sum(), i.eaten_baseline.sum(), i.catches.sum(), state.pred.alive.sum(),
                                   (state.prey.energy * state.prey.alive).sum(), state.prey.alive.sum(),
                                   (state.prey.genes[..., 2] * state.prey.alive).sum()]).astype(jnp.float32)
            return (res.state, acc, forget_after(mem, res)), None

        init = (state, jnp.zeros(7, jnp.float32), pair_memory(prey_pi, pred_pi, state))
        (_, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    e, b, c, p, pe, pa, br = (float(v) for v in np.asarray(run(key)))
    return {"selectivity": e / max(b, 1e-6), "energy_per_catch": e / max(c, 1.0), "catches_per_pred_step": c / max(p, 1.0),
            "catches": c, "prey_mean_energy": pe / max(pa, 1.0), "prey_breed_gene": br / max(pa, 1.0)}


def load(path: Path, evolved: bool):
    config = json.loads((path / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, _ = load_brains(path / "checkpoints" / "latest.msgpack", template)
    pool = None
    if evolved:
        from scripts.league import evolved_pool
        pool, _ = evolved_pool(path)
    return world, wcfg, network_policy(net, prey_params), network_policy(net, pred_params), pool


def verdict(run: dict, control: dict, blind: dict) -> dict:
    out = {"select_ratio": run["selectivity"] / max(control["selectivity"], 1e-9),
           "blind_drop": (run["energy_per_catch"] - blind["energy_per_catch"]) / max(run["energy_per_catch"], 1e-9)}
    out["select_pass"] = bool(out["select_ratio"] >= SELECT_TARGET)
    out["sight_pass"] = bool(out["blind_drop"] >= DROP_TARGET)
    out["pass"] = out["select_pass"] and out["sight_pass"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true")
    args = p.parse_args(argv)
    run_dir, ctl_dir = Path(args.run_dir), Path(args.control)
    key = jax.random.PRNGKey(EVAL_SEED)
    world, cfg, prey_pi, pred_pi, pool = load(run_dir, args.evolved)
    run = hunting(world, cfg, prey_pi, pred_pi, key, args.episodes, pool)
    blind = hunting(world, cfg, prey_pi, no_fat_sense(pred_pi, fat_channel(cfg)), key, args.episodes, pool)
    w2, c2, p2, q2, pool2 = load(ctl_dir, args.evolved)
    control = hunting(w2, c2, p2, q2, key, args.episodes, pool2)
    out = {"run": run, "blind": blind, "control": control, **verdict(run, control, blind)}
    (run_dir / "meals.json").write_text(json.dumps(out, indent=1))
    f = lambda d: f"selectivity {d['selectivity']:.3f}, energy per catch {d['energy_per_catch']:.1f}, catches {d['catches_per_pred_step']:.4f}, prey energy {d['prey_mean_energy']:.1f}, breed gene {d['prey_breed_gene']:.3f}"
    print(f"stored meals: {f(run)}\nno fat sense: {f(blind)}\ncontrol:      {f(control)}")
    print(f"selectivity {out['select_ratio']:.2f}x control (bar {SELECT_TARGET}), meal value drops {100 * out['blind_drop']:.1f}% blind (bar {100 * DROP_TARGET:.0f}%): {'PASS' if out['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
