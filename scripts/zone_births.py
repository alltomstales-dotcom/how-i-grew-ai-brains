"""Where are children born in a mutation-zone world? python -m scripts.zone_births --run-dir runs/NAME

Runs the trained brains and evolved bodies with no restarts and counts births in the zones (every fourth column), which a zone's
average mutation rests on: an even spread gives a quarter. Written to <run>/zone_births.json.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from agents.checkpoint import load_brains
from agents.networks import init_params
from agents.policies import network_policy
from scripts.eval import EVAL_SEED, load_world_config
from scripts.league import evolved_pool
from simulations.worlds import make_net, obs_channels


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, 0)
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_p, pred_p = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)[:2]
    prey_pi, pred_pi = network_policy(net, prey_p), network_policy(net, pred_p)
    pool, _ = evolved_pool(run_dir)
    n = args.episodes
    key, k_reset = jax.random.split(jax.random.PRNGKey(EVAL_SEED))
    state = jax.vmap(lambda c: world.reset(wcfg, c, pool))(jax.random.split(k_reset, n))

    def count(sp, parent):
        slots = jnp.arange(parent.shape[-1])
        born = parent != slots
        zone = sp.pos[..., 0] % 4 == 0
        return born.sum(), (born & zone).sum()

    @jax.jit
    def body(carry, k):
        s = carry
        ka, kb, kc = jax.random.split(k, 3)
        po, qo = jax.vmap(partial(world.observe, wcfg))(s)
        pa = jax.vmap(prey_pi)(jax.random.split(ka, n), po)
        qa = jax.vmap(pred_pi)(jax.random.split(kb, n), qo)
        res = jax.vmap(partial(world.advance, wcfg))(jax.random.split(kc, n), s, pa, qa)
        return res.state, count(res.state.prey, res.info.prey_parent) + count(res.state.pred, res.info.pred_parent)

    state, totals = jax.lax.scan(body, state, jax.random.split(key, args.steps))
    g, gz, h, hz = (float(x.sum()) for x in totals)
    out = {"grazer_births": g, "grazer_share_in_zones": gz / max(g, 1.0), "hunter_births": h, "hunter_share_in_zones": hz / max(h, 1.0)}
    (run_dir / "zone_births.json").write_text(json.dumps(out, indent=1))
    print(f"{run_dir.name}: grazers {out['grazer_share_in_zones']:.3f} of {g:.0f} births in zones, hunters {out['hunter_share_in_zones']:.3f} of {h:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
