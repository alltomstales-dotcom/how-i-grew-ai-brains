"""How big a map can one machine run? python -m scripts.mapbench [--sizes 48,128,256,512,1024,2048] [--steps 200]

One world at each size, with animals at the density of the 48 x 48 world (96 grazers, 32 hunters at most), hand-written
hunters and grazers, and evolution on. Reports steps per second and peak device memory. No training: a brain's memory and
time come on top, and scale with the number of animals. Stops at the first size that runs out of memory.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import jax
import jax.numpy as jnp

from agents.policies import chase_policy
from simulations import evo


def bench(size: int, steps: int, per_cell: bool = True) -> dict:
    k = (size / 48) ** 2
    cfg = replace(evo.LastingStealthConfig(), size=size, max_prey=max(1, int(96 * k)), max_pred=max(1, int(32 * k)),
                  init_prey=max(1, int(24 * k)), init_pred=max(1, int(8 * k)), max_steps=steps + 1)
    graze, hunt = chase_policy(0), chase_policy(2)

    @jax.jit
    def run(key):
        state = evo.reset(cfg, key)

        def body(state, k):
            kp, kq, ks = jax.random.split(k, 3)
            prey_obs, pred_obs = evo.observe(cfg, state)
            return evo.advance(cfg, ks, state, graze(kp, prey_obs), hunt(kq, pred_obs)).state, (state.prey.alive.sum(), state.pred.alive.sum())

        return jax.lax.scan(body, state, jax.random.split(key, steps))[1]

    jax.block_until_ready(run(jax.random.PRNGKey(0)))  # compile
    t0 = time.time()
    g, h = jax.block_until_ready(run(jax.random.PRNGKey(1)))
    dt = time.time() - t0
    stats = jax.devices()[0].memory_stats() or {}
    return {"size": size, "slots": (cfg.max_prey, cfg.max_pred), "steps_per_s": steps / dt, "cell_steps_per_s": steps * size ** 2 / dt,
            "grazers": int(g[-1]), "hunters": int(h[-1]), "peak_gib": stats.get("peak_bytes_in_use", 0) / 2 ** 30}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="48,128,256,512,1024,2048")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args(argv)
    print(jax.devices()[0].device_kind, jnp.zeros(1).device)
    for size in map(int, args.sizes.split(",")):
        try:
            r = bench(size, args.steps)
        except Exception as e:  # out of memory, most likely
            print(f"size {size}: stopped, {type(e).__name__}: {str(e)[:160]}", flush=True)
            break
        print(f"size {r['size']:5d}: slots {r['slots'][0]:7d} + {r['slots'][1]:6d}, {r['steps_per_s']:9.1f} steps/s, {r['cell_steps_per_s'] / 1e6:9.1f} M cell-steps/s, "
              f"{r['grazers']} grazers, {r['hunters']} hunters at the end, peak {r['peak_gib']:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
