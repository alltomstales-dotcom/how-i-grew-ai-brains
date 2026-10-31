"""GPU feasibility check for a DGX Spark: JAX must see the GB10 and run jit + vmap."""
from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp


def main() -> int:
    print("jax", jax.__version__, flush=True)
    devices = jax.devices()
    print("devices", devices, flush=True)
    if devices[0].platform != "gpu":
        print("FAIL: no GPU device visible to JAX", flush=True)
        return 1

    x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))
    f = jax.jit(lambda a: (a @ a.T).sum())
    f(x).block_until_ready()
    t0 = time.time()
    f(x).block_until_ready()
    print(f"matmul 4096x4096: {time.time() - t0:.4f} s", flush=True)

    grids = jnp.zeros((4096, 48, 48), jnp.int32)
    shifts = jnp.arange(4096) % 48
    roll = jax.jit(jax.vmap(lambda g, s: jnp.roll(g + 1, s, axis=0)))
    roll(grids, shifts).block_until_ready()
    t0 = time.time()
    for _ in range(100):
        grids = roll(grids, shifts)
    grids.block_until_ready()
    print(f"vmapped grid steps/s: {4096 * 100 / (time.time() - t0):,.0f}", flush=True)

    import flax
    import imageio_ffmpeg
    import optax

    print("flax", flax.__version__, "optax", optax.__version__, flush=True)
    print("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe(), flush=True)
    print("memory_stats", devices[0].memory_stats(), flush=True)
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
