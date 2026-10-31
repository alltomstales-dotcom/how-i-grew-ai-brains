"""Fixed-seed evaluation of a (prey policy, predator policy) pairing."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from agents.policies import forget_after, pair_act, pair_memory
from simulations import predprey as pp


def evaluate(cfg: pp.WorldConfig, prey_policy, pred_policy, key, num_episodes: int, world=None, pool=None) -> dict:
    """Runs each of `num_episodes` worlds for the full `max_steps` and returns:
    catches_per_pred_step: catches / living-predator-steps;
    prey_lifetime: living-prey-steps / prey that ever lived (initial + born);
    mean_episode_len: steps until each episode would have ended (first extinction or limit).
    Counting continues past an extinction, so prey that outlive (or starve) the
    predators are credited for the time they survive. `pool` (evo worlds): start every
    episode from these evolved bodies instead of founders."""

    world = world or pp

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        keys = jax.random.split(k_reset, num_episodes)
        if pool is None:
            state = jax.vmap(partial(world.reset, cfg))(keys)
        else:
            state = jax.vmap(partial(world.reset, cfg), in_axes=(0, None))(keys, pool)
        names = ("catches", "pred_alive_steps", "prey_alive_steps", "prey_births", "episode_steps")
        acc0 = {name: jnp.zeros((), jnp.float32) for name in names}

        def body(carry, k):
            state, active, acc, mem = carry
            kp, kq = jax.random.split(k)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, cfg))(state)
            kw = jax.random.split(jax.random.fold_in(k, 1), num_episodes)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(world.advance, cfg))(kw, state, prey_a, pred_a)
            mem = forget_after(mem, res)
            acc = {
                "catches": acc["catches"] + res.info.catches.sum(),
                "pred_alive_steps": acc["pred_alive_steps"] + state.pred.alive.sum(),
                "prey_alive_steps": acc["prey_alive_steps"] + state.prey.alive.sum(),
                "prey_births": acc["prey_births"] + res.info.prey_births.sum(),
                "episode_steps": acc["episode_steps"] + active.sum(),
            }
            return (res.state, active & ~res.done, acc, mem), None

        init = (state, jnp.ones(num_episodes, bool), acc0, pair_memory(prey_policy, pred_policy, state))
        (_, _, acc, _), _ = jax.lax.scan(body, init, jax.random.split(k_steps, cfg.max_steps))
        return acc

    acc = {k: float(v) for k, v in jax.device_get(run(key)).items()}
    return {
        "catches_per_pred_step": acc["catches"] / max(acc["pred_alive_steps"], 1.0),
        "prey_lifetime": acc["prey_alive_steps"] / (cfg.init_prey * num_episodes + acc["prey_births"]),
        "mean_episode_len": acc["episode_steps"] / num_episodes,
    }


def lineage_share(cfg, prey_policy, pred_policy, key, num_episodes: int, pool, genes=None) -> dict:
    """Evolved-versus-founder test for the evo world. Each episode starts with
    half of each species' founders carrying pooled (evolved) genes, tag 1, and
    half carrying founder genes, tag 0; both halves use the same policies.
    Returns, per species, the tag-1 share of living-agent-steps over the second
    half of the episode (None when none of the species lived that long). `genes` picks which
    genes the tag-1 half takes from the pool (None: all of them; the rest stay founder genes)."""
    from simulations import evo

    start = cfg.max_steps // 2

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(evo.reset_mixed, cfg, genes=genes), in_axes=(0, None))(
            jax.random.split(k_reset, num_episodes), pool)
        acc0 = jnp.zeros(4, jnp.float32)  # prey evolved, prey all, pred evolved, pred all

        def body(carry, x):
            state, acc, mem = carry
            k, t = x
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(kw, num_episodes), state, prey_a, pred_a)
            mem = forget_after(mem, res)
            s = res.state
            counts = jnp.stack([
                (s.prey.alive & (s.prey.tag == 1)).sum(), s.prey.alive.sum(),
                (s.pred.alive & (s.pred.tag == 1)).sum(), s.pred.alive.sum(),
            ]).astype(jnp.float32)
            return (s, acc + jnp.where(t >= start, counts, 0.0), mem), None

        xs = (jax.random.split(k_steps, cfg.max_steps), jnp.arange(cfg.max_steps))
        (_, acc, _), _ = jax.lax.scan(body, (state, acc0, pair_memory(prey_policy, pred_policy, state)), xs)
        return acc

    acc = [float(v) for v in jax.device_get(run(key))]
    return {"prey": acc[0] / acc[1] if acc[1] else None, "pred": acc[2] / acc[3] if acc[3] else None}


def burrow_use(cfg, prey_policy, pred_policy, key, num_episodes: int) -> dict:
    """Terrain worlds: share of living prey-steps spent in a burrow when a predator is
    within THREAT cells ("near") and when none is ("far"), and near / far. None when
    there were no such prey-steps."""
    from simulations import evo

    @jax.jit
    def run(key):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(evo.reset, cfg))(jax.random.split(k_reset, num_episodes))

        def body(carry, k):
            state, acc, mem = carry
            kp, kq, kw = jax.random.split(k, 3)
            prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state)
            prey_a, pred_a, mem = pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem)
            res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(kw, num_episodes), state, prey_a, pred_a)
            mem = forget_after(mem, res)
            i = res.info
            counts = jnp.stack([i.prey_near.sum(), i.prey_near_burrow.sum(),
                                i.prey_far.sum(), i.prey_far_burrow.sum()]).astype(jnp.float32)
            return (res.state, acc + counts, mem), None

        (_, acc, _), _ = jax.lax.scan(body, (state, jnp.zeros(4, jnp.float32), pair_memory(prey_policy, pred_policy, state)),
                                   jax.random.split(k_steps, cfg.max_steps))
        return acc

    n_near, hid_near, n_far, hid_far = (float(v) for v in jax.device_get(run(key)))
    near = hid_near / n_near if n_near else None
    far = hid_far / n_far if n_far else None
    return {"near": near, "far": far, "ratio": near / far if near is not None and far else None}
