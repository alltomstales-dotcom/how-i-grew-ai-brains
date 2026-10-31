"""Policies map (key, obs[..., view, view, C]) to integer actions[...].

A policy with memory (attribute `memory` > 0) also has `step(key, obs, h) -> (actions, h')`.
Loops that run policies use `init_memory` and `act`, which work for both kinds: a
memoryless policy carries an empty (..., 0) state.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from simulations.predprey import NUM_ACTIONS


def random_policy(key: jax.Array, obs: jax.Array) -> jax.Array:
    return jax.random.randint(key, obs.shape[:-3], 0, NUM_ACTIONS)


def chase_policy(prey_channel: int):
    """Hand-written predator: step along the longer axis toward the nearest prey in view (Manhattan
    distance, first found on ties); wander at random when none is seen. Actions as predprey.DELTAS."""
    def policy(key: jax.Array, obs: jax.Array) -> jax.Array:
        view = obs.shape[-2]
        r = view // 2
        offs = jnp.arange(view) - r
        dy, dx = jnp.meshgrid(offs, offs, indexing="ij")  # obs[..., row = y, col = x, C]
        seen = obs[..., prey_channel] > 0
        dist = jnp.where(seen, jnp.abs(dx) + jnp.abs(dy), 10 * view)
        flat = jnp.argmin(dist.reshape(dist.shape[:-2] + (-1,)), axis=-1)
        tx, ty = dx.reshape(-1)[flat], dy.reshape(-1)[flat]
        toward = jnp.where(jnp.abs(tx) >= jnp.abs(ty), jnp.where(tx > 0, 3, 4), jnp.where(ty > 0, 2, 1))
        toward = jnp.where((tx == 0) & (ty == 0), 0, toward)
        return jnp.where(seen.any((-2, -1)), toward, random_policy(key, obs)).astype(jnp.int32)

    return policy


class NetworkPolicy:
    def __init__(self, net, params):
        self.net, self.params, self.memory = net, params, net.memory

    def __call__(self, key: jax.Array, obs: jax.Array) -> jax.Array:
        return jax.random.categorical(key, self.net.apply(self.params, obs)[0])

    def step(self, key: jax.Array, obs: jax.Array, h: jax.Array):
        logits, _, h = self.net.apply(self.params, obs, h)
        return jax.random.categorical(key, logits), h


def network_policy(net, params):
    return NetworkPolicy(net, params)


def init_memory(policy, batch_shape: tuple) -> jax.Array:
    return jnp.zeros(tuple(batch_shape) + (getattr(policy, "memory", 0),), jnp.float32)


def act(policy, key: jax.Array, obs: jax.Array, h: jax.Array, alive: jax.Array):
    """Actions and the next memory. Empty slots start from zero memory."""
    if not getattr(policy, "memory", 0):
        return policy(key, obs), h
    action, h = policy.step(key, obs, jnp.where(alive[..., None], h, 0.0))
    return action, jnp.where(alive[..., None], h, 0.0)  # an empty slot leaves nothing for a newborn


def forget(mem, prey_done, pred_done):
    """After a step: clear the memory of every slot whose animal died, was born or whose world ended,
    so a newborn in a slot freed this same step starts blank."""
    return (jnp.where(prey_done[..., None], 0.0, mem[0]), jnp.where(pred_done[..., None], 0.0, mem[1]))


def forget_after(mem, res):
    """forget() for evaluation loops, which keep stepping a world after it ends (one species gone):
    clear slots whose animal died or was born, but not every slot of an ended world."""
    live = ~res.done[..., None]
    return forget(mem, res.prey_done & live, res.pred_done & live)


def pair_memory(prey_policy, pred_policy, state):
    """Starting memory for both species of a (possibly batched) world state."""
    return init_memory(prey_policy, state.prey.alive.shape), init_memory(pred_policy, state.pred.alive.shape)


def pair_act(prey_policy, pred_policy, kp, kq, prey_obs, pred_obs, state, mem):
    prey_a, prey_h = act(prey_policy, kp, prey_obs, mem[0], state.prey.alive)
    pred_a, pred_h = act(pred_policy, kq, pred_obs, mem[1], state.pred.alive)
    return prey_a, pred_a, (prey_h, pred_h)
