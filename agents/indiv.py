"""Individual brains (Stage 14a): every animal holds its own weights, learns from its own life, and passes a brain on.

mode "darwin": no learning; a child gets its parent's birth brain plus noise.
mode "baldwin": learning in life; a child gets its parent's birth brain plus noise.
mode "lamarck": learning in life; a child gets its parent's brain as it stands, plus noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

MODES = ("darwin", "baldwin", "lamarck")


@dataclass(frozen=True)
class IndivConfig:
    mode: str = "lamarck"
    lr: float = 0.001
    sigma: float = 0.01
    gamma: float = 0.98
    energy_scale: float = 25.0
    clip: float = 1.0  # each animal's gradient norm, each step; without it brains blow up in a few steps
    gene_lr: bool = False  # Stage 14b: each animal's rate is 2 x lr x its brain gene, so founders at 0.5 learn at lr
    steps: int = 32


class Pop(NamedTuple):
    params: Any  # pytree with leading (E, N)
    birth: Any  # the same, as each animal was born


class Carry(NamedTuple):
    state: Any
    prey: Pop
    pred: Pop
    prey_obs: jax.Array
    pred_obs: jax.Array
    key: jax.Array


def tile(params, e: int, n: int, key, noise: float) -> Pop:
    """One brain copied to every slot of every world, with a little noise on each copy."""
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    out = [leaf[None, None] + noise * jax.random.normal(k, (e, n) + leaf.shape) for leaf, k in zip(leaves, keys)]
    p = jax.tree.unflatten(tree, out)
    return Pop(p, p)


def _where(mask, a, b):
    """mask has shape (E, N): choose per slot across a pytree."""
    return jax.tree.map(lambda x, y: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - 2)), x, y), a, b)


def _take(tree, index):
    """index (E, N): each slot takes the leaf of slot index[e, n] in its own world."""
    return jax.tree.map(lambda x: jax.vmap(lambda w, i: w[i])(x, index), tree)


def _noisy(tree, key, sigma):
    leaves, t = jax.tree.flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(t, [x + sigma * jax.random.normal(k, x.shape) for x, k in zip(leaves, keys)])


def inherit(pop: Pop, parent, key, cfg: IndivConfig) -> Pop:
    """Newborns (slots whose parent isn't themselves) take a brain from their parent."""
    e, n = parent.shape
    born = parent != jnp.arange(n)[None, :]
    source = pop.params if cfg.mode == "lamarck" else pop.birth
    child = _noisy(_take(source, parent), key, cfg.sigma)
    return Pop(_where(born, child, pop.params), _where(born, child, pop.birth))


def redraw(pop: Pop, was_alive, done, key, cfg: IndivConfig) -> Pop:
    """The animals of a world that just ended take brains from living animals in any world."""
    e, n = was_alive.shape
    weights = was_alive.reshape(-1).astype(jnp.float32)
    index = jax.random.categorical(key, jnp.log(jnp.where(weights.sum() > 0, weights, 1.0) + 1e-12), shape=(e, n))
    source = pop.params if cfg.mode == "lamarck" else pop.birth
    drawn = jax.tree.map(lambda x: x.reshape((e * n,) + x.shape[2:])[index], source)
    mask = jnp.broadcast_to(done[:, None], (e, n))
    return Pop(_where(mask, drawn, pop.params), _where(mask, drawn, pop.birth))


def make_step(net, world, wcfg, cfg: IndivConfig, restart: bool = True, fixed_pred=None):
    """One step of every world: act, step, learn, inherit. restart False: worlds carry on past an extinction, for evaluation.
    fixed_pred: a hand-written hunter policy, with hunters' brains left alone."""
    apply_one = lambda p, o: net.apply(p, o)
    apply_all = jax.vmap(jax.vmap(apply_one))

    def loss(p, obs, action, target):
        logits, value = net.apply(p, obs)
        logp = jax.nn.log_softmax(logits)[action]
        return -jax.lax.stop_gradient(target - value) * logp + 0.5 * (target - value) ** 2

    grads_all = jax.vmap(jax.vmap(jax.grad(loss)))

    def reward(sp_before, sp_after, slot_done, parent, cfg_):
        e, n = parent.shape
        gave = jax.vmap(lambda p: jnp.zeros(n, bool).at[p].max(p != jnp.arange(n)))(parent)
        r = (sp_after.energy - sp_before.energy) / cfg_.energy_scale
        return jnp.where(slot_done, -1.0, jnp.where(gave, 1.0, r))

    def learn(pop, obs, action, next_obs, sp_before, sp_after, slot_done, parent, world_done):
        if cfg.mode == "darwin":
            return pop
        next_value = apply_all(pop.params, next_obs)[1]
        target = reward(sp_before, sp_after, slot_done, parent, cfg) + cfg.gamma * next_value * ~slot_done
        mask = sp_before.alive & ~world_done[:, None]
        grads = grads_all(pop.params, obs, action, target)
        norm = jnp.sqrt(sum((g ** 2).sum(tuple(range(2, g.ndim))) for g in jax.tree.leaves(grads)))
        scale = jnp.where(mask, jnp.minimum(1.0, cfg.clip / jnp.maximum(norm, 1e-9)), 0.0)
        if cfg.gene_lr:
            scale = scale * 2.0 * sp_before.genes[..., 4]
        return pop._replace(params=jax.tree.map(
            lambda p, g: p - cfg.lr * g * scale.reshape(scale.shape + (1,) * (p.ndim - 2)), pop.params, grads))

    def step(carry: Carry, _):
        key, k_prey, k_pred, k_step, k_i1, k_i2, k_r1, k_r2 = jax.random.split(carry.key, 8)
        prey_logits, _v = apply_all(carry.prey.params, carry.prey_obs)
        prey_a = jax.random.categorical(k_prey, prey_logits)
        if fixed_pred is None:
            pred_logits, _v = apply_all(carry.pred.params, carry.pred_obs)
            pred_a = jax.random.categorical(k_pred, pred_logits)
        else:
            pred_a = fixed_pred(k_pred, carry.pred_obs)
        before = carry.state
        if restart:
            res, prey_obs, pred_obs = world.batch_step(wcfg, k_step, before, prey_a, pred_a)
            world_done = res.done
        else:
            n = prey_a.shape[0]
            res = jax.vmap(partial(world.advance, wcfg))(jax.random.split(k_step, n), before, prey_a, pred_a)
            prey_obs, pred_obs = jax.vmap(partial(world.observe, wcfg))(res.state)
            world_done = jnp.zeros(n, bool)
        prey = learn(carry.prey, carry.prey_obs, prey_a, prey_obs, before.prey, res.state.prey, res.prey_done, res.info.prey_parent, world_done)
        prey = inherit(prey, res.info.prey_parent, k_i1, cfg)
        pred = carry.pred
        if fixed_pred is None:
            pred = learn(pred, carry.pred_obs, pred_a, pred_obs, before.pred, res.state.pred, res.pred_done, res.info.pred_parent, world_done)
            pred = inherit(pred, res.info.pred_parent, k_i2, cfg)
        if restart:
            prey = redraw(prey, before.prey.alive, world_done, k_r1, cfg)
            pred = redraw(pred, before.pred.alive, world_done, k_r2, cfg)
        return Carry(res.state, prey, pred, prey_obs, pred_obs, key), res.info

    return step


def spread(pop: Pop, alive) -> dict:
    """How far weights differ across living animals, and how far they've moved from birth (root mean square per weight)."""
    w = alive.astype(jnp.float32)
    count = jnp.maximum(w.sum(), 1.0)
    var, moved, size = 0.0, 0.0, 0
    for x, b in zip(jax.tree.leaves(pop.params), jax.tree.leaves(pop.birth)):
        m = w.reshape(w.shape + (1,) * (x.ndim - 2))
        mean = (x * m).sum((0, 1)) / count
        var = var + (((x - mean) ** 2) * m).sum() / count
        moved = moved + (((x - b) ** 2) * m).sum() / count
        size = size + mean.size
    return {"weight_spread": jnp.sqrt(var / size), "weight_moved": jnp.sqrt(moved / size)}
