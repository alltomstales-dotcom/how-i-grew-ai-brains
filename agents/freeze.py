"""Stage 20b: freeze one species' body genes and let the other's evolve.

Stage 20 froze both species at once with `gene_sigma=0`. `FrozenBodies(world, "pred")` wraps an evo world module so that,
after every step, the named species' body genes are set back to the founder body (`founder_animal`), in every slot. A
child born this step therefore never acts in a mutated body: the world's next step and the observations made from it
already see the founder genes. Gene mutation still draws its random numbers as before, so freezing both species this way
gives the same worlds as `gene_sigma=0` (stage 20 arm D), and the one-species cells differ from it only in what they freeze.
Restarted worlds draw founders from the living animals, whose frozen genes are the founder body, so restarts keep it too.
Spec: docs/specs/2026-11-04-whose-bodies-design.md
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

FREEZE = ("none", "prey", "pred", "both")


def clamp(cfg, state, which: str):
    """state with the named species' genes set to the founder body (an unbatched world state)."""
    founder = jnp.asarray(cfg.founder_animal, jnp.float32)
    for kind in ("prey", "pred"):
        if which in (kind, "both"):
            sp = getattr(state, kind)
            sp = sp._replace(genes=jnp.broadcast_to(founder, sp.genes.shape).astype(sp.genes.dtype))
            state = state._replace(**{kind: sp})
    return state


class FrozenBodies:
    """An evo world module whose `advance` and `batch_step` keep the named species in the founder body; all else passes through."""

    def __init__(self, world, which: str):
        if which not in FREEZE:
            raise ValueError(f"freeze must be one of {FREEZE}, not {which!r}")
        self._world, self.which = world, which

    def __getattr__(self, name):
        return getattr(self._world, name)

    def reset(self, cfg, key, pool=None):
        return clamp(cfg, self._world.reset(cfg, key, pool), self.which)

    def advance(self, cfg, key, state, prey_act, pred_act):
        res = self._world.advance(cfg, key, state, prey_act, pred_act)
        return res._replace(state=clamp(cfg, res.state, self.which))

    def batch_step(self, cfg, key, states, prey_act, pred_act):
        """As the world's batch_step (simulations/evo.py), with each world's step clamped before ended worlds restart."""
        w = self._world
        n = prey_act.shape[0]
        k_step, k_reset = jax.random.split(key)
        res = jax.vmap(partial(self.advance, cfg))(jax.random.split(k_step, n), states, prey_act, pred_act)

        def reseed(s):
            k_thin, k_worlds = jax.random.split(k_reset)
            pool = w.thin_plants(k_thin, w.gene_pool(s))
            fresh = jax.vmap(partial(self.reset, cfg), in_axes=(0, None))(jax.random.split(k_worlds, n), pool)
            return jax.tree.map(lambda a, b: jnp.where(res.done.reshape((n,) + (1,) * (a.ndim - 1)), a, b), fresh, s)

        next_state = jax.lax.cond(res.done.any(), reseed, lambda s: s, res.state)
        prey_obs, pred_obs = jax.vmap(partial(w.observe, cfg))(next_state)
        return res._replace(state=next_state), prey_obs, pred_obs


def freeze_world(world, which: str):
    """world unchanged for "none", else wrapped."""
    return world if which == "none" else FrozenBodies(world, which)
