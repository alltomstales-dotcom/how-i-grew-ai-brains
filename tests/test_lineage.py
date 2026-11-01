"""Stage 11c: a lineage mark survives a world's restart, carried through the gene pool."""
import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo

CFG = evo.ContinentConfig(size=12, max_prey=16, max_pred=8, init_prey=12, init_pred=6, max_steps=4)
KEY = jax.random.PRNGKey(3)


def test_founders_keep_the_marks_of_the_pool_they_came_from():
    s = jax.vmap(lambda k: evo.reset(CFG, k))(jax.random.split(KEY, 4))
    half = jnp.arange(CFG.max_prey) % 2
    s = s._replace(prey=s.prey._replace(tag=jnp.broadcast_to(half, s.prey.tag.shape)))
    pool = evo.gene_pool(s)
    fresh = evo.reset(CFG, jax.random.PRNGKey(9), pool)
    tags = np.asarray(fresh.prey.tag)[np.asarray(fresh.prey.alive)]
    assert set(np.unique(tags)) == {0, 1}
    assert (np.asarray(fresh.pred.tag) == 0).all()


def test_a_marked_line_is_still_marked_after_a_batch_restart():
    s = jax.vmap(lambda k: evo.reset(CFG, k))(jax.random.split(KEY, 4))
    s = s._replace(prey=s.prey._replace(tag=jnp.ones_like(s.prey.tag)))
    still = jnp.zeros((4, CFG.max_prey), jnp.int32), jnp.zeros((4, CFG.max_pred), jnp.int32)
    k = KEY
    for _ in range(CFG.max_steps + 1):
        k, sub = jax.random.split(k)
        res, _, _ = evo.batch_step(CFG, sub, s, *still)
        s = res.state
    assert int(s.t.max()) < CFG.max_steps  # the worlds did restart
    tags = np.asarray(s.prey.tag)[np.asarray(s.prey.alive)]
    assert len(tags) and (tags == 1).all()


def test_a_pool_without_marks_still_works():
    s = jax.vmap(lambda k: evo.reset(CFG, k))(jax.random.split(KEY, 2))
    pool = evo.gene_pool(s)._replace(prey_tag=None, pred_tag=None)
    fresh = evo.reset(CFG, KEY, pool)
    assert (np.asarray(fresh.prey.tag) == 0).all()
