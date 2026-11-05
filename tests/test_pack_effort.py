"""Stage 23: packs with cheaters (effort gene, equal / proportional / neutral sharing, random breeding order).
Spec: docs/specs/2026-11-05-pack-cheaters-design.md"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulations import evo
from simulations.worlds import get_world

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
BASE = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40, pack_size=2, **QUIET)
EQUAL = dataclasses.replace(BASE, effort="equal")
PROP = dataclasses.replace(BASE, effort="prop")
NEUTRAL = dataclasses.replace(BASE, effort="neutral")


def hunter(x, y, energy, effort):
    return (x, y, energy, 0.0, 0.0, 1.0, 0.5, 0.5, effort)  # speed, sight, breed, size, brain, effort (slot 5)


def grazer(x, y):
    return (x, y, 20.0, 0.0, 0.0, 1.0)


def step(c, s, key=0):
    return evo.advance(c, jax.random.PRNGKey(key), s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32))


def test_off_by_default():
    assert evo.EvoConfig().effort == "" and evo.EvoConfig().breed_random is False
    assert get_world("pack")[1]().effort == "" and get_world("lasting_stealth")[1]().breed_random is False
    s = evo.make_state(BASE, prey=[grazer(5, 5)], pred=[hunter(4, 5, 30.0, 0.5), hunter(6, 6, 30.0, 0.5)])
    assert step(BASE, s).info.effort is None
    for name, mode in (("pack_effort_equal", "equal"), ("pack_effort_prop", "prop"), ("pack_effort_neutral", "neutral")):
        c = get_world(name)[1]()
        assert c.effort == mode and c.pack_size == 2 and c.max_pred == 64 and c.breed_random and c.plant_sigma == 0
        assert c.stealth_radius == 1 and c.gene_sigma == 0.05 and c.founder_animal[evo.EFFORT] == 0.5


def test_effort_refuses_omnivores_and_lone_hunters():
    s = evo.make_state(BASE, prey=[grazer(5, 5)], pred=[hunter(4, 5, 30.0, 0.5)])
    for bad in (dict(omnivore=True), dict(brains=True), dict(pack_size=0), dict(effort="half")):
        with pytest.raises(ValueError):
            step(dataclasses.replace(EQUAL, **bad), s)


def _catch_rate(c, e1, e2, n=4000):
    s = evo.make_state(c, prey=[grazer(5, 5)], pred=[hunter(4, 5, 30.0, e1), hunter(6, 6, 30.0, e2)])
    run = jax.jit(jax.vmap(lambda k: evo.advance(c, k, s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32)).info.catches))
    return float(run(jax.random.split(jax.random.PRNGKey(3), n)).mean())


def test_catch_chance_follows_summed_effort():
    for e1, e2 in ((0.5, 0.5), (0.1, 0.2), (0.9, 0.0)):
        want = 1 - np.exp(-2 * (e1 + e2))
        assert abs(_catch_rate(EQUAL, e1, e2) - want) < 0.03, (e1, e2)
    assert _catch_rate(EQUAL, 0.0, 0.0, 500) == 0.0
    # neutral: .5 stands in for the gene, so two do-nothing genes still catch at 1 - exp(-2)
    assert abs(_catch_rate(NEUTRAL, 0.0, 0.0) - (1 - np.exp(-2))) < 0.03


def test_a_lone_hunter_never_catches_and_pays_for_effort():
    s = evo.make_state(EQUAL, prey=[grazer(5, 5)], pred=[hunter(4, 5, 30.0, 1.0)])
    r = step(EQUAL, s)
    assert int(r.info.catches) == 0
    upkeep = float(evo._upkeep(EQUAL, s.pred.genes[0], EQUAL.pred_base_cost))
    assert np.isclose(float(r.state.pred.energy[0]), 30.0 - upkeep - 0.1 * 1.0, atol=1e-5)
    rn = step(NEUTRAL, s)  # neutral pays for .5, whatever the gene
    assert np.isclose(float(rn.state.pred.energy[0]), 30.0 - upkeep - 0.1 * 0.5, atol=1e-5)
    assert int(r.info.effort.lone) == 1 and int(r.info.effort.attempts) == 0


def _shared(c, e1, e2):
    """Energy each of two hunters gains from a sure catch, against the same two hunters with no grazer."""
    c = dataclasses.replace(c, effort_k=200.0)  # the catch is certain
    pred = [hunter(4, 5, 10.0, e1), hunter(6, 6, 10.0, e2)]  # low enough that nobody breeds after the meal
    r = step(c, evo.make_state(c, prey=[grazer(5, 5)], pred=pred))
    alone = step(c, evo.make_state(c, pred=pred))
    assert int(r.info.catches) == 1
    return np.asarray(r.state.pred.energy[:2]) - np.asarray(alone.state.pred.energy[:2]), r


def test_equal_and_proportional_splits():
    gained, r = _shared(EQUAL, 0.2, 0.8)
    np.testing.assert_allclose(gained, [25.0, 25.0], atol=1e-4)
    x = r.info.effort
    np.testing.assert_allclose(np.asarray(x.contrib[:2]), [0.2, 0.8], atol=1e-5)
    assert bool(x.caught[0]) and bool(x.caught[1]) and int(x.free_riders) == 1  # the .2 hunter takes half for a fifth
    assert np.isclose(float(x.free_kept), 25.0, atol=1e-4) and np.asarray(x.group[:2]).tolist() == [2, 2]
    assert np.asarray(x.assigned[:2]).tolist() == [0, 0] and int(x.assigned[2]) == -1
    gained, r = _shared(PROP, 0.2, 0.8)
    np.testing.assert_allclose(gained, [10.0, 40.0], atol=1e-4)
    assert int(r.info.effort.free_riders) == 0
    gained, r = _shared(NEUTRAL, 0.2, 0.8)
    np.testing.assert_allclose(gained, [25.0, 25.0], atol=1e-4)
    assert int(r.info.effort.free_riders) == 0
    np.testing.assert_allclose(np.asarray(r.info.effort.e[:2]), [0.5, 0.5])


def test_meal_kept_after_the_cap():
    c = dataclasses.replace(PROP, effort_k=200.0)
    pred = [hunter(4, 5, 55.0, 0.5), hunter(6, 6, 10.0, 0.5)]
    s = evo.make_state(c, prey=[grazer(5, 5)], pred=pred)
    r = step(c, s)
    upkeep = float(evo._upkeep(c, s.pred.genes[0], c.pred_base_cost))  # the step's own genes; it ends at the founder body
    kept = np.asarray(r.info.effort.kept[:2])
    np.testing.assert_allclose(kept, [c.pred_energy_cap - (55.0 - upkeep), 25.0], atol=1e-4)
    assert np.isclose(float(r.info.effort.meal_total), 50.0, atol=1e-4)
    # the hunter that lost meal to the cap kept less than its effort share, so its partner free-rode
    assert int(r.info.effort.free_riders) == 1 and bool(r.info.effort.caught[1])


def test_only_effort_mutates_and_resets_hold_the_founder_body():
    c = dataclasses.replace(get_world("pack_effort_equal")[1](), size=16, max_prey=16, max_pred=16, init_prey=8, init_pred=6,
                            gene_sigma=0.3, pred_repro_energy=10.0, prey_repro_energy=10.0, pred_base_cost=0.0)
    founder = np.asarray(c.founder_animal, np.float32)
    s = evo.reset(c, jax.random.PRNGKey(0))
    for sp in (s.prey, s.pred):
        np.testing.assert_array_equal(np.asarray(sp.genes), np.broadcast_to(founder, sp.genes.shape))

    def body(state, k):
        res = evo.advance(c, k, state, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32))
        return res.state, res.info.pred_births

    s, births = jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(1), 6))
    assert int(births.sum()) > 0
    pg, qg = np.asarray(s.pred.genes), np.asarray(s.prey.genes)
    other = [i for i in range(evo.ANIMAL_GENES) if i != evo.EFFORT]
    np.testing.assert_array_equal(pg[:, other], np.broadcast_to(founder[other], pg[:, other].shape))
    np.testing.assert_array_equal(qg, np.broadcast_to(founder, qg.shape))
    alive = np.asarray(s.pred.alive)
    assert np.std(pg[alive, evo.EFFORT]) > 0.01  # effort did mutate
    # a restart from the pool carries effort and nothing else
    s2 = evo.reset(c, jax.random.PRNGKey(2), evo.gene_pool(jax.tree.map(lambda a: a[None], s)))
    g2 = np.asarray(s2.pred.genes)
    np.testing.assert_array_equal(g2[:, other], np.broadcast_to(founder[other], g2[:, other].shape))
    assert set(np.round(g2[np.asarray(s2.pred.alive), evo.EFFORT], 5)) <= set(np.round(pg[alive, evo.EFFORT], 5))


def test_random_breeding_order_is_fair():
    """Two ready hunters, one free slot: slot order always picks slot 0; breed_random picks each about half the time."""
    c = dataclasses.replace(BASE, max_pred=3, pred_repro_energy=10.0)
    pred = [hunter(1, 1, 50.0, 0.5), hunter(8, 8, 50.0, 0.5)]
    s = evo.make_state(c, pred=pred)

    def first_parent(cfg):
        f = jax.jit(jax.vmap(lambda k: evo.advance(cfg, k, s, jnp.zeros(cfg.max_prey, jnp.int32), jnp.zeros(cfg.max_pred, jnp.int32)).info.pred_parent[2]))
        return np.asarray(f(jax.random.split(jax.random.PRNGKey(4), 2000)))

    assert (first_parent(c) == 0).all()
    share = (first_parent(dataclasses.replace(c, breed_random=True)) == 0).mean()
    assert 0.45 < share < 0.55


def test_blocked_births_and_metrics():
    c = dataclasses.replace(EQUAL, max_pred=2, pred_repro_energy=10.0)
    s = evo.make_state(c, prey=[grazer(5, 5)], pred=[hunter(1, 1, 50.0, 0.3), hunter(8, 8, 50.0, 0.7)])
    r = step(c, s)
    x = r.info.effort
    assert int(x.eligible) == 2 and int(x.blocked) == 2  # no free slot for either
    stacked = jax.tree.map(lambda a: jnp.stack([a, a]), r.info)
    m = evo.extra_metrics(stacked)
    assert np.isclose(float(m["blocked_share"]), 1.0) and np.isclose(float(m["pred_effort"]), 0.5, atol=1e-5)
    assert np.isclose(float(m["pred_effort_std"]), 0.2, atol=1e-4)
    assert float(m["effort_energy_corr"]) < -0.99  # same start, so the only energy difference is what effort cost
    assert "pred_effort" not in evo.extra_metrics(jax.tree.map(lambda a: jnp.stack([a, a]), step(BASE, evo.make_state(BASE, pred=[hunter(1, 1, 50.0, .5)])).info))
