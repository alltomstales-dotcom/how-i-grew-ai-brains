"""Stage 6a: prey that only two predators together can catch."""
import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40, pack_size=2, **QUIET)


def step(c, s):
    return evo.advance(c, jax.random.PRNGKey(0), s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32))


def test_a_lone_predator_catches_nothing():
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.0, 1.0)], pred=[(5, 5, 30.0, 0.0, 0.0, 1.0)])
    res = step(CFG, s)
    assert bool(res.state.prey.alive[0]) and int(res.info.catches) == 0


def test_two_predators_next_to_a_prey_share_the_meal():
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.0, 1.0)],
                       pred=[(4, 5, 30.0, 0.0, 0.0, 1.0), (6, 6, 30.0, 0.0, 0.0, 1.0)])
    alone = step(CFG, evo.make_state(CFG, pred=[(4, 5, 30.0, 0.0, 0.0, 1.0), (6, 6, 30.0, 0.0, 0.0, 1.0)]))
    res = step(CFG, s)
    assert not bool(res.state.prey.alive[0]) and int(res.info.catches) == 1
    gained = np.asarray(res.state.pred.energy[:2]) - np.asarray(alone.state.pred.energy[:2])
    np.testing.assert_allclose(gained, CFG.catch_energy, atol=1e-4)  # 2 * catch_energy shared by 2


def test_hunters_two_cells_away_do_not_count():
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.0, 1.0)],
                       pred=[(5, 5, 30.0, 0.0, 0.0, 1.0), (7, 5, 30.0, 0.0, 0.0, 1.0)])
    assert bool(step(CFG, s).state.prey.alive[0])


def test_each_predator_joins_one_hunt_and_edges_wrap():
    # Pred 0 sits between two prey; it joins the lower-index one. Pred 1 wraps around the edge.
    s = evo.make_state(CFG, prey=[(0, 5, 20.0, 0.0, 0.0, 1.0), (2, 5, 20.0, 0.0, 0.0, 1.0)],
                       pred=[(1, 5, 30.0, 0.0, 0.0, 1.0), (11, 5, 30.0, 0.0, 0.0, 1.0)])
    alive = np.asarray(step(CFG, s).state.prey.alive)
    assert not alive[0] and alive[1]


def test_without_packs_catches_are_unchanged():
    c = evo.EvoConfig(**{**CFG.__dict__, "pack_size": 0})
    s = evo.make_state(c, prey=[(5, 5, 20.0, 0.0, 0.0, 1.0)], pred=[(5, 5, 30.0, 0.0, 0.0, 1.0)])
    res = step(c, s)
    assert not bool(res.state.prey.alive[0]) and int(res.info.catches) == 1


def test_pack_world_is_registered():
    from simulations.worlds import get_world
    module, config_cls = get_world("pack")
    assert module is evo and config_cls().pack_size == 2


FLANK = evo.EvoConfig(**{**CFG.__dict__, "flank": True})
PREY = [(5, 5, 20.0, 0.0, 0.0, 1.0)]


def hunted_by(c, *cells):
    return bool(step(c, evo.make_state(c, prey=PREY, pred=[(x, y, 30.0, 0.0, 0.0, 1.0) for x, y in cells])).state.prey.alive[0])


def test_flanked_prey_need_hunters_on_opposite_sides():
    assert hunted_by(FLANK, (4, 5), (4, 6))  # both west: it gets away
    assert hunted_by(FLANK, (5, 5), (6, 5))  # one on the prey's own cell has no side
    assert not hunted_by(FLANK, (4, 5), (6, 5))  # west and east
    assert not hunted_by(FLANK, (5, 4), (5, 6))  # north and south
    assert not hunted_by(FLANK, (6, 4), (4, 6))  # north-east and south-west
    assert hunted_by(FLANK, (4, 4), (4, 6))  # north-west and south-west: both west (review finding)
    assert hunted_by(FLANK, (4, 4), (6, 4))  # north-west and north-east: both north
    assert hunted_by(FLANK, (5, 4), (6, 6))  # north and south-east: not exactly opposite
    assert not hunted_by(FLANK, (4, 4), (6, 6), (6, 4))  # a third hunter does not spoil a flank
    assert not hunted_by(CFG, (4, 5), (4, 6))  # without flanking, any two will do


def test_flanking_wraps_around_the_edges():
    s = evo.make_state(FLANK, prey=[(0, 5, 20.0, 0.0, 0.0, 1.0)],
                       pred=[(11, 5, 30.0, 0.0, 0.0, 1.0), (1, 5, 30.0, 0.0, 0.0, 1.0)])
    assert not bool(step(FLANK, s).state.prey.alive[0])


def test_flank_world_is_registered():
    from simulations.worlds import get_world
    c = get_world("flank")[1]()
    assert c.pack_size == 2 and c.flank and not c.signals
    c = get_world("flank_signals")[1]()
    assert c.pack_size == 2 and c.flank and c.signals
