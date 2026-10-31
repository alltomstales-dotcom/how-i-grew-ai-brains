import jax
import jax.numpy as jnp
import numpy as np

from simulations import predprey as pp

CFG = pp.WorldConfig()


def test_reset_counts_and_energy():
    s = pp.reset(CFG, jax.random.PRNGKey(0))
    assert int(s.prey.alive.sum()) == CFG.init_prey
    assert int(s.pred.alive.sum()) == CFG.init_pred
    assert np.all(np.asarray(s.prey.energy)[: CFG.init_prey] == CFG.prey_start_energy)
    assert np.all(np.asarray(s.prey.energy)[CFG.init_prey:] == 0.0)
    assert s.prey.pos.shape == (CFG.max_prey, 2)
    assert s.grass.shape == (CFG.size, CFG.size)
    assert int(s.t) == 0
    grown = float((s.grass == 0).mean())
    assert 0.4 < grown < 0.6


def test_move_wraps_on_torus():
    pos = jnp.array([[0, 0], [0, 0], [47, 47], [47, 47], [5, 5]], jnp.int32)
    acts = jnp.array([4, 1, 3, 2, 0], jnp.int32)  # west, north, east, south, stay
    out = np.asarray(pp.move(pos, acts, 48))
    assert out.tolist() == [[47, 0], [0, 47], [0, 47], [47, 0], [5, 5]]


def test_make_state_places_agents():
    s = pp.make_state(CFG, prey=[(3, 4, 10.0)], pred=[(5, 6, 20.0)])
    assert np.asarray(s.prey.pos)[0].tolist() == [3, 4]
    assert bool(s.prey.alive[0]) and not bool(s.prey.alive[1])
    assert float(s.pred.energy[0]) == 20.0
    assert int((s.grass == 0).sum()) == CFG.size * CFG.size


def test_observation_sees_neighbour_with_offset():
    # Predator at (7, 5), prey at (5, 5): the prey is 2 cells west -> row 4 (dy=0), column 2 (dx=-2).
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(7, 5, 30.0)])
    prey_obs, pred_obs = pp.observe(CFG, s)
    assert prey_obs.shape == (CFG.max_prey, 9, 9, 4)
    assert pred_obs.shape == (CFG.max_pred, 9, 9, 4)
    w = np.asarray(pred_obs[0], dtype=np.float32)
    assert w[4, 2, 1] == 1.0  # prey channel
    assert w[:, :, 1].sum() == 1.0
    assert w[4, 4, 2] == 1.0  # itself, in the predator channel
    assert np.allclose(w[:, :, 3], 30.0 / CFG.pred_energy_cap, atol=1e-2)
    assert np.all(w[:, :, 0] == 1.0)  # all grass grown


def test_observation_wraps():
    s = pp.make_state(CFG, prey=[(47, 47, 10.0)], pred=[(0, 0, 30.0)])
    _, pred_obs = pp.observe(CFG, s)
    w = np.asarray(pred_obs[0], dtype=np.float32)
    assert w[3, 3, 1] == 1.0


def test_dead_slots_observe_zeros():
    s = pp.make_state(CFG, prey=[(1, 1, 10.0)], pred=[(2, 2, 30.0)])
    prey_obs, _ = pp.observe(CFG, s)
    assert float(jnp.abs(prey_obs[1].astype(jnp.float32)).sum()) == 0.0
