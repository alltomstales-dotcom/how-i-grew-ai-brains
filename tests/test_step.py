from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from simulations import predprey as pp

CFG = pp.WorldConfig()


def stay(n):
    return jnp.zeros((n,), jnp.int32)


def bare():
    """No grown grass anywhere, so grazing never interferes."""
    return jnp.full((CFG.size, CFG.size), CFG.regrow_steps, jnp.int32)


def run(s, prey_act=None, pred_act=None):
    prey_act = stay(CFG.max_prey) if prey_act is None else prey_act
    pred_act = stay(CFG.max_pred) if pred_act is None else pred_act
    return pp.step(CFG, s, prey_act, pred_act)


def test_predator_eats_lowest_index_prey_once():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0), (5, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    r = run(s)
    assert not bool(r.state.prey.alive[0]) and bool(r.state.prey.alive[1])
    assert float(r.state.pred.energy[0]) == 30.0 + CFG.catch_energy - 1.0
    assert int(r.info.catches) == 1
    assert bool(r.prey_done[0]) and float(r.prey_reward[0]) == CFG.death_reward
    assert not bool(r.done)


def test_two_predators_one_prey_single_catch():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(5, 5, 30.0), (5, 5, 30.0)], grass=bare())
    r = run(s)
    assert int(r.info.catches) == 1
    assert float(r.state.pred.energy[0]) == 44.0
    assert float(r.state.pred.energy[1]) == 29.0


def test_moving_onto_prey_catches():
    s = pp.make_state(CFG, prey=[(6, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    pred_act = stay(CFG.max_pred).at[0].set(3)  # east
    r = run(s, pred_act=pred_act)
    assert int(r.info.catches) == 1


def test_grazing_lowest_index_eats_and_starts_regrowth():
    grass = bare().at[5, 5].set(0)
    s = pp.make_state(CFG, prey=[(5, 5, 10.0), (5, 5, 10.0)], pred=[(20, 20, 30.0)], grass=grass)
    r = run(s)
    e = np.asarray(r.state.prey.energy)
    assert e[0] == 10.0 + CFG.grass_energy - 1.0
    assert e[1] == 9.0
    assert int(r.state.grass[5, 5]) == CFG.regrow_steps - 1


def test_grass_timer_counts_down_to_grown():
    grass = bare().at[0, 0].set(1)
    s = pp.make_state(CFG, prey=[(30, 30, 10.0)], pred=[(20, 20, 30.0)], grass=grass)
    r = run(s)
    assert int(r.state.grass[0, 0]) == 0
    assert int(r.state.grass[1, 1]) == CFG.regrow_steps - 1


def test_energy_is_capped():
    grass = bare().at[5, 5].set(0)
    s = pp.make_state(CFG, prey=[(5, 5, 39.0)], pred=[(20, 20, 30.0)], grass=grass)
    r = run(s)
    # 39 + 5 - 1 = 43 is capped at 40, then the prey reproduces (>= 30) and halves to 20.
    assert float(r.state.prey.energy[0]) == CFG.prey_energy_cap / 2


def test_starvation_kills_with_penalty():
    s = pp.make_state(CFG, prey=[(1, 1, 10.0)], pred=[(20, 20, 1.0), (30, 30, 30.0)], grass=bare())
    r = run(s)
    assert not bool(r.state.pred.alive[0]) and bool(r.state.pred.alive[1])
    assert bool(r.pred_done[0]) and not bool(r.pred_done[1])
    assert float(r.pred_reward[0]) == CFG.death_reward
    assert abs(float(r.pred_reward[1]) - CFG.alive_reward) < 1e-7
    assert int(r.info.pred_deaths) == 1
    assert float(r.pred_reward[5]) == 0.0 and bool(r.pred_done[5])  # empty slot


def test_reproduce_splits_energy_into_first_free_slot():
    sp = pp.Species(
        pos=jnp.array([[3, 4], [0, 0], [7, 7]], jnp.int32),
        energy=jnp.array([30.0, 0.0, 10.0]),
        alive=jnp.array([True, False, True]),
    )
    out, born = pp.reproduce(sp, 30.0)
    assert np.asarray(born).tolist() == [True, False, False]
    assert np.asarray(out.alive).tolist() == [True, True, True]
    assert np.asarray(out.energy).tolist() == [15.0, 15.0, 10.0]
    assert np.asarray(out.pos)[1].tolist() == [3, 4]


def test_reproduce_without_free_slot_does_nothing():
    sp = pp.Species(
        pos=jnp.zeros((2, 2), jnp.int32),
        energy=jnp.array([40.0, 40.0]),
        alive=jnp.array([True, True]),
    )
    out, born = pp.reproduce(sp, 30.0)
    assert not np.asarray(born).any()
    assert np.asarray(out.energy).tolist() == [40.0, 40.0]


def test_slot_freed_by_death_and_refilled_still_ends_trajectory():
    # Prey 0 is eaten; prey 1 is fat and gives birth into slot 0 in the same step.
    s = pp.make_state(CFG, prey=[(5, 5, 10.0), (9, 9, 39.0)], pred=[(5, 5, 30.0)], grass=bare())
    r = run(s)
    assert bool(r.state.prey.alive[0])  # the newborn
    assert np.asarray(r.state.prey.pos)[0].tolist() == [9, 9]
    assert bool(r.prey_done[0]) and float(r.prey_reward[0]) == CFG.death_reward
    assert abs(float(r.prey_reward[1]) - (CFG.alive_reward + CFG.birth_reward)) < 1e-6
    assert int(r.info.prey_deaths) == 1 and int(r.info.prey_births) == 1


def test_extinction_ends_episode_and_cuts_all_trajectories():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    r = run(s)
    assert bool(r.done)
    assert bool(r.pred_done[0])


def test_time_limit_ends_episode():
    s = pp.make_state(CFG, prey=[(1, 1, 10.0)], pred=[(20, 20, 30.0)], grass=bare(), t=CFG.max_steps - 1)
    r = run(s)
    assert bool(r.done)


def test_nearest_dist_handles_empty_and_wraps():
    a = pp.Species(jnp.array([[0, 0]], jnp.int32), jnp.ones(1), jnp.array([True]))
    b = pp.Species(jnp.array([[47, 0], [10, 10]], jnp.int32), jnp.ones(2), jnp.array([True, False]))
    total, count = pp.nearest_dist(a, b, 48)
    assert float(total) == 1.0 and int(count) == 1
    nobody = b._replace(alive=jnp.array([False, False]))
    total, count = pp.nearest_dist(a, nobody, 48)
    assert float(total) == 0.0 and int(count) == 0


def test_nearest_dist_excludes_self():
    a = pp.Species(jnp.array([[0, 0], [3, 4]], jnp.int32), jnp.ones(2), jnp.array([True, True]))
    total, count = pp.nearest_dist(a, a, 48, exclude_self=True)
    assert abs(float(total) - 10.0) < 1e-5 and int(count) == 2


def test_autoreset_after_done_gives_fresh_world():
    s = pp.make_state(CFG, prey=[(5, 5, 10.0)], pred=[(5, 5, 30.0)], grass=bare())
    r, prey_obs, pred_obs = pp.step_autoreset(
        CFG, jax.random.PRNGKey(1), s, stay(CFG.max_prey), stay(CFG.max_pred)
    )
    assert bool(r.done)
    assert int(r.state.t) == 0
    assert int(r.state.prey.alive.sum()) == CFG.init_prey
    assert int(r.info.prey_alive) == 0  # info describes the world before the reset
    assert prey_obs.shape == (CFG.max_prey, 9, 9, 4)


def test_vmapped_jitted_step_is_deterministic():
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    states = jax.vmap(partial(pp.reset, CFG))(keys)
    step_v = jax.jit(jax.vmap(partial(pp.step, CFG)))
    pa = jax.random.randint(jax.random.PRNGKey(1), (8, CFG.max_prey), 0, pp.NUM_ACTIONS)
    qa = jax.random.randint(jax.random.PRNGKey(2), (8, CFG.max_pred), 0, pp.NUM_ACTIONS)
    r1 = step_v(states, pa, qa)
    r2 = step_v(states, pa, qa)
    assert r1.state.prey.pos.shape == (8, CFG.max_prey, 2)
    for x, y in zip(jax.tree.leaves(r1), jax.tree.leaves(r2)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
