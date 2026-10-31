"""Stage 4: night, seasons and recurrent brains."""
import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0)
CFG = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(9)


def cfg(**kw):
    return evo.EvoConfig(**{**CFG.__dict__, **kw})


def keen_prey_with_far_plant(c, t):
    # A prey with full sight (radius 4) and a plant 3 cells east of it.
    return evo.make_state(c, prey=[(5, 5, 20.0, 0.0, 1.0, 0.5)], plants=[(8, 5, 0.5, 0.0)], t=t)


def test_night_cuts_sight_to_one_cell():
    c = cfg(day_len=10, night_frac=0.3)  # steps 7, 8, 9 of each day are night
    for t, sees in ((2, True), (8, False), (12, True), (19, False)):
        obs, _ = evo.observe(c, keen_prey_with_far_plant(c, t))
        assert bool(np.asarray(obs[0], np.float32)[4, 7, 0] == 1.0) is sees, t


def test_no_day_length_means_always_day():
    obs, _ = evo.observe(CFG, keen_prey_with_far_plant(CFG, 9))
    assert np.asarray(obs[0], np.float32)[4, 7, 0] == 1.0


def test_plants_out_of_season_cannot_seed():
    c = cfg(season_len=10, seed_rate=1.0, tough_seed_cost=0.0, gene_sigma=0.0, bite_regrow=0.0)
    # Season 0 (t < 10): the left half (x < 6) is in season.
    s = evo.make_state(c, plants=[(2, 5, 1.0, 0.0), (9, 5, 1.0, 0.0)], t=3)
    plant = np.asarray(evo.advance(c, KEY, s, jnp.zeros(8, jnp.int32), jnp.zeros(4, jnp.int32)).state.plant)
    assert plant[:, :6].sum() == 2 and plant[:, 6:].sum() == 1
    s = s._replace(t=jnp.int32(13))  # season 1: the right half is in season
    plant = np.asarray(evo.advance(c, KEY, s, jnp.zeros(8, jnp.int32), jnp.zeros(4, jnp.int32)).state.plant)
    assert plant[:, :6].sum() == 1 and plant[:, 6:].sum() == 2


def test_plants_out_of_season_die_faster():
    c = cfg(season_len=10, die_rate=0.1, plant_base_death=1.0, offseason_death=10.0)
    s = evo.make_state(c, plants=[(2, 5, 0.0, 0.0), (9, 5, 0.0, 0.0)], t=3)
    plant = np.asarray(evo.advance(c, KEY, s, jnp.zeros(8, jnp.int32), jnp.zeros(4, jnp.int32)).state.plant)
    assert plant[5, 9] == 0  # die chance 0.1 * 10 = 1 out of season


def test_seasons_world_is_registered():
    from simulations.worlds import get_world
    module, config_cls = get_world("seasons")
    c = config_cls()
    assert module is evo and c.day_len > 0 and c.season_len > 0


# --- recurrent brains ---------------------------------------------------------------------

def test_feedforward_param_names_are_unchanged():
    from agents.networks import ActorCritic, init_params
    params = init_params(ActorCritic(), jax.random.PRNGKey(0), channels=10)["params"]
    assert set(params) == {"Conv_0", "Conv_1", "Dense_0", "Dense_1", "Dense_2"}


def test_memory_brain_carries_state_and_loads_feedforward_weights(tmp_path):
    from agents.checkpoint import load_brains, save_checkpoint
    from agents.networks import ActorCritic, init_params
    ff, mem = ActorCritic(), ActorCritic(memory=16)
    ff_params = init_params(ff, jax.random.PRNGKey(0), channels=10)
    mem_params = init_params(mem, jax.random.PRNGKey(1), channels=10)
    assert {"gru", "mem_proj"} <= set(mem_params["params"])
    obs = jax.random.uniform(KEY, (3, 9, 9, 10))
    h = jnp.zeros((3, 16))
    logits, value, h1 = mem.apply(mem_params, obs, h)
    assert logits.shape == (3, 5) and value.shape == (3,) and h1.shape == (3, 16)
    assert not np.allclose(np.asarray(h1), 0)
    # A feed-forward brain loaded into a memory brain acts exactly as before.
    save_checkpoint(tmp_path / "ff.msgpack", ff_params, ff_params, {"update": 1})
    prey, _, _ = load_brains(tmp_path / "ff.msgpack", mem_params)
    a = ff.apply(ff_params, obs)
    b = mem.apply(prey, obs, jax.random.normal(KEY, (3, 16)))
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), atol=1e-5)


def test_act_resets_memory_for_empty_slots():
    from agents.networks import ActorCritic, init_params
    from agents.policies import act, init_memory, network_policy, random_policy
    net = ActorCritic(memory=8)
    pi = network_policy(net, init_params(net, jax.random.PRNGKey(0), channels=10))
    obs = jax.random.uniform(KEY, (2, 4, 9, 9, 10))
    alive = jnp.array([[True, False, True, False]] * 2)
    h = jnp.ones((2, 4, 8))
    a, h1 = act(pi, KEY, obs, h, alive)
    assert a.shape == (2, 4) and h1.shape == (2, 4, 8)
    # An empty slot starts from zero memory: same result as passing zeros.
    _, h0 = act(pi, KEY, obs, jnp.zeros((2, 4, 8)), alive)
    np.testing.assert_allclose(np.asarray(h1[:, 1]), np.asarray(h0[:, 1]))
    assert init_memory(pi, (2, 4)).shape == (2, 4, 8)
    assert init_memory(random_policy, (2, 4)).shape == (2, 4, 0)
    a, h = act(random_policy, KEY, obs, init_memory(random_policy, (2, 4)), alive)
    assert a.shape == (2, 4) and h.shape == (2, 4, 0)


def test_memory_trainer_learns_and_carries_state():
    from dataclasses import replace
    from agents.ppo import make_trainer, tiny_configs
    wcfg, tcfg = tiny_configs("seasons")
    trainer = make_trainer(wcfg, replace(tcfg, memory=8, sample_frac=0.5), world=evo)
    runner = trainer.init(jax.random.PRNGKey(0))
    assert runner.prey_h.shape == (4, wcfg.max_prey, 8) and runner.pred_h.shape == (4, wcfg.max_pred, 8)
    before = np.asarray(runner.prey.params["params"]["gru"]["ir"]["kernel"]).copy()
    for _ in range(2):
        runner, metrics = trainer.update(runner)
    assert all(np.isfinite(float(v)) for v in jax.device_get(metrics).values())
    assert not np.array_equal(before, np.asarray(runner.prey.params["params"]["gru"]["ir"]["kernel"]))
    h = np.asarray(runner.prey_h)
    alive = np.asarray(runner.env_state.prey.alive)
    assert np.abs(h[alive]).sum() > 0


def test_memory_and_league_together_are_refused():
    import pytest
    from dataclasses import replace
    from agents.ppo import make_trainer, tiny_configs
    wcfg, tcfg = tiny_configs("evo")
    with pytest.raises(ValueError):
        make_trainer(wcfg, replace(tcfg, memory=8, league_size=2), world=evo)


def test_newborn_in_a_slot_freed_this_step_starts_with_blank_memory():
    # Review finding: a slot whose animal died and was refilled in the same step is alive before and
    # after, and the newborn inherited the dead animal's memory. A slot marked done is never carried.
    import types
    from dataclasses import replace
    from agents.ppo import make_trainer, tiny_configs
    wcfg, tcfg = tiny_configs("seasons")

    def batch_step(cfg, key, states, prey_act, pred_act):
        res, prey_obs, pred_obs = evo.batch_step(cfg, key, states, prey_act, pred_act)
        alive = res.state.prey.alive.at[:, 0].set(True)  # slot 0 is filled every step ...
        res = res._replace(state=res.state._replace(prey=res.state.prey._replace(alive=alive)),
                           prey_done=res.prey_done.at[:, 0].set(True))  # ... by a newborn
        return res, prey_obs, pred_obs

    world = types.SimpleNamespace(**{k: getattr(evo, k) for k in dir(evo) if not k.startswith("__")})
    world.batch_step = batch_step
    trainer = make_trainer(wcfg, replace(tcfg, memory=8), world=world)
    runner, _ = trainer.update(trainer.init(jax.random.PRNGKey(0)))
    h = np.asarray(runner.prey_h)
    assert np.asarray(runner.env_state.prey.alive)[:, 0].all()
    assert np.abs(h[:, 0]).sum() == 0
    assert np.abs(h[:, 1:]).sum() > 0


def test_forget_clears_memory_of_slots_that_ended():
    from agents.policies import forget
    mem = (jnp.ones((2, 3, 4)), jnp.ones((2, 2, 4)))
    prey_done = jnp.array([[True, False, False], [False, False, True]])
    pred_done = jnp.array([[False, True], [False, False]])
    prey_h, pred_h = forget(mem, prey_done, pred_done)
    np.testing.assert_array_equal(np.asarray(prey_h).sum(-1), np.where(np.asarray(prey_done), 0, 4))
    np.testing.assert_array_equal(np.asarray(pred_h).sum(-1), np.where(np.asarray(pred_done), 0, 4))


def test_evaluation_forgets_dead_and_newborn_slots_but_not_a_whole_ended_world():
    import types
    from agents.policies import forget_after
    mem = (jnp.ones((2, 3, 4)), jnp.ones((2, 2, 4)))
    res = types.SimpleNamespace(done=jnp.array([False, True]),
                                prey_done=jnp.array([[True, False, False], [True, True, True]]),
                                pred_done=jnp.array([[False, False], [True, True]]))
    prey_h, pred_h = forget_after(mem, res)
    assert np.asarray(prey_h).sum((-1)).tolist() == [[0, 4, 4], [4, 4, 4]]
    assert np.asarray(pred_h).sum((-1)).tolist() == [[4, 4], [4, 4]]
