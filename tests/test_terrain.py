"""Stage 3a: walls, burrows and portable brains."""
import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0)
CFG = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(5)
WALL, BURROW = evo.WALL, evo.BURROW
EAST, STAY = 3, 0


def terrain(cells):
    t = np.zeros((CFG.size, CFG.size), np.int8)
    for x, y, kind in cells:
        t[y, x] = kind
    return t


def acts(n, *moves):
    a = np.zeros(n, np.int32)
    a[:len(moves)] = moves
    return jnp.asarray(a)


def test_plain_evo_world_has_no_terrain():
    s = evo.reset(CFG, KEY)
    assert s.terrain.shape == (12, 12) and not np.asarray(s.terrain).any()


def test_terrain_config_draws_walls_and_burrows():
    c = evo.TerrainConfig()
    s = evo.reset(c, KEY)
    t = np.asarray(s.terrain)
    assert (t == BURROW).sum() >= c.burrows // 2 and (t == WALL).sum() >= c.walls * 2
    plant = np.asarray(s.plant)
    assert not plant[t != 0].any()
    for sp in (s.prey, s.pred):
        xy = np.asarray(sp.pos)[np.asarray(sp.alive)]
        assert (t[xy[:, 1], xy[:, 0]] == 0).all()


def test_terrain_is_redrawn_at_each_reset():
    c = evo.TerrainConfig()
    a, b = evo.reset(c, jax.random.PRNGKey(1)), evo.reset(c, jax.random.PRNGKey(2))
    assert not np.array_equal(np.asarray(a.terrain), np.asarray(b.terrain))


def test_walls_block_everyone_and_burrows_block_predators():
    s = evo.make_state(CFG, prey=[(2, 2, 20.0, 0.0, 0.0, 1.0), (2, 6, 20.0, 0.0, 0.0, 1.0)],
                       pred=[(2, 9, 30.0, 0.0, 0.0, 1.0)],
                       terrain=terrain([(3, 2, WALL), (3, 6, BURROW), (3, 9, BURROW)]))
    out = evo.advance(CFG, KEY, s, acts(8, EAST, EAST), acts(4, EAST)).state
    assert np.asarray(out.prey.pos[:2]).tolist() == [[2, 2], [3, 6]]
    assert np.asarray(out.pred.pos[0]).tolist() == [2, 9]


def test_fast_second_step_is_blocked_too():
    s = evo.make_state(CFG, prey=[(2, 2, 20.0, 1.0, 0.0, 1.0)], terrain=terrain([(4, 2, WALL)]))
    out = evo.advance(CFG, KEY, s, acts(8, EAST), acts(4)).state
    assert np.asarray(out.prey.pos[0]).tolist() == [3, 2]


def test_prey_in_a_burrow_are_safe():
    s = evo.make_state(CFG, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0)], pred=[(2, 3, 30.0, 0.0, 0.0, 1.0)],
                       terrain=terrain([(3, 3, BURROW)]))
    res = evo.advance(CFG, KEY, s, acts(8), acts(4, EAST))
    assert bool(res.state.prey.alive[0]) and int(res.info.catches) == 0


def test_plants_never_seed_onto_terrain():
    c = evo.EvoConfig(**{**CFG.__dict__, "seed_rate": 1.0, "tough_seed_cost": 0.0, "germinate_rate": 1.0})
    s = evo.make_state(c, plants=[(5, 5, 1.0, 0.0)],
                       terrain=terrain([(4, 5, WALL), (6, 5, BURROW), (5, 4, WALL), (5, 6, BURROW)]))
    for i in range(3):
        s = evo.advance(c, jax.random.PRNGKey(i), s, acts(8), acts(4)).state
    t, plant = np.asarray(s.terrain), np.asarray(s.plant)
    assert not plant[t != 0].any() and plant[t == 0].all()


def test_observation_appends_wall_and_burrow_channels():
    assert evo.OBS_CHANNELS == 11  # size (Stage 5a) comes after terrain
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 1.0, 0.5)], terrain=terrain([(6, 5, WALL), (5, 3, BURROW)]))
    prey_obs, _ = evo.observe(CFG, s)
    o = np.asarray(prey_obs[0], np.float32)  # (9, 9, 11), centre (4, 4) = the animal
    assert o.shape == (9, 9, 11)
    assert o[4, 5, 8] == 1 and o[4, 5, 9] == 0
    assert o[2, 4, 9] == 1 and o[2, 4, 8] == 0
    assert o[..., 8].sum() == 1 and o[..., 9].sum() == 1


def test_burrow_metrics_split_threatened_and_safe_prey():
    # Prey 0 is in a burrow with a predator 2 cells away; prey 1 is in the open, far from it.
    s = evo.make_state(CFG, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0), (9, 9, 20.0, 0.0, 0.0, 1.0)],
                       pred=[(5, 3, 30.0, 0.0, 0.0, 1.0)], terrain=terrain([(3, 3, BURROW)]))
    info = evo.advance(CFG, KEY, s, acts(8), acts(4)).info
    assert int(info.prey_near) == 1 and int(info.prey_near_burrow) == 1
    assert int(info.prey_far) == 1 and int(info.prey_far_burrow) == 0
    m = evo.extra_metrics(jax.tree.map(lambda x: x[None], info))
    assert float(m["burrow_use_near"]) == 1.0 and float(m["burrow_use_far"]) == 0.0


def test_terrain_world_is_registered():
    from simulations.worlds import get_world
    module, config_cls = get_world("terrain")
    assert module is evo and config_cls is evo.TerrainConfig


def test_render_draws_walls_and_burrows():
    from viz.render import BURROW_RGB, CELL, HEADER, WALL_RGB, render_frame
    s = evo.make_state(CFG, terrain=terrain([(1, 1, WALL), (2, 1, BURROW)]))
    img = render_frame(CFG, s)
    row = HEADER + 1 * CELL + 1
    assert img[row, 1 * CELL + 1].tolist() == list(WALL_RGB)
    assert img[row, 2 * CELL + 1].tolist() == list(BURROW_RGB)


def test_loaded_brain_ignores_new_channels_until_they_light_up(tmp_path):
    from agents.checkpoint import load_brains, save_checkpoint
    from agents.networks import ActorCritic, init_params
    net = ActorCritic()
    old_prey = init_params(net, jax.random.PRNGKey(0), channels=8)
    old_pred = init_params(net, jax.random.PRNGKey(1), channels=8)
    save_checkpoint(tmp_path / "c.msgpack", old_prey, old_pred, {"update": 7})
    template = init_params(net, jax.random.PRNGKey(2), channels=10)
    prey, pred, meta = load_brains(tmp_path / "c.msgpack", template)
    assert meta["update"] == 7
    obs8 = jax.random.uniform(jax.random.PRNGKey(3), (5, 9, 9, 8))
    obs10 = jnp.concatenate([obs8, jnp.zeros((5, 9, 9, 2))], -1)
    for old, new in ((old_prey, prey), (old_pred, pred)):
        a, b = net.apply(old, obs8), net.apply(new, obs10)
        np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), atol=1e-5)
        np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), atol=1e-5)
    # The padded weights exist, so the brain can learn to use terrain.
    assert [x.shape for x in jax.tree.leaves(new)] == [x.shape for x in jax.tree.leaves(template)]


def test_burrow_use_counts_threatened_and_safe_prey():
    from agents.evaluate import burrow_use
    from agents.policies import random_policy
    c = evo.TerrainConfig(size=16, max_prey=16, max_pred=8, init_prey=8, init_pred=4, max_steps=30, burrows=40)
    out = burrow_use(c, random_policy, random_policy, KEY, 4)
    assert set(out) == {"near", "far", "ratio"}
    assert 0.0 <= out["far"] <= 1.0 and (out["near"] is None or 0.0 <= out["near"] <= 1.0)


def test_burrow_verdict_needs_threat_driven_hiding():
    import scripts.eval as ev
    random = {"near": 0.05, "far": 0.05, "ratio": 1.0}
    assert ev.burrow_verdict({"near": 0.20, "far": 0.05, "ratio": 4.0}, random) is True
    # Hiding all the time is not the behaviour we are looking for.
    assert ev.burrow_verdict({"near": 0.30, "far": 0.30, "ratio": 1.0}, random) is False
    assert ev.burrow_verdict({"near": None, "far": 0.1, "ratio": None}, random) is False
