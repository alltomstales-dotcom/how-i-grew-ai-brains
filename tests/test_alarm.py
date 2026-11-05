"""Stage 21: grazers' alarm calls, hunters' eavesdropping, the kin reward and dynasties."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulations import evo
from simulations.worlds import get_world, make_net, num_actions, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=16, max_prey=6, max_pred=4, init_prey=2, init_pred=2, max_steps=40, alarm=True, **QUIET)
KEY = jax.random.PRNGKey(3)
STILL, EAST = 0, 3
CALL = 5  # actions 5-9 are moves 0-4 with the call on
R = CFG.view // 2
HEARD = evo.OBS_CHANNELS  # the call channel


def acts(*a, n):
    return jnp.array(list(a) + [STILL] * (n - len(a)), jnp.int32)


def prey(*xy, e=20.0):
    return [(x, y, e, 0.0, 0.5, 1.0) for x, y in xy]  # too hungry to breed


def heard(obs, i):
    return np.asarray(obs[i], np.float32)[..., HEARD]


def call_step(cfg, s, prey_acts, pred_acts=None):
    return evo.advance(cfg, KEY, s, prey_acts, pred_acts if pred_acts is not None else acts(n=cfg.max_pred)).state


def test_alarm_world_shapes():
    world, cls = get_world("alarm")
    c = cls()
    assert c.alarm and c.dynasties and c.stealth_radius == 1 and c.gene_sigma == 0 and c.plant_sigma == 0
    assert obs_channels(world, c) == evo.OBS_CHANNELS + 1 and num_actions(world, c) == 10
    with pytest.raises(ValueError):
        evo.observe(evo.EvoConfig(**{**CFG.__dict__, "signals": True}), evo.make_state(CFG))


def test_grazers_hear_others_within_four_cells_but_not_themselves():
    s = evo.make_state(CFG, prey=prey((4, 4), (8, 4), (9, 4)))  # 0 and 1 four apart; 2 five from 0
    s = call_step(CFG, s, acts(CALL, CALL, STILL, n=6))
    assert np.asarray(s.prey_signal).tolist()[:3] == [True, True, False]
    o, _ = evo.observe(CFG, s)
    h0, h1, h2 = heard(o, 0), heard(o, 1), heard(o, 2)
    assert h0.sum() == 1 and h0[R, R + 4] == 1  # hears grazer 1 four cells east, not itself
    assert h1.sum() == 1 and h1[R, R - 4] == 1
    assert h2.sum() == 1 and h2[R, R - 1] == 1  # grazer 0 is five away: out of hearing


def test_calls_cost_the_caller_and_hunters_calls_do_nothing():
    c = evo.EvoConfig(**{**CFG.__dict__, "signal_cost": 0.3})
    s = evo.make_state(c, prey=prey((4, 4), (12, 12)), pred=[(2, 10, 50.0, 0.0, 0.5, 1.0), (10, 2, 50.0, 0.0, 0.5, 1.0)])
    r = evo.advance(c, KEY, s, acts(CALL, STILL, n=6), acts(CALL, STILL, n=4))
    e = np.asarray(r.state.prey.energy)
    assert np.isclose(e[1] - e[0], 0.3, atol=1e-4)
    q = np.asarray(r.state.pred.energy)
    assert np.isclose(q[0], q[1], atol=1e-5) and not np.asarray(r.state.pred_signal).any()
    assert np.asarray(r.state.pred.pos)[0].tolist() == [2, 10]  # a hunter's call action is just its move


def test_hunters_hear_grazer_calls_only_when_eavesdropping_and_deaf_grazers_hear_nothing():
    s = evo.make_state(CFG, prey=prey((4, 4), (6, 4)), pred=[(5, 7, 50.0, 0.0, 0.5, 1.0)])
    s = call_step(CFG, s, acts(CALL, n=6))
    _, q = evo.observe(CFG, s)
    assert heard(q, 0).sum() == 0
    ears = evo.EvoConfig(**{**CFG.__dict__, "eavesdrop": True})
    p, q = evo.observe(ears, s)
    assert heard(q, 0).sum() == 1 and heard(q, 0)[R - 3, R - 1] == 1  # grazer 0 at (4, 4) from (5, 7)
    deaf = evo.EvoConfig(**{**CFG.__dict__, "prey_deaf": True})
    assert heard(evo.observe(deaf, s)[0], 1).sum() == 0 and heard(p, 1).sum() == 1


def test_replacement_calls_for_the_shuffled_test():
    s = evo.make_state(CFG, prey=prey((4, 4), (6, 4), (8, 4)), pred=[(5, 7, 50.0, 0.0, 0.5, 1.0)])
    s = call_step(CFG, s, acts(CALL, n=6))
    ears = evo.EvoConfig(**{**CFG.__dict__, "eavesdrop": True})
    swap = jnp.zeros(6, bool).at[2].set(True)
    p, q = evo.observe(ears, s, prey_calls=swap, pred_calls=swap)
    assert heard(p, 0)[R, R + 4] == 1 and heard(p, 0).sum() == 1
    assert heard(p, 2).sum() == 0  # its own (replacement) call isn't heard either
    assert heard(q, 0)[R - 3, R + 3] == 1 and heard(q, 0).sum() == 1


def test_eaten_callers_are_silent_and_newborns_too():
    c = evo.EvoConfig(**{**CFG.__dict__, "prey_repro_energy": 10.0})
    s = evo.make_state(c, prey=[(4, 4, 20.0, 0.0, 0.5, 1.0), (10, 10, 39.0, 0.0, 0.5, 0.0)],
                       pred=[(4, 4, 50.0, 0.0, 0.5, 1.0)])
    r = evo.advance(c, KEY, s, acts(CALL, CALL, n=6), acts(STILL, n=4))
    alive, sig = np.asarray(r.state.prey.alive), np.asarray(r.state.prey_signal)
    assert alive[0] and alive[1]  # grazer 0 was eaten; the newborn took its slot
    assert sig.tolist()[:3] == [False, True, False]
    assert not evo.reset(c, KEY).prey_signal.any()


def test_kin_share_adds_kin_deaths_within_hearing():
    c = evo.EvoConfig(**{**CFG.__dict__, "kin_share": 0.5})
    s = evo.make_state(c, prey=prey((4, 4), (6, 4), (9, 4), (4, 12)), pred=[(4, 4, 50.0, 0.0, 0.5, 1.0)])
    s = s._replace(prey=s.prey._replace(tag=jnp.array([1, 1, 1, 2, 0, 0], jnp.int32)))
    r = evo.advance(c, KEY, s, acts(n=6), acts(n=4))
    rew = np.asarray(r.prey_reward)
    base = c.alive_reward
    assert np.isclose(rew[0], c.death_reward)  # eaten: its own penalty only
    assert np.isclose(rew[1], base + 0.5 * c.death_reward)  # kin, two cells from the death
    assert np.isclose(rew[2], base)  # kin, but five cells away
    assert np.isclose(rew[3], base)  # another dynasty
    plain = evo.advance(CFG, KEY, s, acts(n=6), acts(n=4))
    assert np.isclose(np.asarray(plain.prey_reward)[1], base)


def test_dynasties_give_each_founder_its_own_tag_at_every_reset():
    c = evo.EvoConfig(**{**CFG.__dict__, "dynasties": True})
    s = evo.reset(c, KEY)
    assert np.asarray(s.prey.tag).tolist() == list(range(1, 7)) and np.asarray(s.pred.tag).tolist() == [1, 2, 3, 4]
    pool = evo.gene_pool(jax.tree.map(lambda a: a[None], s))
    pool = pool._replace(prey_tag=jnp.zeros_like(pool.prey_tag))
    assert np.asarray(evo.reset(c, KEY, pool).prey.tag).tolist() == list(range(1, 7))
    assert (np.asarray(evo.reset(CFG, KEY).prey.tag) == 0).all()


def test_children_inherit_the_dynasty():
    c = evo.EvoConfig(**{**CFG.__dict__, "prey_repro_energy": 10.0, "dynasties": True})
    s = evo.make_state(c, prey=[(4, 4, 39.0, 0.0, 0.5, 0.0)])
    s = s._replace(prey=s.prey._replace(tag=jnp.array([7, 0, 0, 0, 0, 0], jnp.int32)))
    r = evo.advance(c, KEY, s, acts(n=6), acts(n=4))
    assert np.asarray(r.state.prey.alive)[1] and int(r.state.prey.tag[1]) == 7


def test_five_action_brains_widen_to_ten_with_the_same_moves():
    from agents.checkpoint import load_brains, save_checkpoint
    from agents.networks import init_params
    import tempfile, pathlib
    base = evo.LastingStealthConfig()
    world, alarm = get_world("alarm")[0], evo.AlarmConfig()
    net5, net10 = make_net(world, base), make_net(world, alarm)
    p5 = init_params(net5, jax.random.PRNGKey(1), base.view, obs_channels(world, base))
    q5 = init_params(net5, jax.random.PRNGKey(2), base.view, obs_channels(world, base))
    # Non-trivial logits, so the test means something.
    p5 = jax.tree.map(lambda x: x * 500 if x.ndim == 2 and x.shape[1] == 5 else x, p5)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "c.msgpack"
        save_checkpoint(path, p5, q5, {"update": 1})
        template = init_params(net10, jax.random.PRNGKey(0), alarm.view, obs_channels(world, alarm))
        p10, q10, _ = load_brains(path, template)
    s = evo.reset(alarm, KEY)
    s = s._replace(prey_signal=s.prey.alive)  # calls in the air: a widened brain starts deaf to them
    o10, _ = evo.observe(alarm, s)
    o5 = o10[..., :-1]
    l5 = jax.nn.softmax(net5.apply(p5, o5)[0].astype(jnp.float32))
    l10 = jax.nn.softmax(net10.apply(p10, o10)[0].astype(jnp.float32))
    moves = l10[:, :5] + l10[:, 5:]
    assert np.allclose(moves, l5, atol=1e-5)
    assert np.allclose(l10[:, 5:].sum(-1), 0.5, atol=1e-5)
    assert float(jnp.abs(l5 - 0.2).max()) > 0.05
    assert np.allclose(net10.apply(p10, o10)[1], net5.apply(p5, o5)[1], atol=1e-4)


def test_shuffle_keeps_each_strata_count_and_moves_calls_only_within_it():
    from scripts.alarm import shuffle_within
    groups = jnp.array([0, 0, 0, 1, 1, 2, 99, 99])
    bits = jnp.array([True, False, False, True, True, False, False, False])
    seen = set()
    for i in range(40):
        out = np.asarray(shuffle_within(jax.random.PRNGKey(i), groups, bits))
        for g in (0, 1, 2, 99):
            m = np.asarray(groups) == g
            assert out[m].sum() == np.asarray(bits)[m].sum()
        seen.add(tuple(out[:3]))
    assert len(seen) == 3  # the one call in group 0 lands on each member


def test_risk_counts_follow_each_animal_to_its_death():
    from scripts.alarm import risk_counts
    T, E, N = 120, 1, 2
    listener = np.zeros((T, E, N), bool)
    died = np.zeros((T, E, N), bool)
    listener[10, 0, 0] = listener[60, 0, 0] = True  # slot 0 dies at 55: 45 steps after 10, none after 60
    died[55, 0, 0] = True
    listener[5, 0, 1] = True  # slot 1 never dies
    listener[80, 0, 1] = True  # too late to follow for 50 steps
    z = np.zeros_like(listener)
    out = risk_counts(listener, z, z, died)
    assert out["listener_n"].tolist() == [3] and out["listener_d"].tolist() == [1]
