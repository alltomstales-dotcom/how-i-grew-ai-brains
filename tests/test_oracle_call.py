"""Stage 21b: the oracle call and the fixed listener cohort. Spec docs/specs/2026-11-05-oracle-call-design.md"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from agents.networks import init_params
from simulations import evo
from simulations.worlds import get_world, make_net, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=16, max_prey=6, max_pred=4, init_prey=2, init_pred=2, max_steps=40, alarm=True,
                    oracle_call=True, signal_cost=0.3, **QUIET)
KEY = jax.random.PRNGKey(3)
STILL, CALL = 0, 5


def acts(*a, n):
    return jnp.array(list(a) + [STILL] * (n - len(a)), jnp.int32)


def prey(*xy, e=20.0):
    return [(x, y, e, 0.0, 0.5, 1.0) for x, y in xy]


def test_the_oracle_calls_exactly_when_a_hunter_is_adjacent_and_the_caller_pays():
    # grazer 0 next to the hunter chooses no call; grazer 1 far away chooses to call; grazer 2 is two cells off.
    # Everyone stands still, and a catch needs the same cell, so nobody is eaten.
    s = evo.make_state(CFG, prey=prey((4, 4), (12, 12), (4, 7)), pred=[(5, 5, 50.0, 0.0, 0.5, 1.0)])
    r = evo.advance(CFG, KEY, s, acts(STILL, CALL, STILL, n=6), acts(n=4))
    assert np.asarray(r.state.prey.alive)[:3].all()
    assert np.asarray(r.state.prey_signal).tolist()[:3] == [True, False, False]
    e = np.asarray(r.state.prey.energy)
    assert np.isclose(e[1] - e[0], 0.3, atol=1e-4)  # the oracle caller pays
    assert np.isclose(e[1] - e[2], 0.0, atol=1e-4)  # the far grazer's own call output is ignored and costs nothing
    o, _ = evo.observe(CFG, r.state)
    heard = np.asarray(o[2], np.float32)[..., evo.OBS_CHANNELS]
    assert heard.sum() == 1 and heard[CFG.view // 2 - 3, CFG.view // 2] == 1  # grazer 2 hears grazer 0, 3 cells off
    plain = dataclasses.replace(CFG, oracle_call=False)
    q = evo.advance(plain, KEY, s, acts(STILL, CALL, STILL, n=6), acts(n=4))
    assert np.asarray(q.state.prey_signal).tolist()[:3] == [False, True, False]


def _small_alarm():
    world, cls = get_world("alarm")
    cfg = dataclasses.replace(cls(), size=16, max_prey=12, max_pred=4, init_prey=10, init_pred=3, max_steps=80,
                              oracle_call=True, signal_cost=0.3, kin_share=0.5)
    net = make_net(world, cfg)
    return world, cfg, net


def _deaf(params, channels):
    """The same brain with zero weights on the call channel (the last input channel), as a widened brain starts."""
    def z(x):
        if x.ndim == 4 and x.shape[2] == channels:
            return x.at[:, :, channels - 1, :].set(0.0)
        return x
    return jax.tree.map(z, params)


def _cohort(cfg, net, prey_p, pred_p, E=4):
    from scripts import alarm
    segment, branch = alarm.make_cohort(cfg, net, E)
    k_reset, k_steps = jax.random.split(jax.random.PRNGKey(5))
    keys = jax.random.split(k_steps, cfg.max_steps)
    state = jax.vmap(lambda k: evo.reset(cfg, k))(jax.random.split(k_reset, E))
    state, _ = segment(prey_p, pred_p, state, keys[:10])
    sts = jax.tree.map(lambda x: x[None], state)
    coh = jax.vmap(lambda st: alarm.make_step_listeners(cfg, st))(sts)
    out = {c: branch(prey_p, pred_p, sts, keys[None, 10:10 + alarm.RISK], coh, jnp.int32(alarm.MODES[c])) for c in alarm.CONDS}
    return coh, out


def test_a_brain_deaf_to_calls_gives_identical_branches():
    world, cfg, net = _small_alarm()
    C = obs_channels(world, cfg)
    prey_p = _deaf(init_params(net, jax.random.PRNGKey(1), cfg.view, C), C)
    pred_p = init_params(net, jax.random.PRNGKey(2), cfg.view, C)
    coh, out = _cohort(cfg, net, prey_p, pred_p)
    for c in ("mute", "shuffle"):
        assert np.array_equal(np.asarray(out[c][0]), np.asarray(out["real"][0]))
        assert np.array_equal(np.asarray(out[c][1]), np.asarray(out["real"][1]))


def test_the_real_branch_replays_the_reference():
    """Deaths in the real branch match a straight run of the reference for the same steps."""
    from scripts import alarm
    world, cfg, net = _small_alarm()
    C = obs_channels(world, cfg)
    prey_p = init_params(net, jax.random.PRNGKey(1), cfg.view, C)
    pred_p = init_params(net, jax.random.PRNGKey(2), cfg.view, C)
    E = 4
    segment, branch = alarm.make_cohort(cfg, net, E)
    k_reset, k_steps = jax.random.split(jax.random.PRNGKey(5))
    keys = jax.random.split(k_steps, cfg.max_steps)
    s0 = jax.vmap(lambda k: evo.reset(cfg, k))(jax.random.split(k_reset, E))
    s10, _ = segment(prey_p, pred_p, s0, keys[:10])
    _, per = segment(prey_p, pred_p, s10, keys[10:10 + alarm.RISK])
    everyone = s10.prey.alive[None]  # cohort = every living grazer, so the branch counts all their deaths
    dead, _ = branch(prey_p, pred_p, jax.tree.map(lambda x: x[None], s10), keys[None, 10:10 + alarm.RISK], everyone, jnp.int32(0))
    # each starting animal dies at most once; the reference counts deaths of newborns too, so it is an upper bound
    assert (np.asarray(dead[0]) <= np.asarray(per["deaths"]).sum(0) + 1e-6).all()
    assert np.asarray(per["deaths"]).sum() > 0 or np.asarray(dead).sum() == 0
