import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.networks import ActorCritic, init_params
from agents.ppo import (
    TrainConfig,
    compute_gae,
    make_trainer,
    masked_normalize,
    ppo_loss,
    tiny_configs,
)


def numpy_gae(r, v, d, last, gamma, lam):
    T = r.shape[0]
    adv = np.zeros_like(r)
    gae = np.zeros_like(last)
    next_v = last
    for t in reversed(range(T)):
        nd = 1.0 - d[t].astype(np.float64)
        delta = r[t] + gamma * next_v * nd - v[t]
        gae = delta + gamma * lam * nd * gae
        adv[t] = gae
        next_v = v[t]
    return adv


def test_gae_matches_numpy_reference():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(6, 3)).astype(np.float32)
    v = rng.normal(size=(6, 3)).astype(np.float32)
    d = np.zeros((6, 3), bool)
    d[2, 0] = True
    d[5, 1] = True
    last = rng.normal(size=3).astype(np.float32)
    adv, ret = compute_gae(jnp.asarray(r), jnp.asarray(v), jnp.asarray(d), jnp.asarray(last), 0.99, 0.95)
    ref = numpy_gae(r, v, d, last, 0.99, 0.95)
    np.testing.assert_allclose(np.asarray(adv), ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), ref + v, rtol=1e-5, atol=1e-5)


def test_masked_normalize_ignores_masked_entries():
    x = jnp.array([1.0, 2.0, 3.0, 1000.0])
    mask = jnp.array([True, True, True, False])
    out = np.asarray(masked_normalize(x, mask))
    assert abs(out[:3].mean()) < 1e-5
    assert abs(out[:3].std() - 1.0) < 1e-3


def test_ppo_loss_policy_term_zero_when_unchanged_and_adv_zero():
    net = ActorCritic()
    params = init_params(net, jax.random.PRNGKey(0), 9)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (8, 9, 9, 4))
    logits, value = net.apply(params, obs)
    action = jnp.zeros((8,), jnp.int32)
    logp = jax.nn.log_softmax(logits)[:, 0]
    mb = {
        "obs": obs,
        "action": action,
        "logp": logp,
        "mask": jnp.ones((8,), bool),
        "adv": jnp.zeros((8,)),
        "ret": value,
    }
    _, aux = ppo_loss(params, mb, net.apply, TrainConfig())
    assert abs(float(aux["pg_loss"])) < 1e-6
    assert abs(float(aux["v_loss"])) < 1e-6
    assert float(aux["entropy"]) > 1.5  # near log(5) = 1.609 at init


def test_minibatch_divisibility_is_checked():
    wcfg, tcfg = tiny_configs()
    bad = TrainConfig(num_envs=3, rollout_len=1, num_minibatches=5)
    with pytest.raises(ValueError):
        make_trainer(wcfg, bad)


def test_trainer_smoke_tiny():
    wcfg, tcfg = tiny_configs()
    trainer = make_trainer(wcfg, tcfg)
    runner = trainer.init(jax.random.PRNGKey(0))
    before = jax.tree.leaves(runner.prey.params)[0].copy()
    for _ in range(3):
        runner, metrics = trainer.update(runner)
    metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
    expected = {
        "prey_alive", "pred_alive", "catches_per_step", "prey_births_per_step",
        "pred_births_per_step", "prey_deaths_per_step", "pred_deaths_per_step",
        "prey_pred_dist", "pred_cluster_dist", "prey_reward", "pred_reward",
        "prey_pg_loss", "prey_v_loss", "prey_entropy", "pred_pg_loss", "pred_v_loss", "pred_entropy",
    }
    assert expected <= set(metrics)
    assert all(math.isfinite(v) for v in metrics.values())
    after = jax.tree.leaves(runner.prey.params)[0]
    assert not np.array_equal(np.asarray(before), np.asarray(after))


def test_default_num_envs_fits_measured_memory():
    # 1024 envs ran out of memory on a DGX Spark's 13 GB XLA cap (smoke-1);
    # 512 peaked at 6.1 GB (smoke-2).
    assert TrainConfig().num_envs == 512


def test_no_learn_keeps_brains_fixed_but_still_steps_the_world():
    # The random-brain control: worlds run and evolve, brains never change.
    from dataclasses import replace
    wcfg, tcfg = tiny_configs("evo")
    from simulations import evo
    trainer = make_trainer(wcfg, replace(tcfg, learn=False), world=evo)
    runner = trainer.init(jax.random.PRNGKey(0))
    before = [np.asarray(x).copy() for x in jax.tree.leaves((runner.prey.params, runner.pred.params))]
    t0 = int(runner.env_state.t[0])
    for _ in range(2):
        runner, metrics = trainer.update(runner)
    after = [np.asarray(x) for x in jax.tree.leaves((runner.prey.params, runner.pred.params))]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert int(runner.env_state.t[0]) != t0
    assert "prey_speed" in metrics and "prey_pg_loss" not in metrics


def test_one_species_can_learn_while_the_other_stays_frozen():
    # For the predator-stall test: predators learn against prey whose brains never change.
    from dataclasses import replace
    from simulations import evo
    wcfg, tcfg = tiny_configs("evo")
    trainer = make_trainer(wcfg, replace(tcfg, learners="pred"), world=evo)
    runner = trainer.init(jax.random.PRNGKey(0))
    prey0 = [np.asarray(x).copy() for x in jax.tree.leaves(runner.prey.params)]
    pred0 = [np.asarray(x).copy() for x in jax.tree.leaves(runner.pred.params)]
    for _ in range(2):
        runner, metrics = trainer.update(runner)
    assert all(np.array_equal(a, np.asarray(b)) for a, b in zip(prey0, jax.tree.leaves(runner.prey.params)))
    assert not all(np.array_equal(a, np.asarray(b)) for a, b in zip(pred0, jax.tree.leaves(runner.pred.params)))
    assert "pred_pg_loss" in metrics and "prey_pg_loss" not in metrics
    import pytest
    with pytest.raises(ValueError):
        make_trainer(wcfg, replace(tcfg, learners="plants"), world=evo)


def test_pick_samples_takes_living_slots_first():
    from agents.ppo import pick_samples
    mask = jnp.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0], bool)
    idx = np.asarray(pick_samples(jax.random.PRNGKey(0), mask, 4))
    assert len(set(idx.tolist())) == 4 and {1, 4, 7} <= set(idx.tolist())
    crowded = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0], bool)
    idx = np.asarray(pick_samples(jax.random.PRNGKey(1), crowded, 4))
    assert np.asarray(crowded)[idx].all()


def test_sampled_learning_still_trains():
    from dataclasses import replace
    wcfg, tcfg = tiny_configs()
    trainer = make_trainer(wcfg, replace(tcfg, sample_frac=0.5))
    runner = trainer.init(jax.random.PRNGKey(0))
    before = np.asarray(jax.tree.leaves(runner.prey.params)[0]).copy()
    runner, metrics = trainer.update(runner)
    assert all(math.isfinite(float(v)) for v in jax.device_get(metrics).values())
    assert not np.array_equal(before, np.asarray(jax.tree.leaves(runner.prey.params)[0]))


def test_league_keep_masks_out_frozen_sides():
    from agents.ppo import league_keep
    prey_keep, pred_keep = league_keep(8, 0.25, jnp.array(True))
    # Worlds 0-1: past predators (their actions are not learned from); worlds 2-3: past prey.
    assert np.asarray(pred_keep).tolist() == [False, False] + [True] * 6
    assert np.asarray(prey_keep).tolist() == [True, True, False, False] + [True] * 4
    prey_keep, pred_keep = league_keep(8, 0.25, jnp.array(False))
    assert np.asarray(prey_keep).all() and np.asarray(pred_keep).all()


def test_league_snapshots_and_trains_against_them():
    from dataclasses import replace
    from simulations import evo
    wcfg, tcfg = tiny_configs("evo")
    trainer = make_trainer(wcfg, replace(tcfg, num_envs=8, league_size=2, league_frac=0.25), world=evo)
    runner = trainer.init(jax.random.PRNGKey(0))
    assert int(runner.league.count) == 0
    runner = trainer.add_to_league(runner)
    snap = [np.asarray(x[0]).copy() for x in jax.tree.leaves(runner.league.pred)]
    assert int(runner.league.count) == 1
    np.testing.assert_array_equal(snap[0], np.asarray(jax.tree.leaves(runner.pred.params)[0]))
    runner, metrics = trainer.update(runner)
    metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
    assert {"league_prey_caught", "league_pred_catch", "self_catch"} <= set(metrics)
    assert all(math.isfinite(v) for v in metrics.values())
    # The snapshot is frozen while the live brain learns.
    np.testing.assert_array_equal(snap[0], np.asarray(jax.tree.leaves(runner.league.pred)[0][0]))
    assert not np.array_equal(snap[0], np.asarray(jax.tree.leaves(runner.pred.params)[0]))


def test_trainer_without_league_has_identity_add():
    wcfg, tcfg = tiny_configs()
    trainer = make_trainer(wcfg, tcfg)
    runner = trainer.init(jax.random.PRNGKey(0))
    assert runner.league is None and trainer.add_to_league(runner) is runner
