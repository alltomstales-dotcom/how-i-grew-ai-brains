"""Stage 6b: a signal action that animals of the same species can see."""
import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo
from simulations.worlds import get_world, num_actions, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, signals=True, **QUIET)
KEY = jax.random.PRNGKey(3)
STILL, EAST = 0, 3
SIG = 5  # actions 5-9 are moves 0-4 with the signal on


def acts(*a, n=4):
    return jnp.array(list(a) + [STILL] * (n - len(a)), jnp.int32)


def test_shapes_follow_the_world():
    assert obs_channels(evo, CFG) == evo.OBS_CHANNELS + 1 and num_actions(evo, CFG) == 10
    off = evo.EvoConfig()
    assert obs_channels(evo, off) == evo.OBS_CHANNELS and num_actions(evo, off) == 5
    world, cls = get_world("signals")
    c = cls()
    assert c.signals and c.pack_size == 2 and num_actions(world, c) == 10
    from simulations import predprey
    assert obs_channels(predprey, predprey.WorldConfig()) == predprey.OBS_CHANNELS


def test_signal_action_moves_like_its_base_move_and_is_remembered():
    s = evo.make_state(CFG, pred=[(2, 2, 50.0, 0.0, 1.0, 0.5), (8, 8, 50.0, 0.0, 1.0, 0.5)])
    r = evo.advance(CFG, KEY, s, acts(), acts(SIG + EAST, EAST))
    pos = np.asarray(r.state.pred.pos)
    assert pos[0].tolist() == [3, 2] and pos[1].tolist() == [9, 8]
    assert np.asarray(r.state.pred_signal).tolist() == [True, False, False, False]
    assert not np.asarray(r.state.prey_signal).any()


def test_signals_cost_energy():
    s = evo.make_state(CFG, pred=[(2, 2, 50.0, 0.0, 1.0, 1.0), (8, 8, 50.0, 0.0, 1.0, 1.0)])  # too hungry to breed
    e = np.asarray(evo.advance(CFG, KEY, s, acts(), acts(SIG, STILL)).state.pred.energy)
    assert np.isclose(e[1] - e[0], CFG.signal_cost, atol=1e-4)


def test_own_species_sees_last_steps_signals_in_the_last_channel():
    s = evo.make_state(CFG, prey=[(4, 4, 20.0, 0.0, 1.0, 0.5)],
                       pred=[(2, 2, 50.0, 0.0, 1.0, 0.5), (3, 2, 50.0, 0.0, 1.0, 0.5)])
    s = evo.advance(CFG, KEY, s, acts(), acts(SIG, STILL)).state
    prey_obs, pred_obs = evo.observe(CFG, s)
    assert pred_obs.shape[-1] == evo.OBS_CHANNELS + 1
    r = CFG.view // 2
    o = np.asarray(pred_obs[1], np.float32)[..., -1]  # predator 1 at (3, 2) sees predator 0 at (2, 2)
    assert o[r, r - 1] == 1.0 and o.sum() == 1.0
    assert np.asarray(prey_obs, np.float32)[..., -1].sum() == 0.0  # prey do not hear predators


def test_dead_and_reborn_slots_are_silent():
    c = evo.EvoConfig(**{**CFG.__dict__, "pred_base_cost": 0.0})
    # Predator 0 starves while signalling; predator 1 breeds into the freed slot the same step.
    s = evo.make_state(c, pred=[(2, 2, 0.5, 0.0, 1.0, 0.5), (8, 8, 59.0, 0.0, 1.0, 0.0)])
    r = evo.advance(c, KEY, s, acts(), acts(SIG, SIG))
    alive = np.asarray(r.state.pred.alive)
    sig = np.asarray(r.state.pred_signal)
    assert alive[0] and alive[1]  # slot 0 holds the newborn
    assert sig.tolist() == [False, True, False, False]


def test_signals_off_leave_the_world_unchanged():
    c = evo.EvoConfig(**{**CFG.__dict__, "signals": False})
    s = evo.make_state(c, pred=[(2, 2, 50.0, 0.0, 1.0, 0.5)])
    prey_obs, pred_obs = evo.observe(c, evo.advance(c, KEY, s, acts(), acts(EAST)).state)
    assert pred_obs.shape[-1] == evo.OBS_CHANNELS


def test_brains_for_signal_worlds_have_ten_actions():
    from agents.ppo import make_trainer, tiny_configs
    wcfg, tcfg = tiny_configs("signals")
    trainer = make_trainer(wcfg, tcfg, world=get_world("signals")[0])
    runner = trainer.init(jax.random.PRNGKey(0))
    leaves = jax.tree.leaves(runner.prey.params)
    assert any(x.shape[-1] == 10 for x in leaves)
    assert any(x.ndim == 4 and x.shape[-2] == evo.OBS_CHANNELS + 1 for x in leaves)
    runner, metrics = trainer.update(runner)
    assert np.isfinite(float(metrics["pred_reward"]))


def test_mutual_information_of_a_two_by_two_table():
    from scripts.signals import mutual_information
    assert abs(mutual_information(np.array([[25, 25], [25, 25]]))) < 1e-9  # independent
    assert abs(mutual_information(np.array([[50, 0], [0, 50]])) - 1.0) < 1e-9  # one fair bit
    assert mutual_information(np.zeros((2, 2))) == 0.0


def test_muted_policy_never_sees_the_signal_channel():
    from scripts.signals import muted
    heard = lambda key, obs: (obs[..., -1].sum((-2, -1)) > 0).astype(jnp.int32)
    obs = jnp.ones((3, 9, 9, evo.OBS_CHANNELS + 1))
    key = jax.random.PRNGKey(0)
    assert np.asarray(heard(key, obs)).tolist() == [1, 1, 1]
    quiet = muted(heard)
    assert np.asarray(quiet(key, obs)).tolist() == [0, 0, 0]
    assert quiet.memory == 0


def test_signal_stats_count_signals_by_what_is_near():
    from scripts.signals import signal_stats
    c = evo.EvoConfig(**{**CFG.__dict__, "max_steps": 3})
    always = lambda key, obs: jnp.full(obs.shape[:-3], SIG, jnp.int32)
    never = lambda key, obs: jnp.zeros(obs.shape[:-3], jnp.int32)
    st = signal_stats(evo, c, never, always, jax.random.PRNGKey(0), 2)
    assert st["pred"]["rate"] == 1.0 and st["prey"]["rate"] == 0.0
    assert st["pred"]["table"].sum() > 0 and st["pred"]["mi_bits"] == 0.0


def test_signals_script_on_tiny_run(tmp_path):
    import json
    import scripts.signals as signals
    import scripts.train as train
    assert train.main(["--tiny", "--world", "signals", "--runs-dir", str(tmp_path), "--run-id", "s1", "--updates", "2"]) == 0
    assert signals.main(["--run-dir", str(tmp_path / "s1"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "s1" / "signals.json").read_text())
    for who in ("prey", "pred"):
        assert 0 <= out[who]["rate"] <= 1 and out[who]["mi_bits"] >= 0
    assert out["catches"] >= 0 and out["catches_muted"] >= 0 and "drop" in out and "pass" in out


def test_signals_are_heard_across_the_whole_view_not_just_within_sight():
    # The listener has sight 0 (radius 1) but hears a signaller 3 cells away (review finding).
    s = evo.make_state(CFG, pred=[(2, 2, 50.0, 0.0, 1.0, 1.0), (5, 2, 50.0, 0.0, 0.0, 1.0)])
    s = evo.advance(CFG, KEY, s, acts(), acts(SIG, STILL)).state
    _, pred_obs = evo.observe(CFG, s)
    r = CFG.view // 2
    o = np.asarray(pred_obs[1], np.float32)
    assert o[r, r - 3, -1] == 1.0  # heard
    assert o[r, r - 3, 3] == 0.0  # but not seen


def test_signals_script_with_evolved_bodies(tmp_path):
    import json
    import scripts.signals as signals
    import scripts.train as train
    assert train.main(["--tiny", "--world", "signals", "--runs-dir", str(tmp_path), "--run-id", "s2", "--updates", "2",
                       "--ckpt-every", "1"]) == 0
    assert signals.main(["--run-dir", str(tmp_path / "s2"), "--episodes", "2", "--evolved"]) == 0
    out = json.loads((tmp_path / "s2" / "signals_evolved.json").read_text())
    assert out["bodies"].startswith("genes_") and "drop" in out


def test_shuffled_policy_hears_signals_from_another_world():
    from scripts.signals import shuffled
    heard = lambda key, obs: obs[..., 4, 4, -1].astype(jnp.int32)  # what each animal hears at its own cell
    obs = jnp.zeros((3, 2, 9, 9, evo.OBS_CHANNELS + 1)).at[0, :, 4, 4, -1].set(1.0)  # only world 0 signals
    key = jax.random.PRNGKey(0)
    assert np.asarray(heard(key, obs)).tolist() == [[1, 1], [0, 0], [0, 0]]
    out = np.asarray(shuffled(heard)(key, obs))
    assert out.sum() == 2 and out[0].sum() == 0  # same signals overall, heard in another world
    assert shuffled(heard).memory == 0


def test_signals_script_reports_the_shuffled_control(tmp_path):
    import json
    import scripts.signals as signals
    import scripts.train as train
    assert train.main(["--tiny", "--world", "signals", "--runs-dir", str(tmp_path), "--run-id", "s3", "--updates", "2"]) == 0
    assert signals.main(["--run-dir", str(tmp_path / "s3"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "s3" / "signals.json").read_text())
    assert "catches_shuffled" in out and "drop_shuffled" in out


def test_muted_and_shuffled_touch_only_the_signal_channel_when_brains_are_on():
    from scripts.signals import muted, shuffled
    sig, brain = evo.OBS_CHANNELS, evo.OBS_CHANNELS + 1  # signal channel, then the brain gene last
    seen = lambda key, obs: obs[..., 4, 4, :]
    obs = jnp.zeros((3, 9, 9, evo.OBS_CHANNELS + 2)).at[0, :, :, sig].set(1.0).at[..., brain].set(0.7)
    key = jax.random.PRNGKey(0)
    quiet = np.asarray(muted(seen)(key, obs))
    assert np.allclose(quiet[:, sig], 0) and np.allclose(quiet[:, brain], 0.7)
    moved = np.asarray(shuffled(seen)(key, obs))
    assert moved[:, sig].tolist() == [0, 1, 0] and np.allclose(moved[:, brain], 0.7)
