"""Stage 8b (ambush: moving costs energy, thickets hide) and Stage 7c (fickle food, learning in one life)."""
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo
from simulations.worlds import get_world, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
AMBUSH = evo.AmbushConfig(size=24, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, **QUIET)
FICKLE = evo.FickleConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(5)


def acts(a=0):
    return jnp.full(4, a, jnp.int32)


def test_old_worlds_are_unchanged():
    c = evo.EvoConfig()
    assert c.move_cost == 0 and c.thickets == 0 and not c.fickle
    assert obs_channels(evo, c) == evo.OBS_CHANNELS


def test_moving_costs_energy_and_standing_still_does_not():
    pred = [(5, 5, 50.0, 0.0, 0.5, 1.0)]
    still = evo.advance(AMBUSH, KEY, evo.make_state(AMBUSH, pred=pred), acts(), acts(0)).state.pred.energy[0]
    moved = evo.advance(AMBUSH, KEY, evo.make_state(AMBUSH, pred=pred), acts(), acts(3)).state.pred.energy[0]
    assert np.isclose(float(still) - float(moved), AMBUSH.move_cost, atol=1e-4)


def test_thickets_are_seamless_patches_of_the_right_share():
    m = evo.thicket_map(replace(AMBUSH, size=48))
    assert abs(m.mean() - 0.25) < 0.02
    assert not np.array_equal(m, evo.thicket_map(replace(AMBUSH, size=48, thicket_seed=1)))


def _cover_cell(cfg, want):
    """A thicket cell, and an open cell two steps to its east (with an open cell for the viewer)."""
    m = evo.thicket_map(cfg)
    for y in range(cfg.size):
        for x in range(cfg.size):
            if m[y, x] == want and not m[y, (x + 3) % cfg.size]:
                return x, y
    raise AssertionError


def test_a_thicket_hides_an_animal_except_from_the_next_cell():
    x, y = _cover_cell(AMBUSH, True)
    near = (x + 1) % 24, y
    far = (x + 3) % 24, y
    hunter = (x, y, 50.0, 0.0, 0.5, 1.0)
    for watcher, seen in ((near, True), (far, False)):
        s = evo.make_state(AMBUSH, prey=[(*watcher, 20.0, 0.0, 1.0, 0.5)], pred=[hunter])
        prey_obs, _ = evo.observe(AMBUSH, s)
        dx = (x - watcher[0] + 12) % 24 - 12
        assert (float(np.asarray(prey_obs[0], np.float32)[4, 4 + dx, 3]) > 0) == seen, watcher
    exposed = replace(AMBUSH, cover_hides_pred=False)
    s = evo.make_state(exposed, prey=[(*far, 20.0, 0.0, 1.0, 0.5)], pred=[hunter])
    prey_obs, _ = evo.observe(exposed, s)
    assert float(np.asarray(prey_obs[0], np.float32)[4, 1, 3]) > 0  # now seen from three cells away


def test_the_thicket_channel_comes_before_the_brain_gene():
    both = replace(AMBUSH, brains=True)
    s = evo.make_state(both, pred=[(5, 5, 50.0, 0.0, 1.0, 0.5, 0.5, 0.7)])
    _, obs = evo.observe(both, s)
    assert obs.shape[-1] == obs_channels(evo, both) == evo.OBS_CHANNELS + 2
    assert np.allclose(np.asarray(obs[0], np.float32)[..., -1], 0.7, atol=1e-2)


def test_the_good_kind_feeds_and_the_other_poisons():
    prey = [(2, 2, 20.0, 0.0, 0.5, 0.5), (8, 8, 20.0, 0.0, 0.5, 0.5)]
    plants = [(2, 2, 0.5, 0.0), (8, 8, 0.5, 1.0)]  # soft at the first, tough at the second
    hungry = evo.make_state(FICKLE, prey=prey)
    base = np.asarray(evo.advance(FICKLE, KEY, hungry, acts(), acts()).state.prey.energy)
    for good, want in ((0, (FICKLE.grass_energy, -FICKLE.poison)), (1, (-FICKLE.poison, FICKLE.grass_energy))):
        s = evo.make_state(FICKLE, prey=prey, plants=plants)._replace(good=jnp.int32(good))
        r = evo.advance(FICKLE, KEY, s, acts(), acts())
        assert np.allclose(np.asarray(r.state.prey.energy)[:2] - base[:2], want, atol=1e-4)
        assert int(r.info.good_bites) == 1 and int(r.info.bad_bites) == 1


def test_the_good_kind_flips_at_random_and_is_not_observed():
    s = evo.reset(FICKLE, KEY)
    flips, key = 0, KEY
    for i in range(200):
        key, k = jax.random.split(key)
        r = evo.advance(replace(FICKLE, flip_rate=0.2), k, s, acts(), acts())
        flips += int(r.state.good != s.good)
        s = s._replace(good=r.state.good, since_flip=r.state.since_flip)
    assert 15 < flips < 70
    a, b = s._replace(good=jnp.int32(0)), s._replace(good=jnp.int32(1))
    for x, y in zip(evo.observe(FICKLE, a), evo.observe(FICKLE, b)):
        assert np.array_equal(np.asarray(x, np.float32), np.asarray(y, np.float32))


def test_both_worlds_train_and_the_scripts_run(tmp_path):
    import scripts.ambush as ambush
    import scripts.episode3d as ep
    import scripts.lifetime as lifetime
    import scripts.train as train
    assert get_world("ambush")[1]().thickets > 0 and get_world("fickle")[1]().fickle
    common = ["--tiny", "--runs-dir", str(tmp_path), "--updates", "2"]
    assert train.main([*common, "--world", "ambush", "--run-id", "a1"]) == 0
    assert ambush.main(["--run-dir", str(tmp_path / "a1"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "a1" / "ambush.json").read_text())
    assert isinstance(out["pass"], bool) and 0 <= out["trained"]["still"] <= 1 and 0 <= out["trained"]["in_cover"] <= 1
    assert ep.main(["--run-dir", str(tmp_path / "a1")]) == 0
    assert "cover" in json.loads((tmp_path / "a1" / "episode3d.json").read_text())
    assert train.main([*common, "--world", "fickle", "--run-id", "f1"]) == 0
    assert train.main([*common, "--world", "fickle", "--run-id", "g1", "--memory", "8"]) == 0
    assert lifetime.main(["--run-dir", str(tmp_path / "g1"), "--baseline", str(tmp_path / "f1"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "g1" / "lifetime.json").read_text())
    assert out["memory"] == 8 and isinstance(out["pass"], bool) and set(out["bites"]) == {"0-10", "10-40", "40+"}
    assert lifetime.main(["--run-dir", str(tmp_path / "g1"), "--baseline", str(tmp_path / "f1"), "--episodes", "2", "--controls"]) == 0
    out = json.loads((tmp_path / "g1" / "lifetime_controls.json").read_text())
    assert set(out) == {"memory", "wiped", "feed_forward"} and 0 <= out["wiped"]["next"]["after_bad"] <= 1


def test_verdicts():
    import scripts.ambush as ambush
    import scripts.lifetime as lifetime
    assert ambush.verdict({"in_cover": 0.4, "still": 0.4}, {"still": 0.2}, 0.25, 1.0, 0.8)["pass"]
    assert not ambush.verdict({"in_cover": 0.4, "still": 0.4}, {"still": 0.2}, 0.25, 1.0, 0.95)["pass"]
    good = {"0-10": {"share": 0.5}, "10-40": {"share": 0.7}}
    assert lifetime.verdict(good, {"10-40": {"share": 0.55}})["pass"]
    assert not lifetime.verdict(good, {"10-40": {"share": 0.65}})["pass"]
