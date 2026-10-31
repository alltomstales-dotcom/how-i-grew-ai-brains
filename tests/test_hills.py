"""Stage 8a: a heightfield on the grid."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo
from simulations.worlds import get_world, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=24, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, hills=True, **QUIET)
KEY = jax.random.PRNGKey(3)


def test_the_height_map_is_fixed_seamless_and_spans_zero_to_one():
    h = evo.height_map(CFG)
    assert h.shape == (24, 24) and h.min() == 0.0 and h.max() == 1.0
    assert np.array_equal(h, evo.height_map(CFG))
    assert not np.array_equal(h, evo.height_map(replace(CFG, hill_seed=1)))
    inner = np.abs(np.diff(h, axis=1)).max()
    assert np.abs(h[:, 0] - h[:, -1]).max() <= inner + 1e-6  # the wrap is no steeper than anywhere else


def test_sight_lines_pass_through_the_cells_between():
    iy, ix, frac, valid = evo.sight_lines(9)
    r = 4
    east = (r, r + 4)  # dy 0, dx +4
    assert valid[east].tolist() == [True, True, True]
    assert ix[east].tolist() == [5, 6, 7] and iy[east].tolist() == [4, 4, 4]
    assert np.allclose(frac[east], [0.25, 0.5, 0.75])
    assert not valid[r, r + 1].any() and not valid[r, r].any()  # neighbours and self: nothing between


def test_a_ridge_hides_what_is_behind_it():
    heights = jnp.zeros((1, 9, 9))
    assert bool(evo._visible(CFG, heights, jnp.zeros(1))[0, 4, 8])
    ridge = heights.at[0, 4, 6].set(0.5)
    vis = evo._visible(CFG, ridge, jnp.zeros(1))[0]
    assert not bool(vis[4, 8]) and bool(vis[4, 6]) and bool(vis[4, 0])
    # From the top of a higher hill, the same ridge no longer hides it.
    assert bool(evo._visible(CFG, ridge, jnp.ones(1))[0, 4, 8])


def test_climbing_costs_energy():
    h = evo.height_map(CFG)
    rise = np.roll(h, -1, axis=1) - h  # going east from (x, y)
    y, x = np.unravel_index(np.argmax(rise), rise.shape)
    pred = [(int(x), int(y), 50.0, 0.0, 0.5, 1.0)]
    east = jnp.full(4, 3, jnp.int32)  # action 3 = one cell east
    flat = replace(CFG, hills=False)
    e_hill = float(evo.advance(CFG, KEY, evo.make_state(CFG, pred=pred), jnp.zeros(4, jnp.int32), east).state.pred.energy[0])
    e_flat = float(evo.advance(flat, KEY, evo.make_state(flat, pred=pred), jnp.zeros(4, jnp.int32), east).state.pred.energy[0])
    assert np.isclose(e_flat - e_hill, CFG.climb_cost * rise[y, x], atol=1e-4)


def test_the_height_channel_comes_before_the_brain_gene():
    s = evo.make_state(CFG, pred=[(5, 5, 50.0, 0.0, 1.0, 0.5)])
    _, obs = evo.observe(CFG, s)
    assert obs.shape[-1] == obs_channels(evo, CFG) == evo.OBS_CHANNELS + 1
    h = evo.height_map(CFG)
    o = np.asarray(obs[0], np.float32)
    assert o[4, 4, -1] == 0.0  # its own cell is at relative height 0
    assert np.isclose(o[4, 5, -1], h[5, 6] - h[5, 5], atol=1e-2)  # the cell to the east, always visible
    both = replace(CFG, brains=True)
    s = evo.make_state(both, pred=[(5, 5, 50.0, 0.0, 1.0, 0.5, 0.5, 0.7)])
    _, obs = evo.observe(both, s)
    assert obs.shape[-1] == evo.OBS_CHANNELS + 2
    assert np.allclose(np.asarray(obs[0], np.float32)[..., -1], 0.7, atol=1e-2)


def test_hills_worlds_are_registered_and_train():
    from agents.ppo import make_trainer, tiny_configs
    assert get_world("hills")[1]().hills and get_world("flank_hills")[1]().flank
    assert not evo.EvoConfig().hills
    wcfg, tcfg = tiny_configs("hills")
    trainer = make_trainer(wcfg, tcfg, world=evo)
    runner, metrics = trainer.update(trainer.init(jax.random.PRNGKey(0)))
    assert np.isfinite(float(metrics["pred_reward"]))


def test_hills_script_and_episode_on_a_tiny_run(tmp_path):
    import json

    import scripts.episode3d as ep
    import scripts.hills as hills
    import scripts.train as train
    assert train.main(["--tiny", "--world", "hills", "--runs-dir", str(tmp_path), "--run-id", "h1", "--updates", "2"]) == 0
    assert hills.main(["--run-dir", str(tmp_path / "h1"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "h1" / "hills.json").read_text())
    assert out["channel"] == evo.OBS_CHANNELS and isinstance(out["pass"], bool)
    for k in ("trained", "random"):
        assert 0 <= out["high_ground"][k]["prey"] <= 1 and 0 <= out["high_ground"][k]["pred"] <= 1
    assert ep.main(["--run-dir", str(tmp_path / "h1")]) == 0
    assert "height" in json.loads((tmp_path / "h1" / "episode3d.json").read_text())
    assert hills.verdict({"catches_per_pred_step": 1.0, "prey_lifetime": 10.0},
                         {"catches_per_pred_step": 0.85, "prey_lifetime": 10.0},
                         {"catches_per_pred_step": 1.0, "prey_lifetime": 9.5})["pass"]
