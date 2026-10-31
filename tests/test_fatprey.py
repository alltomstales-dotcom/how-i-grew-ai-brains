"""Stage 5b: meals worth what the prey stored, and Stage 7d's scatter."""
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo
from simulations.worlds import get_world, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.FatPreyConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(7)


def still():
    return jnp.zeros(4, jnp.int32)


def test_old_worlds_keep_fixed_meals():
    c = evo.EvoConfig()
    assert not c.stored_meals and not c.fat_sense and obs_channels(evo, c) == evo.OBS_CHANNELS


def test_a_meal_is_worth_what_the_prey_stored():
    pred = [(3, 3, 20.0, 0.0, 0.5, 1.0)]
    upkeep = None
    for energy in (10.0, 30.0):
        s = evo.make_state(CFG, prey=[(3, 3, energy, 0.0, 0.5, 0.5)], pred=pred)
        r = evo.advance(CFG, KEY, s, still(), still())
        assert int(r.info.catches) == 1 and np.isclose(float(r.info.eaten_energy), energy)
        hungry = evo.advance(CFG, KEY, evo.make_state(CFG, pred=pred), still(), still()).state.pred.energy[0]
        assert np.isclose(float(r.state.pred.energy[0]) - float(hungry), CFG.meal_share * energy, atol=1e-3)
    fixed = replace(CFG, stored_meals=False)
    s = evo.make_state(fixed, prey=[(3, 3, 10.0, 0.0, 0.5, 0.5)], pred=pred)
    hungry = evo.advance(fixed, KEY, evo.make_state(fixed, pred=pred), still(), still()).state.pred.energy[0]
    gain = float(evo.advance(fixed, KEY, s, still(), still()).state.pred.energy[0]) - float(hungry)
    assert np.isclose(gain, fixed.catch_energy, atol=1e-3)


def test_hunters_can_see_who_is_fat():
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.5, 0.5)], pred=[(4, 5, 30.0, 0.0, 1.0, 0.5)])
    _, obs = evo.observe(CFG, s)
    assert obs.shape[-1] == obs_channels(evo, CFG) == evo.OBS_CHANNELS + 1
    assert np.isclose(np.asarray(obs[0], np.float32)[4, 5, -1], 20.0 / CFG.prey_energy_cap, atol=1e-2)


def test_the_scripts_run_on_tiny_runs(tmp_path):
    import scripts.lifetime as lifetime
    import scripts.meals as meals
    import scripts.scatter as scatter
    import scripts.train as train
    assert get_world("fatprey")[1]().stored_meals and not get_world("fatprey_fixed")[1]().stored_meals
    common = ["--tiny", "--runs-dir", str(tmp_path), "--updates", "2"]
    assert train.main([*common, "--world", "fatprey", "--run-id", "a"]) == 0
    assert train.main([*common, "--world", "fatprey_fixed", "--run-id", "b"]) == 0
    assert meals.main(["--run-dir", str(tmp_path / "a"), "--control", str(tmp_path / "b"), "--episodes", "2"]) == 0
    assert isinstance(json.loads((tmp_path / "a" / "meals.json").read_text())["pass"], bool)
    assert train.main([*common, "--world", "fickle", "--run-id", "f"]) == 0
    assert scatter.main(["--run-dir", str(tmp_path / "f"), "--episodes", "2"]) == 0
    assert scatter.main(["--run-dir", str(tmp_path / "f"), "--episodes", "2", "--worlds"]) == 0
    assert lifetime.main(["--run-dir", str(tmp_path / "f"), "--episodes", "2", "--why"]) == 0


def test_the_meals_verdict():
    import scripts.meals as meals
    good = {"selectivity": 1.3, "energy_per_catch": 30.0}
    assert meals.verdict(good, {"selectivity": 1.0}, {"energy_per_catch": 25.0})["pass"]
    assert not meals.verdict(good, {"selectivity": 1.2}, {"energy_per_catch": 25.0})["pass"]
