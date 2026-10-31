"""Stage 9a: hunters with a diet gene."""
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo
from simulations.worlds import obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0, pred_base_cost=0.0, speed_cost=0.0, sight_cost=0.0)
CFG = evo.OmniConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(7)
STILL = jnp.zeros(4, jnp.int32)


def pred_after(cfg, diet, prey=(), plants=()):
    pred = [(3, 3, 30.0, 0.0, 0.5, 0.5, 0.5, 0.5, diet)]
    s = evo.make_state(cfg, prey=list(prey), pred=pred, plants=list(plants))
    return evo.advance(cfg, KEY, s, STILL, STILL)


def test_diet_splits_a_catch_and_a_plant():
    prey = [(3, 3, 20.0, 0.0, 0.5, 0.5)]
    plant = [(3, 3, 0.5, 0.0)]
    meat = pred_after(CFG, 0.8, prey=prey); greens = pred_after(CFG, 0.8, plants=plant)
    assert np.isclose(float(meat.info.pred_meat_gain), 0.8 * CFG.catch_energy)
    assert np.isclose(float(greens.info.pred_plant_gain), 0.2 * CFG.grass_energy) and int(greens.info.pred_bites) == 1
    assert np.isclose(float(greens.state.pred.energy[0]), 30.0 + 0.2 * CFG.grass_energy)


def test_a_poor_gut_gets_less_and_the_control_ignores_the_gene():
    plant = [(3, 3, 0.5, 0.0)]
    assert np.isclose(float(pred_after(replace(CFG, plant_digest=evo.OmniToughConfig.plant_digest), 0.0, plants=plant).info.pred_plant_gain), 0.3 * CFG.grass_energy)
    neutral = replace(CFG, diet_fixed=evo.OmniNeutralConfig.diet_fixed)
    assert np.isclose(float(pred_after(neutral, 0.9, plants=plant).info.pred_plant_gain), 0.5 * CFG.grass_energy)
    assert np.isclose(float(pred_after(neutral, 0.1, prey=[(3, 3, 20.0, 0.0, 0.5, 0.5)]).info.pred_meat_gain), 0.5 * CFG.catch_energy)


def test_grazers_eat_first_and_hunters_elsewhere_are_unchanged():
    both = pred_after(CFG, 0.0, prey=[(3, 3, 20.0, 0.0, 0.5, 0.5)], plants=[(3, 3, 0.5, 0.0)])
    assert int(both.info.pred_bites) == 0 or int(both.info.catches) == 1
    plain = evo.EvoConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, **QUIET)
    r = pred_after(plain, 0.0, prey=[(3, 3, 20.0, 0.0, 0.5, 0.5)], plants=[(4, 4, 0.5, 0.0)])
    assert np.isclose(float(r.info.pred_meat_gain), plain.catch_energy) and int(r.info.pred_bites) == 0


def test_hunters_see_their_own_diet():
    s = evo.make_state(CFG, pred=[(3, 3, 30.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.25)])
    _, pred_obs = evo.observe(CFG, s)
    assert pred_obs.shape[-1] == obs_channels(evo, CFG) == evo.OBS_CHANNELS + 1
    assert np.isclose(float(pred_obs[0, 0, 0, -1]), 0.25)


def test_the_script_runs_on_tiny_runs(tmp_path):
    import scripts.omnivore as omnivore
    import scripts.train as train
    common = ["--tiny", "--runs-dir", str(tmp_path), "--updates", "2"]
    for name, world in (("a", "omni"), ("b", "omni_tough"), ("c", "omni_neutral")):
        assert train.main([*common, "--world", world, "--run-id", name]) == 0
    assert omnivore.main(["--omni", str(tmp_path / "a"), "--tough", str(tmp_path / "b"), "--neutral", str(tmp_path / "c"), "--episodes", "2"]) == 0
    assert isinstance(json.loads((tmp_path / "a" / "omnivore.json").read_text())["pass"], bool)
    assert train.main([*common, "--world", "omni_d03", "--run-id", "d"]) == 0
    assert omnivore.main(["--ladder", str(tmp_path / "d"), str(tmp_path / "a"), "--episodes", "2"]) == 0
    assert isinstance(json.loads((tmp_path / "d" / "ladder.json").read_text())["pass"], bool)
