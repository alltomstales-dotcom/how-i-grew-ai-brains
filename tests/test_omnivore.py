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


def test_winter_thins_plants_for_grazers_and_hunters():
    w = replace(CFG, winter_len=10, winter_value=0.25)
    prey = [(3, 3, 20.0, 0.0, 0.5, 0.5)]
    plant = [(3, 3, 0.5, 0.0)]
    for t, value in ((0, 1.0), (9, 1.0), (10, 0.25), (19, 0.25), (20, 1.0)):
        hunter = evo.make_state(w, pred=[(3, 3, 30.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0)], plants=plant, t=t)
        assert np.isclose(float(evo.advance(w, KEY, hunter, STILL, STILL).info.pred_plant_gain), value * w.grass_energy)
        grazer = evo.make_state(w, prey=prey, plants=plant, t=t)
        assert np.isclose(float(evo.advance(w, KEY, grazer, STILL, STILL).state.prey.energy[0]), 20.0 + value * w.grass_energy - 1.0, atol=1.5)


def test_winter_verdict_needs_one_mixed_diet_and_a_control_that_is_not():
    import scripts.omnivore as o
    mixed, plain = {"diet": 0.5, "plant_share": 0.5, "diet_sd": 0.1}, {"diet": 0.9}
    assert o.winter_verdict(mixed, plain)["pass"]
    assert not o.winter_verdict({**mixed, "diet_sd": 0.3}, plain)["pass"]  # two camps
    assert not o.winter_verdict({**mixed, "plant_share": 0.95}, plain)["pass"]
    assert not o.winter_verdict(mixed, {"diet": 0.5})["pass"]  # mixed without winter too


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
    assert train.main([*common, "--world", "omni_winter", "--run-id", "w"]) == 0
    assert omnivore.main(["--winter", str(tmp_path / "w"), "--control", str(tmp_path / "d"), "--episodes", "2"]) == 0
    assert isinstance(json.loads((tmp_path / "w" / "winter.json").read_text())["pass"], bool)
    assert omnivore.main(["--camps", str(tmp_path / "a"), "--episodes", "2"]) == 0
    assert isinstance(json.loads((tmp_path / "a" / "camps.json").read_text())["pass"], bool)
    assert omnivore.main(["--camp-curve", str(tmp_path / "a"), "--episodes", "2"]) == 0
    assert json.loads((tmp_path / "a" / "camp_curve.json").read_text())[0]["step"] == 0


def test_the_lean_world_starts_with_fewer_plants():
    rich, lean = evo.OmniD11Config(), evo.OmniD11LeanConfig()
    assert rich.plant_digest == lean.plant_digest == 0.11 and lean.init_grass_frac == 0.2 < rich.init_grass_frac
    assert float(evo.reset(lean, KEY).plant.mean()) < float(evo.reset(rich, KEY).plant.mean())


def test_diet_split_is_scored_on_hunters_only():
    import scripts.species as species
    rng = np.random.default_rng(0)
    both = np.concatenate([rng.normal(0.05, 0.05, 300), rng.normal(0.9, 0.05, 300)])
    genes = np.full((600, evo.ANIMAL_GENES), 0.5); genes[:, 5] = both
    assert species.split_found(species.split(genes, "diet"), "diet")
    drifted = {"separation": 2.6, "minority": 0.3, "size_gap": 0.48}
    assert species.split_found(drifted) and not species.split_found(drifted, "diet")
    one = np.full((600, evo.ANIMAL_GENES), 0.5); one[:, 5] = rng.normal(0.05, 0.05, 600)
    assert not species.split_found(species.split(one, "diet"), "diet")
