"""Stage 7a: brain size as a gene that costs energy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulations import evo
from simulations.worlds import get_world, make_net, obs_channels

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, max_steps=40, brains=True, **QUIET)
KEY = jax.random.PRNGKey(2)


def still(n):
    return jnp.zeros(n, jnp.int32)


def test_every_animal_has_a_brain_gene():
    assert evo.ANIMAL_GENES == 7 and evo.GENE_NAMES["animal"][4] == "brain"
    assert evo.EvoConfig().founder_animal[4] == 0.5
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.5, 0.5, 0.5)])  # written without a brain gene
    assert float(s.prey.genes[0, 4]) == 0.5
    assert evo.pad_genes(np.zeros((2, 4)), 5)[:, 4].tolist() == [0.5, 0.5]


def test_the_brain_gene_is_the_last_channel():
    s = evo.make_state(CFG, pred=[(5, 5, 50.0, 0.0, 0.5, 1.0, 0.5, 0.8)])
    _, pred_obs = evo.observe(CFG, s)
    assert pred_obs.shape[-1] == obs_channels(evo, CFG) == evo.OBS_CHANNELS + 1
    assert np.allclose(np.asarray(pred_obs[0], np.float32)[..., -1], 0.8, atol=1e-2)
    both = evo.EvoConfig(**{**CFG.__dict__, "signals": True})
    _, pred_obs = evo.observe(both, s)
    assert pred_obs.shape[-1] == obs_channels(evo, both) == evo.OBS_CHANNELS + 2
    assert np.allclose(np.asarray(pred_obs[0], np.float32)[..., -1], 0.8, atol=1e-2)
    off = evo.EvoConfig(**{**CFG.__dict__, "brains": False})
    assert evo.observe(off, s)[1].shape[-1] == evo.OBS_CHANNELS


def test_brains_cost_energy():
    s = evo.make_state(CFG, pred=[(2, 2, 50.0, 0.0, 0.5, 1.0, 0.5, 0.0), (8, 8, 50.0, 0.0, 0.5, 1.0, 0.5, 1.0)])
    e = np.asarray(evo.advance(CFG, KEY, s, still(4), still(4)).state.pred.energy)
    assert np.isclose(e[0] - e[1], CFG.brain_cost, atol=1e-4)
    off = evo.EvoConfig(**{**CFG.__dict__, "brains": False})
    e = np.asarray(evo.advance(off, KEY, s, still(4), still(4)).state.pred.energy)
    assert np.isclose(e[0], e[1])


def test_a_small_brain_uses_only_the_first_hidden_units():
    net = make_net(evo, CFG)
    assert net.brain
    obs = jnp.zeros((2, 9, 9, obs_channels(evo, CFG))).at[0, ..., -1].set(0.0).at[1, ..., -1].set(1.0)
    obs = obs.at[:, 4, 4, 0].set(1.0)
    params = net.init(jax.random.PRNGKey(0), obs)
    logits, _ = net.apply(params, obs)
    # Change only what hidden units 8.. pass on: the small brain (gene 0) must not notice.
    p = jax.tree.map(lambda x: x, params)
    names = [k for k in p["params"] if k.startswith("Dense")]
    head = sorted(names)[1]  # the policy head reads the 128 hidden units
    k = p["params"][head]["kernel"]
    p["params"][head]["kernel"] = k.at[8:].add(1.0)
    logits2, _ = net.apply(p, obs)
    assert np.allclose(logits[0], logits2[0]) and not np.allclose(logits[1], logits2[1])


def test_brain_worlds_are_registered_and_train():
    from agents.ppo import TrainConfig, make_trainer, tiny_configs
    for name, flank in (("pack_brains", False), ("flank_brains", True)):
        c = get_world(name)[1]()
        assert c.brains and c.pack_size == 2 and c.flank == flank
    wcfg, tcfg = tiny_configs("flank_brains")
    trainer = make_trainer(wcfg, tcfg, world=evo)
    runner = trainer.init(jax.random.PRNGKey(0))
    runner, metrics = trainer.update(runner)
    assert "pred_brain" in metrics and np.isfinite(float(metrics["pred_reward"]))
    from dataclasses import replace
    with pytest.raises(ValueError):
        make_trainer(wcfg, replace(tcfg, memory=8), world=evo)


def test_runs_saved_with_four_genes_still_load():
    from scripts.eval import load_world_config
    world, cfg = load_world_config({"world_name": "flank_signals",
                                    "world": {"founder_animal": [0.2, 0.5, 0.5, 0.5], "size": 12}})
    assert cfg.founder_animal == (0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    s = world.reset(cfg, KEY)
    assert s.prey.genes.shape[1] == evo.ANIMAL_GENES


def test_brain_worlds_found_with_every_brain_size():
    """Founders spread the brain gene over [0, 1], so every width of the brain is trained before
    selection starts (with all founders near 0.5, units above about 76 would never learn)."""
    big = evo.EvoConfig(**{**CFG.__dict__, "max_prey": 400, "max_pred": 400, "init_prey": 400, "init_pred": 400,
                           "founder_noise": 0.05})
    for name in ("pack_brains", "flank_brains"):
        assert get_world(name)[1]().brain_spread
    spread = evo.EvoConfig(**{**big.__dict__, "brain_spread": True})
    s = evo.reset(spread, KEY)
    g = np.asarray(s.pred.genes)
    assert g[:, 4].min() < 0.1 and g[:, 4].max() > 0.9 and abs(g[:, 4].mean() - 0.5) < 0.06
    assert abs(g[:, 0].mean() - 0.2) < 0.02  # other genes stay at their founder values
    assert np.asarray(evo.reset(big, KEY).pred.genes)[:, 4].std() < 0.1  # off: founders near 0.5


def test_runs_saved_before_a_flag_existed_keep_the_old_behaviour():
    """A field missing from a saved config takes the base world's default (the flag's 'off'), not
    the registered world's newer default, so old runs evaluate as they were trained."""
    from scripts.eval import load_world_config
    world = {"founder_animal": [0.2, 0.5, 0.5, 0.5, 0.5], "size": 12, "brains": True, "pack_size": 2, "flank": True}
    assert not load_world_config({"world_name": "flank_brains", "world": world})[1].brain_spread
    assert load_world_config({"world_name": "flank_brains", "world": {**world, "brain_spread": True}})[1].brain_spread


def test_a_brain_tax_takes_a_share_of_each_meal():
    """Stage 7b: with brain_tax, a brain costs a share of what an animal eats, not energy per step."""
    taxed = evo.EvoConfig(**{**CFG.__dict__, "brain_cost": 0.0, "brain_tax": 0.4})
    # Two prey, one with brain 0 and one with brain 1; each meal is compared with no meal.
    prey = [(2, 2, 10.0, 0.0, 0.5, 0.5, 0.5, 0.0), (8, 8, 10.0, 0.0, 0.5, 0.5, 0.5, 1.0)]
    fed = evo.make_state(taxed, prey=prey, plants=[(2, 2, 0.5, 0.0), (8, 8, 0.5, 0.0)])
    hungry = evo.make_state(taxed, prey=prey)
    energy = lambda st: np.asarray(evo.advance(taxed, KEY, st, still(4), still(4)).state.prey.energy)
    gain = energy(fed) - energy(hungry)
    assert gain[0] > 0 and np.isclose(gain[1], 0.6 * gain[0], atol=1e-4)
    # Without food, the brain costs nothing.
    s = evo.make_state(taxed, pred=[(2, 2, 50.0, 0.0, 0.5, 1.0, 0.5, 0.0), (8, 8, 50.0, 0.0, 0.5, 1.0, 0.5, 1.0)])
    e = np.asarray(evo.advance(taxed, KEY, s, still(4), still(4)).state.pred.energy)
    assert np.isclose(e[0], e[1])


def test_tax_worlds_are_registered():
    for name, flank in (("pack_tax", False), ("flank_tax", True)):
        c = get_world(name)[1]()
        assert c.brains and c.brain_spread and c.brain_cost == 0.0 and c.brain_tax == 0.3 and c.flank == flank
    assert evo.EvoConfig().brain_tax == 0.0
