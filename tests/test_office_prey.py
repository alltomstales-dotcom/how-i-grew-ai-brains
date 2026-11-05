"""Stage 22b: grazers that can't starve (prey_no_upkeep). Spec docs/specs/2026-11-05-office-prey-design.md"""
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from scripts import office as office_script
from simulations import evo
from simulations.worlds import get_world

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=16, max_prey=6, max_pred=4, init_prey=2, init_pred=1, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(5)


def _still(n):
    return jnp.zeros(n, jnp.int32)


def test_off_by_default_and_upkeep_paid_as_before():
    assert evo.EvoConfig().prey_no_upkeep is False
    assert get_world("office")[1]().prey_no_upkeep is False and get_world("office_home")[1]().prey_no_upkeep is False
    s = evo.make_state(CFG, prey=[(3, 3, 20.0, 0.2, 0.5, 0.5)], pred=[(12, 12, 50.0, 0.2, 0.5, 0.5)])
    r = evo.advance(CFG, KEY, s, _still(6), _still(4))
    upkeep = float(evo._upkeep(CFG, s.prey.genes[0]))
    assert upkeep > 0
    assert np.isclose(float(r.state.prey.energy[0]), 20.0 - upkeep, atol=1e-5)


def test_no_upkeep_stops_starvation():
    low = [(3, 3, 0.5, 0.2, 0.5, 0.5)]  # less energy than one step's upkeep, and no plants to eat
    s = evo.make_state(CFG, prey=low, pred=[(12, 12, 50.0, 0.2, 0.5, 0.5)])
    off = evo.advance(CFG, KEY, s, _still(6), _still(4))
    assert not bool(off.state.prey.alive[0]) and int(off.info.prey_deaths) == 1
    on_cfg = dataclasses.replace(CFG, prey_no_upkeep=True)
    on = evo.advance(on_cfg, KEY, s, _still(6), _still(4))
    assert bool(on.state.prey.alive[0]) and int(on.info.prey_deaths) == 0
    assert np.isclose(float(on.state.prey.energy[0]), 0.5)
    # hunters still pay as before
    assert np.isclose(float(on.state.pred.energy[0]), float(off.state.pred.energy[0]))


def test_in_the_office_every_grazer_death_is_a_catch():
    cfg = dataclasses.replace(get_world("office")[1](), max_steps=120)
    on = office_script.layout_config("test", ["prey_no_upkeep=true", "max_steps=120"])
    assert on.prey_no_upkeep and on.office_seed == cfg.office_seed + 1000 and on.max_steps == 120

    def roll(c):
        def body(state, k):
            kp, kq, kw = jax.random.split(k, 3)
            n_prey, n_pred = state.prey.alive.shape[0], state.pred.alive.shape[0]
            pa = jax.random.randint(kp, (n_prey,), 0, evo.NUM_ACTIONS)
            qa = jax.random.randint(kq, (n_pred,), 0, evo.NUM_ACTIONS)
            res = evo.advance(c, kw, state, pa, qa)
            return res.state, (res.info.catches, res.info.prey_deaths)
        state = evo.reset(c, jax.random.PRNGKey(1))
        _, (catches, deaths) = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(2), c.max_steps))
        return int(catches.sum()), int(deaths.sum())

    c_on, d_on = roll(on)
    c_off, d_off = roll(dataclasses.replace(on, prey_no_upkeep=False))
    assert d_on == c_on
    assert d_off > c_off  # without the field, some grazers starve in the same office
