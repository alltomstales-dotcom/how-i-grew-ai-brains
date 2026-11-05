"""Stage 22: the office generator. Spec: docs/specs/2026-11-05-office-design.md"""
import dataclasses

import jax
import numpy as np
import pytest

from simulations import evo, office
from simulations.worlds import get_world

TRAIN = list(range(office.TRAIN_SEED, office.TRAIN_SEED + 16))
TEST = list(range(office.TEST_SEED, office.TEST_SEED + 16))


@pytest.mark.parametrize("seed", TRAIN + TEST)
def test_every_open_cell_is_reachable(seed):
    wall = office.office_map(48, seed)
    open_ = ~wall
    assert open_.any()
    assert office.reachable(open_)[open_].all()


@pytest.mark.parametrize("seed", TRAIN + TEST)
def test_wall_density(seed):
    assert 0.15 <= office.office_map(48, seed).mean() <= 0.35


def test_determinism_by_seed():
    office._layout.cache_clear()
    a = office.office_map(48, 7).copy()
    office._layout.cache_clear()
    assert np.array_equal(a, office.office_map(48, 7))
    assert not np.array_equal(office.office_map(48, 7), office.office_map(48, 8))


def test_train_and_test_banks_share_no_layout():
    train = {office.office_map(48, s).tobytes() for s in TRAIN}
    test = {office.office_map(48, s).tobytes() for s in TEST}
    assert len(train) == len(TRAIN) and len(test) == len(TEST)
    assert not train & test


@pytest.mark.parametrize("seed", [0, 5, 1000, 1011])
def test_has_corridors_rooms_and_doorways(seed):
    rng = np.random.default_rng(office.SEED_BASE + seed)
    lx, ly = office._axis(48, rng), office._axis(48, rng)
    wall = office.office_map(48, seed)
    assert (lx == office.CORRIDOR).sum() >= 4 and (ly == office.CORRIDOR).sum() >= 4
    assert (lx == office.ROOM).any() and (ly == office.ROOM).any()
    # A doorway: a gap in a wall line, where the cell would be a wall without its door.
    X, Y = np.meshgrid(lx, ly)
    plan = ((X == office.WALL_LINE) | (Y == office.WALL_LINE)) & (X != office.CORRIDOR) & (Y != office.CORRIDOR)
    doors = plan & ~wall
    assert doors.sum() >= 9  # at least one per room block on a 3 x 3 grid of bays
    assert not (wall & ~plan).any()  # walls only ever lie on wall lines


def test_office_off_leaves_reset_unchanged():
    cfg = evo.LastingStealthConfig()
    key = jax.random.PRNGKey(3)
    a = evo.reset(cfg, key)
    b = evo.reset(dataclasses.replace(cfg, office=0, office_seed=1000), key)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert not np.asarray(a.terrain).any()


def test_office_reset_uses_a_bank_layout_and_keeps_walls_clear():
    _, cls = get_world("office")
    cfg = cls()
    assert cfg.gene_sigma == 0 and cfg.founder_noise == 0 and cfg.plant_sigma == 0
    bank = office.office_bank(cfg.size, cfg.office_seed, cfg.office)
    seen = set()
    for k in range(12):
        s = evo.reset(cfg, jax.random.PRNGKey(k))
        wall = np.asarray(s.terrain) == evo.WALL
        hits = [i for i in range(cfg.office) if np.array_equal(bank[i], wall)]
        assert len(hits) == 1
        seen.add(hits[0])
        for sp in (s.prey, s.pred):
            pos, alive = np.asarray(sp.pos), np.asarray(sp.alive)
            assert not wall[pos[alive, 1], pos[alive, 0]].any()
        assert not (np.asarray(s.plant) & wall).any()
    assert len(seen) > 1  # resets draw different layouts
    held = evo.reset(dataclasses.replace(cfg, office_seed=office.TEST_SEED), jax.random.PRNGKey(0))
    test_bank = office.office_bank(cfg.size, office.TEST_SEED, cfg.office)
    assert any(np.array_equal(test_bank[i], np.asarray(held.terrain) == evo.WALL) for i in range(cfg.office))


def test_walls_block_both_species():
    _, cls = get_world("office")
    cfg = cls()
    s = evo.reset(cfg, jax.random.PRNGKey(1))
    key = jax.random.PRNGKey(2)
    wall = np.asarray(s.terrain) == evo.WALL
    for t in range(20):
        key, k1, k2, k3 = jax.random.split(key, 4)
        pa = jax.random.randint(k1, (cfg.max_prey,), 0, evo.NUM_ACTIONS)
        qa = jax.random.randint(k2, (cfg.max_pred,), 0, evo.NUM_ACTIONS)
        s = evo.advance(cfg, k3, s, pa, qa).state
        for sp in (s.prey, s.pred):
            pos, alive = np.asarray(sp.pos), np.asarray(sp.alive)
            assert not wall[pos[alive, 1], pos[alive, 0]].any()
