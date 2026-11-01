"""Stage 5a: a size gene, sexual breeding and a measure of lineage splits."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0)
CFG = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40, **QUIET)
KEY = jax.random.PRNGKey(5)
STILL = 0  # predprey.DELTAS[0] is (0, 0)
EAST = 3


def cfg(**kw):
    return evo.EvoConfig(**{**CFG.__dict__, **kw})


def step(c, s, prey_act=None, pred_act=None):
    prey_act = jnp.full(c.max_prey, STILL, jnp.int32) if prey_act is None else prey_act
    pred_act = jnp.full(c.max_pred, STILL, jnp.int32) if pred_act is None else pred_act
    return evo.advance(c, KEY, s, prey_act, pred_act)


def test_stay_action_keeps_animals_in_place():
    from simulations import predprey as pp
    pos = jnp.array([[3, 3]], jnp.int32)
    assert np.array_equal(np.asarray(pp.move(pos, jnp.array([STILL]), 12)), [[3, 3]])


def test_every_animal_has_a_size_gene_seen_in_the_last_channel():
    s = evo.reset(CFG, KEY)
    assert s.prey.genes.shape[1] == evo.ANIMAL_GENES == 7
    assert evo.GENE_NAMES["animal"] == ("speed", "sight", "breed", "size", "brain", "diet", "camo")
    assert CFG.founder_animal[3] == 0.5
    assert evo.OBS_CHANNELS == 11
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.5, 0.5, 0.8)])
    prey_obs, _ = evo.observe(CFG, s)
    o = np.asarray(prey_obs[0], np.float32)
    assert o.shape[-1] == 11 and np.allclose(o[..., 10], 0.8, atol=1e-2)
    # Six-field animals (no size) still work and get the founder size.
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.0, 0.5, 0.5)])
    assert float(s.prey.genes[0, 3]) == 0.5


def test_big_grazers_digest_tough_plants():
    c = cfg(sizes=True, size_cost=0.0)
    s = evo.make_state(c, prey=[(2, 2, 10.0, 0.0, 0.0, 1.0, 1.0), (8, 8, 10.0, 0.0, 0.0, 1.0, 0.0)],
                       plants=[(2, 2, 0.0, 1.0), (8, 8, 0.0, 1.0)])
    e = np.asarray(step(c, s).state.prey.energy)
    assert e[0] - e[1] == c.grass_energy * 0.5  # big: full value; small: half
    # Without sizes, both get half, as before.
    e = np.asarray(step(CFG, s).state.prey.energy)
    assert e[0] == e[1]


def test_only_small_prey_fit_into_burrows():
    terrain = np.zeros((12, 12), np.int8)
    terrain[5, 6] = evo.BURROW  # (x=6, y=5): east of both prey's column
    terrain[8, 6] = evo.BURROW
    c = cfg(sizes=True, burrow_size=0.5)
    s = evo.make_state(c, prey=[(5, 5, 20.0, 0.0, 0.0, 1.0, 0.7), (5, 8, 20.0, 0.0, 0.0, 1.0, 0.3)],
                       terrain=terrain)
    act = jnp.full(c.max_prey, STILL, jnp.int32).at[:2].set(EAST)
    pos = np.asarray(step(c, s, act).state.prey.pos)
    assert pos[0].tolist() == [5, 5] and pos[1].tolist() == [6, 8]
    pos = np.asarray(step(CFG, s, act).state.prey.pos)
    assert pos[0].tolist() == [6, 5]  # without sizes every prey fits


def test_small_predators_cannot_catch_big_prey():
    c = cfg(sizes=True, catch_margin=0.25, catch_scale=0.0)  # 0 = the hard cut-off (pp-niche-001's rule)
    s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, 0.9), (7, 7, 20.0, 0.0, 0.0, 1.0, 0.6)],
                       pred=[(3, 3, 30.0, 0.0, 0.0, 1.0, 0.5), (7, 7, 30.0, 0.0, 0.0, 1.0, 0.5)])
    alive = np.asarray(step(c, s).state.prey.alive)
    assert alive[0] and not alive[1]
    assert not np.asarray(step(CFG, s).state.prey.alive)[:2].any()  # without sizes both are eaten


def test_no_prey_is_immune_and_big_prey_are_bigger_meals():
    # pp-niche-001: under a hard size cut-off prey grew to 0.8+, predators shrank to 0.1 and catches hit zero.
    c = cfg(sizes=True, catch_scale=0.1, catch_energy=20.0)
    assert np.isclose(evo.catch_chance(c, 0.5, 0.5), 0.5)
    assert 0 < evo.catch_chance(c, 0.1, 0.9) < 0.01 and evo.catch_chance(c, 0.9, 0.1) > 0.99
    # A big prey is caught some of the time; each catch feeds by its size.
    s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, 0.7)], pred=[(3, 3, 30.0, 0.0, 0.0, 1.0, 0.6)])
    eaten = []
    for k in range(200):
        res = evo.advance(c, jax.random.PRNGKey(k), s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32))
        eaten.append(not bool(res.state.prey.alive[0]))
        if eaten[-1]:
            gained = float(res.state.pred.energy[0]) - float(step(c, evo.make_state(c, pred=[(3, 3, 30.0, 0.0, 0.0, 1.0, 0.6)])).state.pred.energy[0])
            assert np.isclose(gained, 20.0 * (0.5 + 0.7), atol=1e-4)
    assert 0.15 < np.mean(eaten) < 0.45  # sigmoid(-1) = 0.27


def test_assortative_mating_needs_a_mate_of_similar_size():
    c = cfg(mating=True, mate_radius=2, mate_tolerance=0.2, prey_repro_energy=10.0, breed_span=0.0)
    unlike = evo.make_state(c, prey=[(2, 2, 30.0, 0.0, 0.0, 0.0, 0.2), (3, 3, 4.0, 0.0, 0.0, 0.0, 0.8)])
    assert int(step(c, unlike).info.prey_births) == 0
    like = evo.make_state(c, prey=[(2, 2, 30.0, 0.0, 0.0, 0.0, 0.2), (3, 3, 4.0, 0.0, 0.0, 0.0, 0.3)])
    assert int(step(c, like).info.prey_births) == 1
    # A similar mate further away is chosen over an unlike one next door.
    pick = evo.make_state(c, prey=[(2, 2, 30.0, 0.0, 0.0, 0.0, 0.2), (3, 2, 4.0, 0.0, 0.0, 0.0, 0.9),
                                   (4, 2, 4.0, 0.0, 0.0, 0.0, 0.25)])
    assert int(step(c, pick).info.prey_births) == 1


def test_v3_size_does_not_change_catches_when_size_catch_is_off():
    c = cfg(sizes=True, size_catch=False)
    s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, 0.95)], pred=[(3, 3, 30.0, 0.0, 0.0, 1.0, 0.0)])
    res = step(c, s)
    assert not bool(res.state.prey.alive[0])  # a tiny predator still takes a huge prey
    assert np.isclose(float(res.state.pred.energy[0]) - 30.0, c.catch_energy - (c.pred_base_cost + 0.0), atol=0.6)


def test_stage_12c_catches_and_meals_come_apart():
    prey, pred = [(3, 3, 20.0, 0.0, 0.0, 1.0, 0.9)], [(3, 3, 10.0, 0.0, 0.0, 1.0, 0.1)]  # low energy: no birth to hide the gain
    alone = lambda c, size=0.1: float(step(c, evo.make_state(c, pred=[(3, 3, 10.0, 0.0, 0.0, 1.0, size)])).state.pred.energy[0])

    # A fixed chance of 1: a tiny hunter always takes a huge grazer, and the meal still follows its size.
    c = cfg(sizes=True, catch_scale=0.25, catch_energy=25.0, catch_fixed=1.0)
    res = step(c, evo.make_state(c, prey=prey, pred=pred))
    assert not bool(res.state.prey.alive[0])
    assert np.isclose(float(res.state.pred.energy[0]) - alone(c), 25.0 * 1.4, atol=1e-4)

    # Flat meals: a catch is worth the same, and the chance still follows sizes.
    c = cfg(sizes=True, catch_scale=0.25, catch_energy=25.0, flat_meals=True)
    assert evo.catch_chance(c, 0.1, 0.9) < 0.05
    s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, 0.1)], pred=[(3, 3, 10.0, 0.0, 0.0, 1.0, 0.9)])
    gains = []
    for k in range(40):
        res = evo.advance(c, jax.random.PRNGKey(k), s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32))
        if not bool(res.state.prey.alive[0]):
            gains.append(float(res.state.pred.energy[0]) - alone(c, 0.9))
    assert gains and np.isclose(np.mean(gains), 25.0, atol=1e-3) and np.ptp(gains) < 1e-3

    # A fixed chance: the same odds whatever the sizes.
    c = cfg(sizes=True, catch_scale=0.25, catch_fixed=0.73)
    assert np.isclose(evo.catch_chance(c, 0.1, 0.9), 0.73) and np.isclose(evo.catch_chance(c, 0.9, 0.1), 0.73)
    s = evo.make_state(c, prey=prey, pred=pred)
    eaten = [not bool(evo.advance(c, jax.random.PRNGKey(k), s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32)).state.prey.alive[0]) for k in range(200)]
    assert 0.63 < np.mean(eaten) < 0.83


def test_stage_12c_worlds_differ_only_as_intended():
    from simulations.worlds import get_world
    certain, flat, coin = (get_world(n)[1]() for n in ("bulk_certain", "bulk_flat", "bulk_coin"))
    assert certain.catch_fixed == 1.0 and not certain.flat_meals
    assert flat.catch_fixed == 0.0 and flat.flat_meals
    assert coin.catch_fixed == 0.73 and coin.flat_meals
    assert all(w.sizes and w.size_catch and w.catch_scale == 0.25 and w.size_cost == 0.5 and w.stealth_radius == 1 for w in (certain, flat, coin))


def test_stage_12d_only_meals_change():
    from dataclasses import asdict
    from simulations.worlds import get_world
    for rich, poor, energy in (("bulk_rich", "bulk_base", 37.0), ("bulk_certain_rich", "bulk_certain", 28.0), ("bulk_coin_rich", "bulk_coin", 34.0)):
        r, p = get_world(rich)[1](), get_world(poor)[1]()
        assert r.catch_energy == energy and {**asdict(r), "catch_energy": p.catch_energy} == asdict(p)


def test_stage_12e_size_can_be_free_for_hunters_only():
    def spent(c, size):
        s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, size)], pred=[(8, 8, 30.0, 0.0, 0.0, 1.0, size)])
        res = step(c, s)
        return 20.0 - float(res.state.prey.energy[0]), 30.0 - float(res.state.pred.energy[0])
    usual = cfg(sizes=True, size_cost=0.5)
    free = cfg(sizes=True, size_cost=0.5, pred_size_cost=0.0)
    assert np.isclose(spent(usual, 1.0)[1] - spent(usual, 0.0)[1], 0.5, atol=1e-4)
    assert np.isclose(spent(free, 1.0)[1], spent(free, 0.0)[1], atol=1e-4)
    assert np.isclose(spent(free, 1.0)[0] - spent(free, 0.0)[0], 0.5, atol=0.2)  # grazers still pay (digesting aside)


def test_stage_12e_a_big_meal_fits_whole_under_the_higher_cap():
    from simulations.worlds import get_world
    rich = get_world("bulk_cap_rich")[1]()
    assert rich.catch_energy == 37.0 and rich.pred_energy_cap == 120.0 and rich.size_catch and rich.catch_fixed == 0.0
    threshold = rich.pred_repro_energy * (0.5 + rich.breed_span)  # the highest a hunter holds before breeding
    assert threshold + rich.catch_energy * 1.5 <= rich.pred_energy_cap
    burn = get_world("bulk_cap_burn")[1]()
    assert burn.catch_energy == 47.0 and burn.pred_base_cost == 0.96 and burn.catch_fixed == 0.73 and burn.flat_meals
    assert threshold + burn.catch_energy <= burn.pred_energy_cap
    coin, free = get_world("bulk_cap_coin")[1](), get_world("bulk_cap_free")[1]()
    assert coin.catch_energy == 34.0 and coin.pred_energy_cap == 120.0
    assert free.pred_size_cost == 0.0 and free.size_cost == 0.5 and free.catch_energy == 25.0 and free.pred_energy_cap == 120.0


def test_stage_12f_feasts_fit_and_still_race():
    from simulations.worlds import get_world
    for name, energy in (("bulk_feast_50", 50.0), ("bulk_feast_70", 70.0)):
        c = get_world(name)[1]()
        threshold = c.pred_repro_energy * (0.5 + c.breed_span)
        assert c.catch_energy == energy and threshold + energy * 1.5 <= c.pred_energy_cap
        assert c.sizes and c.size_catch and c.catch_fixed == 0.0 and not c.flat_meals and c.pred_size_cost < 0


def test_stage_12g_only_meals_and_their_cap_change():
    from dataclasses import asdict
    from simulations.worlds import get_world
    for climate in ("meadow", "forest", "tundra"):
        feast, plain = get_world(f"bulk_feast_{climate}")[1](), get_world(f"bulk_{climate}")[1]()
        assert feast.catch_energy == 50.0 and feast.pred_energy_cap == 170.0
        assert {**asdict(feast), "catch_energy": plain.catch_energy, "pred_energy_cap": plain.pred_energy_cap} == asdict(plain)


def test_stage_12h_winter_is_tundras_and_size_can_be_free_for_grazers():
    from dataclasses import asdict
    from simulations.worlds import get_world
    feast, winter, free, tundra = (get_world(n)[1]() for n in ("bulk_feast_50", "bulk_winter", "bulk_winter_free", "bulk_tundra"))
    assert winter.winter_len == tundra.winter_len == 60 and winter.winter_value == tundra.winter_value
    assert {**asdict(winter), "winter_len": feast.winter_len} == asdict(feast)
    assert {**asdict(free), "prey_size_cost": winter.prey_size_cost} == asdict(winter)
    def grazer_spent(c, size):
        s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, size)])
        return 20.0 - float(step(c, s).state.prey.energy[0])
    usual = cfg(sizes=True, size_cost=0.5)
    gone = cfg(sizes=True, size_cost=0.5, prey_size_cost=0.0)
    assert np.isclose(grazer_spent(usual, 1.0) - grazer_spent(usual, 0.0), 0.5, atol=0.2)
    assert np.isclose(grazer_spent(gone, 1.0), grazer_spent(gone, 0.0), atol=0.2)


def test_stage_12i_only_the_meal_changes():
    from dataclasses import asdict
    from simulations.worlds import get_world
    flat, winter = get_world("bulk_winter_flat")[1](), get_world("bulk_winter")[1]()
    assert flat.catch_energy == 71.0 and flat.flat_meals and flat.pred_energy_cap >= 58.5 + 71.0
    assert {**asdict(flat), "catch_energy": winter.catch_energy, "flat_meals": winter.flat_meals} == asdict(winter)


def test_stage_13a_a_grazer_sees_a_hunter_only_within_its_camouflage():
    c = cfg(camo_gene=True)
    r = c.view // 2
    def seen_at(gap, camo):
        s = evo.make_state(c, prey=[(3, 5, 20.0, 0.0, 1.0, 0.5, 0.5)], pred=[(3 + gap, 5, 30.0, 0.0, 0.0, 0.5, 0.5)])
        s = s._replace(pred=s.pred._replace(genes=s.pred.genes.at[0, 6].set(camo)))
        prey_obs, pred_obs = evo.observe(c, s)
        return float(prey_obs[0, r, r + gap, 3]) > 0
    assert int(evo.camo_radius(0.0)) == 4 and int(evo.camo_radius(1.0)) == 1 and int(evo.camo_radius(0.1)) == 4
    assert seen_at(1, 1.0) and not seen_at(2, 1.0)  # fully hidden: only from the next cell
    assert seen_at(2, 0.5) and not seen_at(3, 0.5)  # 4 - floor(2) = 2
    assert seen_at(3, 0.1) and seen_at(4, 0.1)  # hides nothing, within the grazer's sight of 4


def test_stage_13a_camouflage_costs_only_hunters_and_only_where_it_works():
    def spent(c, camo):
        s = evo.make_state(c, prey=[(3, 3, 20.0, 0.0, 0.0, 1.0, 0.5)], pred=[(8, 8, 30.0, 0.0, 0.0, 1.0, 0.5)])
        s = s._replace(pred=s.pred._replace(genes=s.pred.genes.at[0, 6].set(camo)), prey=s.prey._replace(genes=s.prey.genes.at[0, 6].set(camo)))
        r = step(c, s)
        return 30.0 - float(r.state.pred.energy[0]), 20.0 - float(r.state.prey.energy[0])
    priced = cfg(camo_gene=True, camo_cost=0.25)
    assert np.isclose(spent(priced, 1.0)[0] - spent(priced, 0.0)[0], 0.25, atol=1e-4)
    assert np.isclose(spent(priced, 1.0)[1], spent(priced, 0.0)[1], atol=0.2)
    off = cfg(camo_gene=False, camo_cost=0.25)
    assert np.isclose(spent(off, 1.0)[0], spent(off, 0.0)[0], atol=1e-4)


def test_stage_13a_worlds_start_visible_and_differ_only_as_intended():
    from simulations.worlds import get_world
    neutral, free, priced = (get_world(n)[1]() for n in ("camo_neutral", "camo_free", "camo"))
    assert neutral.founder_animal[6] == 0.1 and not neutral.camo_gene and neutral.stealth_radius == 0 and not neutral.sizes
    assert free.camo_gene and free.camo_cost == 0.0
    assert priced.camo_gene and priced.camo_cost == 0.25 and priced.seed_rate == 0.2 and priced.catch_energy == 25.0


def test_stage_13b_only_the_price_or_the_founders_change():
    from dataclasses import asdict
    from simulations.worlds import get_world
    base = get_world("camo")[1]()
    assert int(evo.camo_radius(0.1)) == 4 and int(evo.camo_radius(0.2)) == 3  # founders either side of the first step
    start = get_world("camo_start")[1]()
    assert start.founder_animal[6] == 0.2 and {**asdict(start), "founder_animal": base.founder_animal} == asdict(base)
    for name, price in (("camo_10", 0.10), ("camo_05", 0.05)):
        c = get_world(name)[1]()
        assert c.camo_cost == price and {**asdict(c), "camo_cost": base.camo_cost} == asdict(base)


def test_stage_13c_continents_take_the_gene_and_keep_their_climate():
    from dataclasses import asdict
    from simulations.worlds import get_world
    for climate, base in (("meadow", "ContinentConfig"), ("forest", "ForestConfig"), ("tundra", "TundraConfig")):
        c, b = get_world(f"camo_{climate}")[1](), getattr(evo, base)()
        assert c.camo_gene and c.camo_cost == 0.10 and c.stealth_radius == 0 and c.founder_animal[6] == 0.1 and not c.sizes and c.size == 256
        changed = {"camo_gene": b.camo_gene, "camo_cost": b.camo_cost, "stealth_radius": b.stealth_radius, "founder_animal": b.founder_animal}
        assert {**asdict(c), **changed} == asdict(b)


def test_v3_two_fixed_plant_kinds():
    c = cfg(plant_kinds=True, plant_sigma=0.0, gene_sigma=0.05, seed_rate=1.0, tough_seed_cost=0.0, germinate_rate=0.05)
    s = evo.reset(c, KEY)
    tough = np.asarray(s.plant_genes)[np.asarray(s.plant)][:, 1]
    assert set(np.unique(tough)) <= {0.0, 1.0} and 0.3 < tough.mean() < 0.7
    for k in range(5):  # seeding and germination copy a kind without mutating it
        s = evo.advance(c, jax.random.PRNGKey(k), s, jnp.zeros(c.max_prey, jnp.int32), jnp.zeros(c.max_pred, jnp.int32)).state
    genes = np.asarray(s.plant_genes)[np.asarray(s.plant)]
    assert set(np.unique(genes[:, 1])) <= {0.0, 1.0}
    assert len(np.unique(genes[:, 0])) <= 2  # spread unmutated as well
    # Animals still mutate at gene_sigma.
    assert c.gene_sigma == 0.05


def test_niches_v3_world():
    from simulations.worlds import get_world
    c = get_world("niches")[1]()
    assert c.sizes and c.mating and not c.size_catch and c.plant_kinds and c.plant_sigma == 0.0


def test_v4_small_grazers_digest_soft_plants_best():
    c = cfg(sizes=True, size_cost=0.0, size_tradeoff=True)
    big, small, mid = (2, 2, 10.0, 0.0, 0.0, 1.0, 1.0), (8, 8, 10.0, 0.0, 0.0, 1.0, 0.0), (5, 5, 10.0, 0.0, 0.0, 1.0, 0.5)
    g = c.grass_energy
    for tough, want in ((0.0, (0.5, 1.0, 0.75)), (1.0, (1.0, 0.5, 0.75))):
        s = evo.make_state(c, prey=[big, small, mid], plants=[(2, 2, 0.0, tough), (8, 8, 0.0, tough), (5, 5, 0.0, tough)])
        gain = np.asarray(step(c, s).state.prey.energy)[:3] - 10.0
        want = np.array(want) * g
        assert np.allclose(gain - gain[0], want - want[0])  # every animal pays the same step cost
        assert np.argsort(gain)[1] == 2  # the middle size is second best at both
    assert get_world_cfg("niches").size_tradeoff


def get_world_cfg(name):
    from simulations.worlds import get_world
    return get_world(name)[1]()


def test_size_has_upkeep():
    c = cfg(sizes=True, size_cost=0.5)
    s = evo.make_state(c, prey=[(2, 2, 20.0, 0.0, 0.0, 1.0, 1.0), (8, 8, 20.0, 0.0, 0.0, 1.0, 0.0)])
    e = np.asarray(step(c, s).state.prey.energy)
    assert np.isclose(e[1] - e[0], 0.5)


def test_sexual_breeding_needs_a_mate_nearby_and_mixes_genes():
    c = cfg(mating=True, mate_radius=2, prey_repro_energy=10.0, breed_span=0.0)
    lone = evo.make_state(c, prey=[(2, 2, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0), (9, 9, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0)])
    assert int(step(c, lone).info.prey_births) == 0  # the only other prey is far away
    pair = evo.make_state(c, prey=[(2, 2, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0), (3, 3, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0)])
    kids = []
    for k in range(20):
        res = evo.advance(c, jax.random.PRNGKey(k), pair, jnp.full(c.max_prey, STILL, jnp.int32),
                          jnp.full(c.max_pred, STILL, jnp.int32))
        assert int(res.info.prey_births) == 1  # only the ready one breeds
        kids.append(np.asarray(res.state.prey.genes[2]))
    kids = np.array(kids)[:, :5]  # both parents were written without a diet gene
    assert set(np.unique(kids)) <= {0.0, 1.0}  # each gene from one parent or the other
    assert 0 < kids.mean() < 1 and len({tuple(k) for k in kids}) > 2  # genes mix
    # Without mating the lone animal breeds on its own and copies itself.
    res = step(cfg(prey_repro_energy=10.0, breed_span=0.0), lone)
    assert int(res.info.prey_births) == 1 and np.allclose(np.asarray(res.state.prey.genes[2])[:5], 0.0)


def test_three_gene_snapshots_load_with_founder_size():
    d = {"prey_genes": np.full((5, 3), 0.3), "prey_gen": np.ones(5), "pred_genes": np.full((2, 3), 0.3),
         "pred_gen": np.ones(2), "plant_genes": np.full((4, 2), 0.3)}
    pool = evo.pool_from_arrays(d)
    assert pool.prey_genes.shape == (5, evo.ANIMAL_GENES) and np.allclose(np.asarray(pool.prey_genes)[:, 3:], 0.5)
    from scripts.league import match_pool
    p = match_pool(d, d)
    w = np.asarray(p.prey_w) > 0
    assert np.allclose(np.asarray(p.prey_genes)[w, 3], 0.5)


def test_niches_world_is_registered():
    from simulations.worlds import get_world
    module, config_cls = get_world("niches")
    c = config_cls()
    assert module is evo and c.sizes and c.mating and c.burrows > 0 and c.walls > 0
    assert evo.EvoConfig().size_catch and evo.EvoConfig().plant_sigma < 0  # earlier worlds unchanged


# --- measuring a split --------------------------------------------------------------------

def test_two_clusters_found_and_separated():
    import scripts.species as species
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(0.5, 0.05, 300)] * 3 + [rng.normal(0.15, 0.05, 300)])
    b = np.column_stack([rng.normal(0.5, 0.05, 100)] * 3 + [rng.normal(0.85, 0.05, 100)])
    out = species.split(np.vstack([a, b]))
    assert out["separation"] > 2 and abs(out["minority"] - 0.25) < 0.02 and out["main_gene"] == "size"
    assert abs(out["size_gap"] - 0.7) < 0.03 and species.split_found(out)
    one = species.split(np.column_stack([rng.normal(0.5, 0.1, 400)] * 4))
    assert one["separation"] < 2


def test_clumps_of_clones_are_not_a_split():
    # Resets copy genomes from the pool, so a snapshot holds clumps of identical animals. A narrow
    # clump must not count as a lineage: one bell curve of sizes sampled as 12 clone families.
    import scripts.species as species
    rng = np.random.default_rng(1)
    families = np.column_stack([rng.normal(0.5, 0.05, 12)] * 3 + [rng.normal(0.45, 0.06, 12)])
    clones = families[rng.integers(0, 12, 600)]
    assert not species.split_found(species.split(clones))
    # Two size groups only 0.15 apart are not two bodies either, however clean the gap.
    a = np.column_stack([np.full(300, 0.5)] * 3 + [rng.normal(0.40, 0.01, 300)])
    b = np.column_stack([np.full(200, 0.5)] * 3 + [rng.normal(0.55, 0.01, 200)])
    assert not species.split_found(species.split(np.vstack([a, b])))


def test_species_script_on_tiny_run(tmp_path):
    import scripts.species as species
    import scripts.train as train
    assert train.main(["--tiny", "--world", "niches", "--runs-dir", str(tmp_path), "--run-id", "n1",
                       "--updates", "3", "--ckpt-every", "1"]) == 0
    code = species.main(["--run-dir", str(tmp_path / "n1")])
    out = json.loads((tmp_path / "n1" / "species.json").read_text())
    assert code == (0 if out["pass"] else 1)
    assert [r["update"] for r in out["snapshots"]] == [1, 2, 3]
    assert set(out["snapshots"][0]) >= {"update", "prey", "pred"}


def test_v5_grass_survives_bites_so_both_foods_last():
    c = cfg(sizes=True, size_cost=0.0, plant_kinds=True, plant_sigma=0.0, kinds_persist=True, bite_regrow=0.0)
    s = evo.make_state(c, prey=[(2, 2, 10.0, 0.0, 0.0, 1.0, 0.0)], plants=[(2, 2, 0.0, 0.0)])
    r = step(c, s)
    assert float(r.state.prey.energy[0]) > 10.0 - 2  # it ate
    assert bool(r.state.plant[2, 2]) and bool(r.state.bitten[2, 2])  # the grass is bitten, not gone
    assert get_world_cfg("niches").kinds_persist


def test_v5_plant_kinds_hold_their_share_under_grazing():
    from simulations.worlds import get_world
    c = get_world("niches")[1](size=24, max_prey=64, max_pred=4, init_prey=48, init_pred=1, max_steps=400)
    key = jax.random.PRNGKey(1)
    s = evo.reset(c, key)

    def body(s, k):
        k1, k2 = jax.random.split(k)
        a = jax.random.randint(k1, (c.max_prey,), 0, 5)
        return evo.advance(c, k2, s, a, jnp.zeros(c.max_pred, jnp.int32)).state, None

    s, _ = jax.lax.scan(body, s, jax.random.split(key, 400))
    tough = np.asarray(s.plant_genes)[np.asarray(s.plant)][:, 1]
    assert len(tough) > 50 and 0.25 < tough.mean() < 0.75
