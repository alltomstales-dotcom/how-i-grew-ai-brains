import jax
import jax.numpy as jnp
import numpy as np

from simulations import evo

CFG = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40)


def stack(*states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def uniform_pool(value, gen=0, n=5):
    return evo.GenePool(
        prey_genes=jnp.full((n, evo.ANIMAL_GENES), value), prey_gen=jnp.full((n,), gen, jnp.int32), prey_w=jnp.ones(n),
        pred_genes=jnp.full((n, evo.ANIMAL_GENES), value), pred_gen=jnp.full((n,), gen, jnp.int32), pred_w=jnp.ones(n),
        plant_genes=jnp.full((n, 2), value), plant_w=jnp.ones(n),
    )


# --- Task 1: genes, reset, observe -------------------------------------------------------

def test_reset_founders_in_range():
    s = evo.reset(CFG, jax.random.PRNGKey(0))
    for sp, n in ((s.prey, 4), (s.pred, 2)):
        g = np.asarray(sp.genes)[np.asarray(sp.alive)]
        assert g.shape == (n, evo.ANIMAL_GENES)
        assert (g >= 0).all() and (g <= 1).all()
        assert np.allclose(g.mean(0), [0.2, 0.5, 0.5, 0.5, 0.5, 0.5], atol=0.12)
        assert (np.asarray(sp.gen) == 0).all()
    plant = np.asarray(s.plant)
    assert 0.2 < plant.mean() < 0.8
    pg = np.asarray(s.plant_genes)[plant]
    assert np.allclose(pg.mean(0), [0.5, 0.2], atol=0.05)


def test_reset_draws_founders_from_pool():
    s = evo.reset(CFG, jax.random.PRNGKey(1), uniform_pool(0.9, gen=7))
    for sp in (s.prey, s.pred):
        alive = np.asarray(sp.alive)
        np.testing.assert_allclose(np.asarray(sp.genes)[alive], 0.9)
        assert (np.asarray(sp.gen)[alive] == 7).all()
    np.testing.assert_allclose(np.asarray(s.plant_genes)[np.asarray(s.plant)], 0.9)


def test_gene_pool_weights_follow_alive():
    a = evo.make_state(CFG, prey=[(1, 1, 10.0, 0.9, 0.9, 0.9)], plants=[(2, 2, 0.3, 0.4)])
    b = evo.make_state(CFG, pred=[(3, 3, 10.0, 0.1, 0.1, 0.1)])
    pool = evo.gene_pool(stack(a, b))
    assert pool.prey_genes.shape == (2 * CFG.max_prey, evo.ANIMAL_GENES)
    assert float(pool.prey_w.sum()) == 1.0 and float(pool.pred_w.sum()) == 1.0
    assert float(pool.plant_w.sum()) == 1.0
    np.testing.assert_allclose(np.asarray(pool.plant_genes)[np.asarray(pool.plant_w) > 0], [[0.3, 0.4]])


def test_empty_pool_falls_back_to_founders():
    pool = uniform_pool(0.9)._replace(prey_w=jnp.zeros(5), pred_w=jnp.zeros(5), plant_w=jnp.zeros(5))
    s = evo.reset(CFG, jax.random.PRNGKey(2), pool)
    g = np.asarray(s.prey.genes)[np.asarray(s.prey.alive)]
    assert np.isfinite(g).all()
    assert np.allclose(g.mean(0), [0.2, 0.5, 0.5, 0.5, 0.5, 0.5], atol=0.12)
    pg = np.asarray(s.plant_genes)[np.asarray(s.plant)]
    assert np.allclose(pg.mean(0), [0.5, 0.2], atol=0.05)


def test_sight_radius_mapping():
    g = jnp.array([0.0, 0.16, 0.17, 0.5, 0.83, 0.84, 1.0])
    assert np.asarray(evo.sight_radius(g)).tolist() == [1, 1, 2, 3, 3, 4, 4]


def test_observe_masks_outside_sight():
    blind = evo.make_state(CFG, prey=[(5, 5, 10.0, 0.0, 0.0, 0.5)], pred=[(7, 5, 10.0, 0.0, 0.0, 0.5)],
                           plants=[(5, 6, 0.5, 0.25)])
    prey_obs, _ = evo.observe(CFG, blind)
    o = np.asarray(prey_obs[0], np.float32)
    assert o[:, :, 3].sum() == 0  # predator 2 cells away, radius 1
    assert o[5, 4, 0] == 1.0 and o[5, 4, 1] == 0.25  # plant 1 cell south is visible
    keen = blind._replace(prey=blind.prey._replace(genes=blind.prey.genes.at[0, 1].set(1.0)))
    prey_obs, _ = evo.observe(CFG, keen)
    o = np.asarray(prey_obs[0], np.float32)
    assert o[4, 6, 3] == 1.0
    assert o[:, :, 3].sum() == 1.0


def test_observe_self_channels():
    s = evo.make_state(CFG, prey=[(5, 5, 20.0, 0.25, 0.0, 0.75)])
    prey_obs, _ = evo.observe(CFG, s)
    o = np.asarray(prey_obs[0], np.float32)
    assert prey_obs.shape[-1] == evo.OBS_CHANNELS == 11
    assert prey_obs.dtype == jnp.bfloat16
    # Own-state channels are not masked by sight, even in the corners.
    np.testing.assert_allclose(o[0, 0, 4:8], [20.0 / CFG.prey_energy_cap, 0.25, 0.0, 0.75], atol=1e-2)
    empty_slot = np.asarray(prey_obs[1], np.float32)
    assert (empty_slot == 0).all()


# --- Task 2: step ------------------------------------------------------------------------

QUIET = dict(seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0)
KEY = jax.random.PRNGKey(3)
STAY = jnp.zeros(CFG.max_prey, jnp.int32)
PRED_STAY = jnp.zeros(CFG.max_pred, jnp.int32)
EAST = 3


def cfg(**kw):
    return evo.EvoConfig(**{**CFG.__dict__, **QUIET, **kw})


def step(c, s, prey_act=STAY, pred_act=PRED_STAY, key=KEY):
    return evo.advance(c, key, s, prey_act, pred_act)


def test_speed_one_moves_two_cells():
    c = cfg()
    s = evo.make_state(c, prey=[(5, 5, 20.0, 1.0, 0.0, 1.0), (5, 7, 20.0, 0.0, 0.0, 1.0),
                                (11, 3, 20.0, 1.0, 0.0, 1.0)])
    res = step(c, s, STAY.at[:3].set(EAST))
    assert np.asarray(res.state.prey.pos[:3]).tolist() == [[7, 5], [6, 7], [1, 3]]


def test_stay_never_moves():
    c = cfg()
    s = evo.make_state(c, prey=[(5, 5, 20.0, 1.0, 0.0, 1.0)])
    assert np.asarray(step(c, s).state.prey.pos[0]).tolist() == [5, 5]


def test_metabolism_includes_trait_costs():
    c = cfg()
    s = evo.make_state(c, prey=[(5, 5, 20.0, 0.5, 1.0, 1.0)], pred=[(0, 0, 30.0, 0.0, 0.0, 1.0)])
    res = step(c, s)
    # 1 base + 1.0 * 0.5**2 speed + 0.5 * (4 - 1) / 3 sight
    assert abs(float(res.state.prey.energy[0]) - (20.0 - 1.75)) < 1e-5
    # Predators idle cheaply (base 0.5): pp-evo-001's predators starved at 1 per step.
    assert abs(float(res.state.pred.energy[0]) - 29.5) < 1e-5


def test_breed_threshold_per_animal():
    c = cfg()
    # Threshold 30 * (0.5 + 0.8 b): b=0 breeds at 15, b=1 needs 39.
    s = evo.make_state(c, prey=[(1, 1, 16.0, 0.0, 0.0, 0.0), (3, 3, 40.0, 0.0, 0.0, 1.0),
                                (5, 5, 38.0, 0.0, 0.0, 1.0)])
    res = step(c, s)
    assert int(res.state.prey.alive.sum()) == 5
    assert abs(float(res.state.prey.energy[0]) - 7.5) < 1e-5
    assert abs(float(res.state.prey.energy[1]) - 19.5) < 1e-5
    assert abs(float(res.state.prey.energy[2]) - 37.0) < 1e-5
    assert int(res.info.prey_births) == 2


def test_every_breed_gene_is_fertile_below_the_energy_cap():
    # pp-evo-001's predators drifted to breed 0.85, a threshold above their cap: no births at all.
    c = evo.EvoConfig()
    for repro, cap in ((c.prey_repro_energy, c.prey_energy_cap), (c.pred_repro_energy, c.pred_energy_cap)):
        assert repro * (0.5 + c.breed_span) < cap


def test_plants_that_never_spread_still_die():
    # pp-evo-001's plants went to spread 0.02, toughness 0.99: immortal rocks.
    c = cfg(die_rate=1.0, plant_base_death=1.0)
    s = evo.make_state(c, plants=[(7, 7, 0.0, 1.0)])
    assert not bool(step(c, s).state.plant[7, 7])


def test_child_inherits_genes_gen_and_tag():
    c = cfg(gene_sigma=0.0)
    s = evo.make_state(c, prey=[(2, 2, 20.0, 0.3, 0.6, 0.0)])
    s = s._replace(prey=s.prey._replace(gen=s.prey.gen.at[0].set(4), tag=s.prey.tag.at[0].set(1)))
    p = step(c, s).state.prey
    assert bool(p.alive[1])
    np.testing.assert_allclose(np.asarray(p.genes[1]), [0.3, 0.6, 0.0, evo.FOUNDER_SIZE, evo.FOUNDER_SIZE, evo.FOUNDER_SIZE])
    assert int(p.gen[1]) == 5 and int(p.tag[1]) == 1
    assert np.asarray(p.pos[1]).tolist() == [2, 2]


def test_mutation_stays_in_range():
    c = cfg(gene_sigma=1.0)
    s = evo.make_state(c, prey=[(i, 0, 30.0, 0.5, 0.5, 0.0) for i in range(4)])
    p = step(c, s).state.prey
    children = np.asarray(p.genes)[4:8]
    assert np.asarray(p.alive[4:8]).all()
    assert (children >= 0).all() and (children <= 1).all()
    assert not np.allclose(children, 0.5)


def test_tough_plant_survives_grazing():
    c = cfg()
    s = evo.make_state(c, prey=[(2, 2, 20.0, 0.0, 0.0, 1.0), (6, 6, 20.0, 0.0, 0.0, 1.0)],
                       plants=[(2, 2, 0.5, 1.0), (6, 6, 0.5, 0.0)])
    res = step(c, s)
    assert bool(res.state.plant[2, 2]) and not bool(res.state.plant[6, 6])
    np.testing.assert_allclose(np.asarray(res.state.prey.energy[:2]), [20 + 2.5 - 1, 20 + 5 - 1], atol=1e-5)


def test_plant_seeds_one_neighbour():
    c = cfg(seed_rate=1.0, tough_seed_cost=0.0, gene_sigma=0.0)
    s = evo.make_state(c, plants=[(5, 5, 1.0, 0.3)])
    plant = np.asarray(step(c, s).state.plant)
    assert plant.sum() == 2 and plant[5, 5]
    (y, x), = [yx for yx in zip(*np.nonzero(plant)) if yx != (5, 5)]
    assert abs(y - 5) + abs(x - 5) == 1
    np.testing.assert_allclose(np.asarray(step(c, s).state.plant_genes[y, x]), [1.0, 0.3])


def test_seed_conflict_one_parent():
    c = cfg(seed_rate=1.0, tough_seed_cost=0.0, gene_sigma=0.0)
    plants = [(x, y, 0.0, 0.0) for y in range(c.size) for x in range(c.size)
              if (x, y) not in ((5, 5), (4, 5), (6, 5))]
    s = evo.make_state(c, plants=plants + [(4, 5, 1.0, 0.1), (6, 5, 1.0, 0.7)])
    seen = set()
    for k in jax.random.split(KEY, 8):
        out = step(c, s, key=k).state
        assert bool(out.plant.all())
        g = tuple(round(v, 3) for v in np.asarray(out.plant_genes[5, 5]).tolist())
        assert g in {(1.0, 0.1), (1.0, 0.7)}
        seen.add(g)
    assert len(seen) == 2


def test_plant_death():
    c = cfg(die_rate=1.0, plant_base_death=0.0)
    s = evo.make_state(c, plants=[(5, 5, 1.0, 0.0), (7, 7, 0.0, 0.0)])
    plant = np.asarray(step(c, s).state.plant)
    assert not plant[5, 5] and plant[7, 7]


def test_germination_on_empty_world():
    c = cfg(germinate_rate=1.0)
    out = step(c, evo.make_state(c)).state
    assert bool(out.plant.all())
    np.testing.assert_allclose(np.asarray(out.plant_genes).reshape(-1, 2).mean(0), [0.5, 0.2], atol=0.03)


def test_batch_step_reseeds_done_world_from_pool():
    c = cfg(gene_sigma=0.0, max_steps=10)
    prey = [(1, 1, 20.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9), (3, 3, 20.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)]
    pred = [(8, 1, 20.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9), (8, 3, 20.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)]
    plants = [(5, 5, 0.9, 0.9)]
    done_world = evo.make_state(c, prey=prey, pred=pred, plants=plants, t=9)
    live_world = evo.make_state(c, prey=prey, pred=pred, plants=plants, t=0)
    res, prey_obs, _ = evo.batch_step(c, KEY, stack(done_world, live_world),
                                      jnp.zeros((2, c.max_prey), jnp.int32), jnp.zeros((2, c.max_pred), jnp.int32))
    assert np.asarray(res.done).tolist() == [True, False]
    assert np.asarray(res.state.t).tolist() == [0, 1]
    fresh = jax.tree.map(lambda x: x[0], res.state)
    np.testing.assert_allclose(np.asarray(fresh.prey.genes)[np.asarray(fresh.prey.alive)], 0.9)
    np.testing.assert_allclose(np.asarray(fresh.plant_genes)[np.asarray(fresh.plant)], 0.9)
    assert int(fresh.prey.alive.sum()) == c.init_prey
    assert prey_obs.shape == (2, c.max_prey, c.view, c.view, evo.OBS_CHANNELS)


def test_extra_metrics_finite():
    s = evo.reset(CFG, KEY)
    info = step(evo.EvoConfig(**CFG.__dict__), s).info
    m = {k: float(v) for k, v in evo.extra_metrics(info).items()}
    expected = {f"{sp}_{g}{suffix}" for sp in ("prey", "pred") for g in ("speed", "sight", "breed")
                for suffix in ("", "_std")}
    expected |= {f"plant_{g}{suffix}" for g in ("spread", "tough") for suffix in ("", "_std")}
    expected |= {"plant_cover", "prey_gen", "pred_gen"}
    assert expected <= set(m)
    assert all(np.isfinite(v) for v in m.values())
    assert abs(m["prey_speed"] - 0.2) < 0.15


# --- Task 3: world-generic trainer -------------------------------------------------------

def test_get_world_registry():
    from simulations import predprey
    from simulations.worlds import get_world
    assert get_world("evo") == (evo, evo.EvoConfig)
    assert get_world("predprey") == (predprey, predprey.WorldConfig)


def test_predprey_batch_step_matches_autoreset():
    from functools import partial
    from simulations import predprey as pp
    c = pp.WorldConfig(size=12, max_prey=8, max_pred=4, init_prey=4, init_pred=2, max_steps=40)
    states = jax.vmap(partial(pp.reset, c))(jax.random.split(KEY, 3))
    a, b = jnp.ones((3, 8), jnp.int32), jnp.ones((3, 4), jnp.int32)
    got = pp.batch_step(c, KEY, states, a, b)
    want = jax.vmap(partial(pp.step_autoreset, c))(jax.random.split(KEY, 3), states, a, b)
    for x, y in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert pp.extra_metrics(got[0].info) == {}


def test_evo_trainer_smoke_tiny():
    import math
    from agents.ppo import make_trainer, tiny_configs
    wcfg, tcfg = tiny_configs("evo")
    assert isinstance(wcfg, evo.EvoConfig)
    trainer = make_trainer(wcfg, tcfg, world=evo)
    runner = trainer.init(jax.random.PRNGKey(0))
    assert runner.prey_obs.shape[-1] == evo.OBS_CHANNELS
    for _ in range(2):
        runner, metrics = trainer.update(runner)
    metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
    assert {"prey_speed", "pred_breed", "plant_tough", "plant_cover", "prey_entropy"} <= set(metrics)
    assert all(math.isfinite(v) for v in metrics.values())


# --- Task 4: rendering, evaluation and lineage share ---------------------------------------

def test_reset_mixed_tags_half():
    s = evo.reset_mixed(CFG, KEY, uniform_pool(0.9, gen=3))
    for sp, n in ((s.prey, CFG.init_prey), (s.pred, CFG.init_pred)):
        alive = np.asarray(sp.alive)
        tag = np.asarray(sp.tag)[alive]
        genes = np.asarray(sp.genes)[alive]
        assert alive.sum() == n and (tag == 1).sum() == n // 2
        np.testing.assert_allclose(genes[tag == 1], 0.9)
        assert np.allclose(genes[tag == 0].mean(0), [0.2, 0.5, 0.5, 0.5, 0.5, 0.5], atol=0.15)
        assert (np.asarray(sp.gen)[alive][tag == 0] == 0).all()
    np.testing.assert_allclose(np.asarray(s.plant_genes)[np.asarray(s.plant)], 0.9)


def test_lineage_share_bounds():
    from agents.evaluate import lineage_share
    from agents.policies import random_policy
    out = lineage_share(CFG, random_policy, random_policy, KEY, 4, uniform_pool(0.5))
    assert set(out) == {"prey", "pred"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_render_evo_frame_and_clip(tmp_path):
    import imageio.v2 as imageio
    from agents.policies import random_policy
    from viz.render import HEADER, render_frame, write_clip
    c = evo.EvoConfig()
    f = render_frame(c, evo.reset(c, KEY), label="evo")
    assert f.shape == (HEADER + 480, 480, 3)
    start = evo.reset(c, KEY, uniform_pool(0.9))
    n = write_clip(c, random_policy, random_policy, KEY, tmp_path / "e.mp4", max_frames=6,
                   world=evo, init_state=start)
    assert n == 6 and imageio.get_reader(tmp_path / "e.mp4").count_frames() == 6


# --- Final-review fixes ------------------------------------------------------------------

def test_predators_start_with_more_energy_in_evo_only():
    from simulations import predprey
    assert evo.EvoConfig().pred_start_energy == 60.0
    assert predprey.WorldConfig().pred_start_energy == 30.0


def test_reset_mixed_tags_are_not_tied_to_low_slots():
    low_only = []
    for k in jax.random.split(KEY, 6):
        s = evo.reset_mixed(CFG, k, uniform_pool(0.9))
        tag = np.asarray(s.prey.tag)
        assert tag[np.asarray(s.prey.alive)].sum() == CFG.init_prey // 2
        assert (tag[~np.asarray(s.prey.alive)] == 0).all()
        low_only.append(bool(tag[:CFG.init_prey // 2].all()))
    assert not all(low_only)


def test_reset_mixed_can_evolve_one_gene_only():
    s = evo.reset_mixed(CFG, KEY, uniform_pool(0.9), genes=(0,))
    alive, tag = np.asarray(s.prey.alive), np.asarray(s.prey.tag)
    evolved = np.asarray(s.prey.genes)[alive & (tag == 1)]
    np.testing.assert_allclose(evolved[:, 0], 0.9)
    assert np.allclose(evolved[:, 1:].mean(0), [0.5, 0.5, 0.5, 0.5, 0.5], atol=0.15)


def test_thinned_plant_pool_keeps_living_genomes():
    pool = uniform_pool(0.5)._replace(plant_genes=jnp.array([[0.1, 0.1], [0.9, 0.9], [0.3, 0.3]]),
                                      plant_w=jnp.array([0.0, 1.0, 0.0]))
    thin = evo.thin_plants(KEY, pool, 16)
    assert thin.plant_genes.shape == (16, 2)
    np.testing.assert_allclose(np.asarray(thin.plant_genes), 0.9)
    empty = evo.thin_plants(KEY, pool._replace(plant_w=jnp.zeros(3)), 16)
    assert float(empty.plant_w.sum()) == 0.0


def test_birth_costs_energy():
    c = cfg(birth_cost=4.0)
    s = evo.make_state(c, prey=[(1, 1, 16.0, 0.0, 0.0, 0.0)])
    p = step(c, s).state.prey
    # 16 - 1 metabolism = 15 >= threshold 15; the birth costs 4, and the remaining 11 is split.
    np.testing.assert_allclose(np.asarray(p.energy[:2]), [5.5, 5.5], atol=1e-5)


def test_lineage_share_is_none_when_a_species_dies_out_early():
    # pp-evo-001 reported predators 0.00, but no predator lived into the scored half at all.
    from agents.evaluate import lineage_share
    from agents.policies import random_policy
    c = evo.EvoConfig(**{**CFG.__dict__, "pred_start_energy": 1.0, "max_steps": 20})
    out = lineage_share(c, random_policy, random_policy, KEY, 4, uniform_pool(0.5))
    assert out["pred"] is None and out["prey"] is not None


def test_evolved_pass_ignores_species_without_data():
    import scripts.eval as ev
    assert ev.evolved_verdict({"prey": 0.7, "pred": None}) is True
    assert ev.evolved_verdict({"prey": 0.5, "pred": None}) is False
    assert ev.evolved_verdict({"prey": None, "pred": None}) is False


def test_a_bitten_plant_must_regrow_before_it_feeds_again():
    # pp-evo-002: tough plants survived every bite, so trained prey ate without limit.
    c = cfg(bite_regrow=0.0)
    s = evo.make_state(c, prey=[(5, 5, 20.0, 0.0, 0.0, 1.0)], plants=[(5, 5, 0.5, 1.0)])
    s = step(c, s).state
    assert bool(s.plant[5, 5]) and bool(s.bitten[5, 5])
    e = float(s.prey.energy[0])
    s = step(c, s).state
    assert abs(float(s.prey.energy[0]) - (e - 1.0)) < 1e-5  # nothing to eat, only metabolism
    prey_obs, _ = evo.observe(c, s)
    assert float(prey_obs[0][4, 4, 0]) == 0.0  # prey see it as not edible
    c2 = cfg(bite_regrow=1.0)
    s = step(c2, s).state
    assert not bool(s.bitten[5, 5])


def test_bitten_plants_do_not_seed():
    c = cfg(seed_rate=1.0, tough_seed_cost=0.0, gene_sigma=0.0, bite_regrow=0.0)
    s = evo.make_state(c, plants=[(5, 5, 1.0, 0.0)])
    s = s._replace(bitten=s.bitten.at[5, 5].set(True))
    out = step(c, s).state
    assert int(out.plant.sum()) == 1
