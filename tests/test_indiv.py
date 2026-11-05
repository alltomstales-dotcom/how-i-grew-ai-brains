import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents import indiv
from agents.networks import init_params
from simulations import evo
from simulations.worlds import get_world, make_net, obs_channels


def test_the_small_brain_is_small_and_the_world_reports_parents():
    world, cls = get_world("tiny_stealth")
    c = cls()
    p = init_params(make_net(world, c, 0), jax.random.PRNGKey(0), c.view, obs_channels(world, c))
    assert sum(x.size for x in jax.tree.leaves(p)) < 6000
    t = evo.EvoConfig(size=12, max_prey=4, max_pred=2, init_prey=2, init_pred=1, prey_repro_energy=10.0, breed_span=0.0, gene_sigma=0.0)
    s = evo.make_state(t, prey=[(2, 2, 30.0, 0.0, 0.0, 0.0, 0.5)])
    res = evo.advance(t, jax.random.PRNGKey(0), s, jnp.zeros(4, jnp.int32), jnp.zeros(2, jnp.int32))
    assert np.asarray(res.info.prey_parent).tolist() == [0, 0, 2, 3]  # slot 1 holds slot 0's child
    assert np.asarray(res.info.pred_parent).tolist() == [0, 1]


def small_pop():
    params = {"w": jnp.arange(4.0)[None, :, None] * jnp.ones((1, 4, 3))}
    birth = {"w": jnp.zeros((1, 4, 3))}
    return indiv.Pop(params, birth)


def test_a_lamarckian_child_gets_what_was_learned_and_a_baldwinian_child_the_birth_brain():
    parent = jnp.array([[0, 2, 2, 3]])  # slot 1 holds slot 2's child
    lam = indiv.inherit(small_pop(), parent, jax.random.PRNGKey(0), indiv.IndivConfig(mode="lamarck", sigma=0.0))
    assert np.allclose(lam.params["w"][0, 1], 2.0) and np.allclose(lam.params["w"][0, 0], 0.0) and np.allclose(lam.params["w"][0, 3], 3.0)
    assert np.allclose(lam.birth["w"][0, 1], 2.0)  # what it starts life with is its birth brain
    bal = indiv.inherit(small_pop(), parent, jax.random.PRNGKey(0), indiv.IndivConfig(mode="baldwin", sigma=0.0))
    assert np.allclose(bal.params["w"][0, 1], 0.0) and np.allclose(bal.params["w"][0, 2], 2.0)
    noisy = indiv.inherit(small_pop(), parent, jax.random.PRNGKey(0), indiv.IndivConfig(mode="lamarck", sigma=0.5))
    assert not np.allclose(noisy.params["w"][0, 1], 2.0) and np.allclose(noisy.params["w"][0, 2], 2.0)  # only children mutate


def test_learning_never_moves_a_brain_more_than_the_clip_allows(tmp_path):
    import scripts.train_indiv as train
    assert train.main(["--tiny", "--world", "tiny_stealth", "--mode", "lamarck", "--run-id", "c", "--runs-dir", str(tmp_path), "--updates", "4", "--ckpt-every", "2", "--num-envs", "2", "--lr", "0.5"]) == 0
    rows = [json.loads(l) for l in open(tmp_path / "c" / "metrics.jsonl")]
    assert all(np.isfinite(r["prey_weight_moved"]) and r["prey_weight_moved"] < 1.0 for r in rows)


def test_a_heritable_learning_rate_can_switch_learning_off(tmp_path):
    from dataclasses import replace
    import jax.numpy as jnp
    world, cls = get_world("tiny_stealth")
    c = replace(cls(), size=12, max_prey=4, max_pred=2, init_prey=2, init_pred=1, founder_animal=(0.2, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5), founder_noise=0.0, gene_sigma=0.0)
    net = make_net(world, c, 0)
    p = init_params(net, jax.random.PRNGKey(0), c.view, obs_channels(world, c))
    moved = {}
    for gene_lr in (False, True):
        cfg = indiv.IndivConfig(mode="lamarck", lr=0.02, sigma=0.0, gene_lr=gene_lr)
        s = jax.vmap(lambda k: world.reset(c, k))(jax.random.split(jax.random.PRNGKey(1), 2))
        po, qo = jax.vmap(lambda x: world.observe(c, x))(s)
        carry = indiv.Carry(s, indiv.tile(p, 2, 4, jax.random.PRNGKey(2), 0.0), indiv.tile(p, 2, 2, jax.random.PRNGKey(3), 0.0), po, qo, jax.random.PRNGKey(4))
        carry, _ = jax.lax.scan(indiv.make_step(net, world, c, cfg), carry, None, length=3)
        moved[gene_lr] = float(indiv.spread(carry.prey, carry.state.prey.alive)["weight_moved"])
    assert moved[False] > 0 and moved[True] == 0.0  # a gene of 0 learns nothing


def test_a_new_world_draws_brains_from_the_living():
    pop = indiv.Pop({"w": jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])}, {"w": jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])})
    alive = jnp.array([[False, False], [True, False]])  # only world 1, slot 0 lives, with 3.0
    out = indiv.redraw(pop, alive, jnp.array([True, False]), jax.random.PRNGKey(0), indiv.IndivConfig())
    assert np.allclose(out.params["w"][0], 3.0) and np.allclose(out.params["w"][1], [[3.0], [4.0]])


@pytest.mark.parametrize("mode,moves", [("darwin", False), ("baldwin", True), ("lamarck", True)])
def test_only_learners_change_their_weights_in_life(tmp_path, mode, moves):
    import scripts.train_indiv as train
    import scripts.eval_indiv as ev
    run = f"r{mode}"
    assert train.main(["--tiny", "--world", "tiny_stealth", "--mode", mode, "--run-id", run, "--runs-dir", str(tmp_path), "--updates", "3", "--ckpt-every", "1", "--num-envs", "2"]) == 0
    rows = [json.loads(l) for l in open(tmp_path / run / "metrics.jsonl")]
    assert len(rows) == 3 and "pred_weight_spread" in rows[0]
    assert (rows[-1]["pred_weight_moved"] + rows[-1]["prey_weight_moved"] > 0) == moves
    assert ev.main(["--run-dir", str(tmp_path / run), "--episodes", "2", "--steps", "100"]) == 0
    out = json.loads((tmp_path / run / "eval_indiv.json").read_text())
    assert out["mode"] == mode and 0.0 <= out["learned"]["both_at_end"] <= 1.0


def test_hunters_and_grazers_can_come_from_different_runs(tmp_path):
    import scripts.cross_indiv as cross
    import scripts.train as shared
    import scripts.train_indiv as train
    for run, mode in (("a", "darwin"), ("b", "lamarck")):
        assert train.main(["--tiny", "--world", "tiny_stealth", "--mode", mode, "--run-id", run, "--runs-dir", str(tmp_path), "--updates", "2", "--ckpt-every", "1", "--num-envs", "2"]) == 0
    assert cross.main(["--hunters", str(tmp_path / "b"), "--grazers", str(tmp_path / "a"), "--episodes", "2", "--steps", "100"]) == 0
    out = json.loads((tmp_path / "a" / "cross_b.json").read_text())
    assert out["hunters"] == "b" and out["grazers"] == "a" and out["catches_per_hunter_step"] >= 0
    assert shared.main(["--tiny", "--runs-dir", str(tmp_path), "--updates", "2", "--world", "tiny_stealth", "--no-clips", "--ckpt-every", "1", "--run-id", "f"]) == 0
    assert cross.main(["--hunters", str(tmp_path / "f"), "--grazers", str(tmp_path / "a"), "--founder", "--episodes", "2", "--steps", "100"]) == 0



def culture_world():
    from dataclasses import replace
    world, cls = get_world("tiny_stealth")
    c = replace(cls(), size=12, max_prey=6, max_pred=4, init_prey=3, init_pred=2)
    net = make_net(world, c, 0)
    blank = indiv.blank_brain(net, c.view, obs_channels(world, c))
    return world, c, net, blank


def test_a_child_of_a_culture_arm_gets_the_blank_brain_and_its_parent_as_teacher():
    pop = small_pop()
    kin = indiv.fresh_kin(1, 4, 8)
    parent = jnp.array([[0, 2, 2, 3]])
    blank = {"w": 9.0 * jnp.ones(3)}
    done = jnp.zeros((1, 4), bool)
    new, kin = indiv.culture_birth(pop, kin, parent, done, jax.random.PRNGKey(0), indiv.IndivConfig(mode="teach", sigma=0.0, childhood=8), blank)
    assert np.allclose(new.params["w"][0, 1], 9.0) and np.allclose(new.params["w"][0, 2], 2.0)  # the child is blank, the parent unchanged
    assert int(kin.teacher[0, 1]) == 2 and int(kin.age[0, 1]) == 0 and int(kin.teacher[0, 2]) == -1
    kin = kin._replace(age=jnp.zeros((1, 4), jnp.int32))
    died = jnp.array([[False, False, True, False]])
    same = jnp.arange(4)[None, :]
    _, kin = indiv.culture_birth(new, kin, same, died, jax.random.PRNGKey(0), indiv.IndivConfig(mode="teach", sigma=0.0, childhood=8), blank)
    assert int(kin.teacher[0, 1]) == -1  # its teacher died


@pytest.mark.parametrize("mode", ["watch", "teach"])
def test_lessons_need_a_near_living_parent_and_stop_after_childhood(mode):
    world, c, net, blank = culture_world()
    s = world.reset(c, jax.random.PRNGKey(1))
    sp = jax.tree.map(lambda x: x[None], s.prey)
    n = sp.alive.shape[1]
    sp = sp._replace(alive=jnp.ones((1, n), bool), pos=jnp.zeros((1, n, 2), jnp.int32))
    obs = jax.vmap(lambda x: world.observe(c, x))(jax.tree.map(lambda x: x[None], s))[0]
    action = jnp.ones((1, n), jnp.int32)
    tree = jax.tree.map(lambda x: jnp.broadcast_to(x, (1, n) + x.shape), blank)
    keys = jax.random.split(jax.random.PRNGKey(2), len(jax.tree.leaves(blank)))
    other = jax.tree.map(lambda x: x, tree)
    pop = indiv.Pop(indiv._noisy(other, keys[0], 0.1), tree)
    teacher = jnp.full((1, n), -1, jnp.int32).at[0, 1].set(0).at[0, 2].set(0).at[0, 3].set(0)
    age = jnp.zeros((1, n), jnp.int32).at[0, 2].set(8)  # slot 2 has grown up
    kin = indiv.fresh_kin(1, n, 8)._replace(teacher=teacher, age=age)
    sp = sp._replace(pos=sp.pos.at[0, 3].set(jnp.array([6, 6])))  # slot 3 is out of sight when view is small
    cfg = indiv.IndivConfig(mode=mode, lr=0.02, childhood=8, teach_cost=0.5)
    lesson = indiv.make_lesson(net, 1, c.size)
    new, kin2, given = lesson(pop, kin, obs, action, sp, jnp.zeros(1, bool), cfg)
    moved = [not np.allclose(a[0, i], b[0, i]) for i in range(4) for a, b in [(jax.tree.leaves(new.params)[0], jax.tree.leaves(pop.params)[0])]]
    assert moved == [False, True, False, False]
    assert float(kin2.lessons) == 1.0 and float(kin2.child_steps) == 2.0 and float(given[0, 0]) == 1.0 and float(given.sum()) == 1.0


@pytest.mark.parametrize("mode,childhood,cost", [("blank", 0, 0.0), ("watch", 8, 0.0), ("teach", 8, 0.05)])
def test_culture_arms_train_and_can_be_tested(tmp_path, mode, childhood, cost):
    import scripts.train_indiv as train
    import scripts.cross_indiv as cross
    common = ["--tiny", "--world", "tiny_stealth", "--runs-dir", str(tmp_path), "--updates", "2", "--ckpt-every", "2"]
    assert train.main(common + ["--mode", mode, "--run-id", "a", "--childhood", str(childhood), "--teach-cost", str(cost)]) == 0
    rows = [json.loads(l) for l in open(tmp_path / "a" / "metrics.jsonl")]
    assert "pred_lessons" in rows[-1]
    assert train.main(common + ["--mode", "darwin", "--run-id", "b"]) == 0
    assert cross.main(["--hunters", str(tmp_path / "a"), "--grazers", str(tmp_path / "b"), "--episodes", "2", "--steps", "100"]) == 0
    assert "both_alive" in json.load(open(tmp_path / "b" / "cross_a.json"))
    assert cross.main(["--hunters", str(tmp_path / "b"), "--grazers", str(tmp_path / "b"), "--episodes", "2", "--steps", "100", "--random"]) == 0
    assert cross.main(["--hunters", str(tmp_path / "a"), "--grazers", str(tmp_path / "a"), "--episodes", "2", "--steps", "100", "--both-live", "--out", "own.json"]) == 0


def test_with_innate_a_culture_child_is_born_with_its_parents_birth_brain():
    pop = indiv.Pop({"w": jnp.arange(4.0)[None, :, None] * jnp.ones((1, 4, 3))}, {"w": 10.0 + jnp.arange(4.0)[None, :, None] * jnp.ones((1, 4, 3))})
    parent = jnp.array([[0, 2, 2, 3]])
    new, kin = indiv.culture_birth(pop, indiv.fresh_kin(1, 4, 8), parent, jnp.zeros((1, 4), bool), jax.random.PRNGKey(0),
                                   indiv.IndivConfig(mode="watch", sigma=0.0, childhood=8, innate=True), {"w": 9.0 * jnp.ones(3)})
    assert np.allclose(new.params["w"][0, 1], 12.0) and np.allclose(new.birth["w"][0, 1], 12.0)  # the parent's birth brain, not what it learned
    assert int(kin.teacher[0, 1]) == 2


def test_subspace_mutation_moves_brains_only_along_its_directions():
    pop = indiv.Pop({"a": jnp.zeros((1, 6, 5)), "b": jnp.zeros((1, 6, 3))}, {"a": jnp.zeros((1, 6, 5)), "b": jnp.zeros((1, 6, 3))})
    parent = jnp.array([[0, 0, 0, 0, 0, 0]])  # five children of slot 0
    new = indiv.inherit(pop, parent, jax.random.PRNGKey(3), indiv.IndivConfig(mode="darwin", sigma=1.0, subspace=2))
    flat = np.concatenate([np.asarray(new.params["a"][0]), np.asarray(new.params["b"][0])], axis=1)  # (6, 8)
    assert np.allclose(flat[0], 0.0) and not np.allclose(flat[1], 0.0)  # only children mutate
    assert np.linalg.matrix_rank(flat[1:], tol=1e-4) <= 2  # every child lies in the same 2-direction plane
    again = indiv.inherit(pop, parent, jax.random.PRNGKey(4), indiv.IndivConfig(mode="darwin", sigma=1.0, subspace=2))
    both = np.concatenate([flat[1:], np.concatenate([np.asarray(again.params["a"][0]), np.asarray(again.params["b"][0])], axis=1)[1:]])
    assert np.linalg.matrix_rank(both, tol=1e-4) <= 2  # the directions are fixed, not redrawn


def anchor_world():
    from dataclasses import replace
    world, cls = get_world("tiny_stealth")
    c = replace(cls(), size=12, max_prey=8, max_pred=4, init_prey=3, init_pred=2, prey_start_energy=35.0, prey_repro_energy=10.0, breed_span=0.0, max_steps=50)
    net = make_net(world, c, 0)
    brain = lambda k: init_params(net, jax.random.PRNGKey(k), c.view, obs_channels(world, c))
    return world, c, net, brain


def run_anchored(frac, pool, steps=6):
    world, c, net, brain = anchor_world()
    s = jax.vmap(lambda k: world.reset(c, k))(jax.random.split(jax.random.PRNGKey(1), 2))
    po, qo = jax.vmap(lambda x: world.observe(c, x))(s)
    carry = indiv.Carry(s, indiv.tile(brain(10), 2, 8, jax.random.PRNGKey(2), 0.1), indiv.tile(brain(11), 2, 4, jax.random.PRNGKey(3), 0.1), po, qo, jax.random.PRNGKey(4))
    step = indiv.make_step(net, world, c, indiv.IndivConfig(mode="darwin", sigma=0.03, anchor_frac=frac), anchor_pool=pool)
    return carry, jax.lax.scan(step, carry, None, length=steps)[0]


def test_anchored_newborn_grazers_get_pool_brains_and_unanchored_worlds_run_as_before():
    _, _, _, brain = anchor_world()
    pool = jax.tree.map(lambda *x: jnp.stack(x), *[brain(k) for k in (20, 21, 22)])
    start, plain = run_anchored(0.0, None)
    _, anchored = run_anchored(0.5, pool)  # world 0 is anchored, world 1 is not
    hit = np.asarray(indiv.from_pool(anchored.prey.params, pool))
    born = np.asarray(anchored.state.prey.alive & ~start.state.prey.alive)
    assert born[0].sum() >= 2 and born[1].sum() >= 2  # grazers were born in both worlds
    assert hit[0][born[0]].all()  # newborns in the anchored world carry pool brains, not their parent's
    same = np.all([np.asarray(a[0] == b[0]).reshape(8, -1).all(-1) for a, b in zip(jax.tree.leaves(anchored.prey.params), jax.tree.leaves(start.prey.params))], 0)
    alive = np.asarray(anchored.state.prey.alive[0])
    assert (hit[0] | same)[alive].all() and not hit[0][same].any()  # every grazer there is an original, or carries a pool brain
    assert not hit[1].any()  # no pool brain in the unanchored world
    for a, b in zip(jax.tree.leaves(anchored.prey.params), jax.tree.leaves(plain.prey.params)):
        assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))  # the unanchored world ran exactly as with no anchoring
    for a, b in zip(jax.tree.leaves(anchored.pred.params), jax.tree.leaves(plain.pred.params)):
        assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_an_unanchored_world_that_restarts_never_draws_an_anchored_grazer():
    pop = indiv.Pop({"w": jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])}, {"w": jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])})
    alive = jnp.array([[True, True], [False, True]])
    out = indiv.redraw(pop, alive, jnp.array([False, True]), jax.random.PRNGKey(0), indiv.IndivConfig(), allowed=jnp.array([False, True]))
    assert np.allclose(out.params["w"][1], 4.0) and np.allclose(out.params["w"][0], [[1.0], [2.0]])
    none = jnp.array([[True, True], [False, False]])  # nobody alive outside the anchored world: draw from the unanchored world's slots
    out = indiv.redraw(pop, none, jnp.array([False, True]), jax.random.PRNGKey(0), indiv.IndivConfig(), allowed=jnp.array([False, True]))
    assert set(np.asarray(out.params["w"][1]).ravel().tolist()) <= {3.0, 4.0}


def test_anchored_training_runs_and_logs_pool_brains_by_world(tmp_path):
    import scripts.train_indiv as train
    import scripts.cross_indiv as cross
    common = ["--tiny", "--world", "tiny_stealth", "--runs-dir", str(tmp_path), "--mode", "darwin", "--updates", "2", "--ckpt-every", "1"]
    assert train.main(common + ["--run-id", "pool"]) == 0
    assert train.main(common + ["--run-id", "half", "--anchor-frac", "0.5", "--anchor-pool", str(tmp_path / "pool"), "--sigma", "0.03", "--seed", "1"]) == 0
    rows = [json.loads(l) for l in open(tmp_path / "half" / "metrics.jsonl")]
    assert all(r["anch_prey_from_pool"] == 1.0 and r["free_prey_from_pool"] == 0.0 for r in rows)
    assert "anch_pred_alive" in rows[-1] and "free_catches_per_step" in rows[-1]
    assert cross.main(["--hunters", str(tmp_path / "half"), "--grazers", str(tmp_path / "pool"), "--episodes", "2", "--steps", "100"]) == 0
    with pytest.raises(SystemExit):
        train.main(common + ["--run-id", "bad", "--anchor-frac", "0.5"])


def test_world_set_reads_each_field_as_its_type_and_refuses_unknown_fields():
    from simulations.worlds import world_set
    _, cls = get_world("tiny_stealth")
    c = world_set(cls(), ["pred_base_cost=-0.02", "gene_sigma=0", "max_pred=16", "sizes=true", "founder_plant=[0.4, 0.1]"])
    assert c.pred_base_cost == -0.02 and c.gene_sigma == 0.0 and isinstance(c.gene_sigma, float)
    assert c.max_pred == 16 and c.sizes is True and c.founder_plant == (0.4, 0.1)
    assert world_set(cls(), []) == cls()
    for bad in (["no_such_field=1"], ["gene_sigma"], ["sizes=maybe"]):
        with pytest.raises(ValueError):
            world_set(cls(), bad)


def test_hunter_upkeep_is_what_a_hunter_that_does_not_eat_loses_each_step():
    t = evo.EvoConfig(size=12, max_prey=2, max_pred=2, init_prey=0, init_pred=1, gene_sigma=0.0, pred_base_cost=0.3)
    s = evo.make_state(t, prey=[], pred=[(2, 2, 30.0, 0.4, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5)])
    res = evo.advance(t, jax.random.PRNGKey(0), s, jnp.zeros(2, jnp.int32), jnp.zeros(2, jnp.int32))
    lost = 30.0 - float(res.state.pred.energy[0])
    assert np.isclose(lost, float(evo.hunter_upkeep(t, s)[0]), atol=1e-5)
    assert np.isclose(lost, 0.3 + 0.4 ** 2 + 0.5 * 3 / 3, atol=1e-5)  # base, speed squared, sight radius 4


def test_frozen_bodies_and_a_changed_upkeep_reach_training_and_the_test(tmp_path):
    import scripts.cross_indiv as cross
    import scripts.train_indiv as train
    frozen = ["--world-set", "gene_sigma=0", "--world-set", "founder_noise=0", "--world-set", "plant_sigma=0.05"]
    common = ["--tiny", "--world", "tiny_stealth", "--runs-dir", str(tmp_path), "--mode", "darwin", "--updates", "3", "--ckpt-every", "1", "--sigma", "0.03"]
    assert train.main(common + ["--run-id", "pool"] + frozen) == 0
    assert train.main(common + ["--run-id", "cheap", "--anchor-frac", "1.0", "--anchor-pool", str(tmp_path / "pool"), "--world-set", "pred_base_cost=0.1"] + frozen) == 0
    world = json.loads((tmp_path / "cheap" / "config.json").read_text())["world"]
    assert world["pred_base_cost"] == 0.1 and world["gene_sigma"] == 0.0 and world["founder_noise"] == 0.0
    with np.load(tmp_path / "cheap" / "checkpoints" / "genes_latest.npz") as d:
        genes = {k: d[k] for k in d.files}
    _, cls = get_world("tiny_stealth")
    founder = np.asarray(cls().founder_animal, np.float32)
    assert len(genes["prey_genes"]) + len(genes["pred_genes"]) > 0
    for kind in ("prey", "pred"):
        assert np.allclose(genes[f"{kind}_genes"], founder)  # every living body is the founder body
    for extra in ([], ["--world-set", "pred_base_cost=0.1", "--eval-seed", "7", "--bodies", str(tmp_path / "pool"), "--out", "b.json"]):
        assert cross.main(["--hunters", str(tmp_path / "cheap"), "--grazers", str(tmp_path / "pool"), "--episodes", "2", "--steps", "100"] + extra) == 0
    a = json.loads((tmp_path / "pool" / "cross_cheap.json").read_text())
    b = json.loads((tmp_path / "pool" / "b.json").read_text())
    assert b["eval_seed"] == 7 and b["bodies"] == str(tmp_path / "pool") and a["catches"] >= 0 and a["hunter_steps"] > 0
    assert np.isclose(a["upkeep_per_hunter_step"] - b["upkeep_per_hunter_step"], cls().pred_base_cost - 0.1, atol=1e-4)  # founder bodies both: only the base differs
    assert np.isclose(a["break_even_catches_per_hunter_step"], a["upkeep_per_hunter_step"] / cls().catch_energy)
    with pytest.raises(SystemExit):
        train.main(common + ["--run-id", "typo", "--world-set", "pred_base_cots=0.1"])
