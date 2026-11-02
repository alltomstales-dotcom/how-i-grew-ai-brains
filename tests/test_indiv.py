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

