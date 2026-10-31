"""A hand-written chase rule, to tell a ceiling from a learning limit in the predator stall."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from agents.policies import chase_policy


def view_with_prey(dx, dy, view=9, channels=11, prey_channel=2):
    obs = np.zeros((view, view, channels), np.float32)
    if dx is not None:
        obs[view // 2 + dy, view // 2 + dx, prey_channel] = 1.0
    return jnp.asarray(obs)


def test_chaser_steps_toward_the_nearest_prey():
    pi = chase_policy(prey_channel=2)
    key = jax.random.PRNGKey(0)
    # Actions: 1 = (0, -1) north, 2 = (0, 1) south, 3 = (1, 0) east, 4 = (-1, 0) west.
    for (dx, dy), want in (((3, 0), 3), ((-2, 0), 4), ((0, -4), 1), ((0, 2), 2), ((3, -1), 3), ((1, -3), 1)):
        assert int(pi(key, view_with_prey(dx, dy)[None])[0]) == want, (dx, dy)
    two = view_with_prey(4, 0).at[4, 3, 2].set(1.0)  # a second prey 1 cell west is nearer
    assert int(pi(key, two[None])[0]) == 4
    a = pi(key, jnp.stack([view_with_prey(None, None)] * 50))
    assert ((a >= 0) & (a < 5)).all() and len(set(np.asarray(a).tolist())) > 1  # nothing seen: wander


def test_chase_script_on_tiny_run(tmp_path):
    import scripts.chase as chase
    import scripts.train as train
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "c1", "--updates", "2"]) == 0
    assert chase.main(["--run-dir", str(tmp_path / "c1"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "c1" / "chase.json").read_text())
    for k in ("chase_vs_random_prey", "trained_vs_random_prey", "chase_vs_trained_prey", "trained_vs_trained_prey"):
        assert out[k] >= 0
    assert "ratio_vs_random_prey" in out and "ratio_vs_trained_prey" in out


def test_evaluate_can_start_from_evolved_bodies():
    from agents.evaluate import evaluate
    from agents.policies import random_policy
    from simulations import evo
    from scripts.league import match_pool
    cfg = evo.EvoConfig(size=12, max_prey=8, max_pred=4, init_prey=6, init_pred=3, max_steps=40)
    chaser = chase_policy(evo.PREY_CHANNEL)

    def era(pred_speed, pred_sight):
        return {"prey_genes": np.tile([0.2, 0.5, 0.5, 0.5], (50, 1)), "prey_gen": np.zeros(50, np.int32),
                "pred_genes": np.tile([pred_speed, pred_sight, 0.5, 0.5], (50, 1)), "pred_gen": np.zeros(50, np.int32),
                "plant_genes": np.tile([0.5, 0.2], (50, 1))}

    key = jax.random.PRNGKey(0)
    run = lambda e: evaluate(cfg, random_policy, chaser, key, 64, evo, pool=match_pool(e, e))["catches_per_pred_step"]
    assert run(era(1.0, 1.0)) > 1.5 * run(era(0.0, 0.0))


def test_chase_script_with_evolved_bodies(tmp_path):
    import scripts.chase as chase
    import scripts.train as train
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "c2", "--updates", "2",
                       "--ckpt-every", "1"]) == 0
    assert chase.main(["--run-dir", str(tmp_path / "c2"), "--episodes", "2", "--evolved"]) == 0
    out = json.loads((tmp_path / "c2" / "chase_evolved.json").read_text())
    assert out["bodies"].startswith("genes_") and "ratio_vs_trained_prey" in out
