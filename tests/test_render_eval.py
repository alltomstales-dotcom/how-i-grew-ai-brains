import math
from dataclasses import replace

import imageio.v2 as imageio
import jax
import numpy as np

from agents.evaluate import evaluate
from agents.policies import random_policy
from agents.ppo import tiny_configs
from simulations import predprey as pp
from viz.render import HEADER, PRED, PREY, render_frame, write_clip

CFG = pp.WorldConfig()


def test_render_frame_size_and_colours():
    s = pp.make_state(CFG, prey=[(0, 0, 10.0)], pred=[(1, 0, 30.0)])
    f = render_frame(CFG, s, label="test")
    assert f.shape == (HEADER + 480, 480, 3) and f.dtype == np.uint8
    assert tuple(int(c) for c in f[HEADER + 5, 5]) == PREY
    assert tuple(int(c) for c in f[HEADER + 5, 15]) == PRED


def test_write_clip_frame_count(tmp_path):
    path = tmp_path / "c.mp4"
    n = write_clip(CFG, random_policy, random_policy, jax.random.PRNGKey(0), path, max_frames=20)
    assert n == 20
    reader = imageio.get_reader(path)
    assert reader.count_frames() == 20
    assert reader.get_data(0).shape == (HEADER + 480, 480, 3)


def test_clip_stops_early_when_episode_ends(tmp_path):
    # One starving predator dies on the first step, so the episode ends at once.
    cfg = replace(CFG, init_pred=1, init_prey=1, pred_start_energy=1.0)
    n = write_clip(cfg, random_policy, random_policy, jax.random.PRNGKey(0), tmp_path / "e.mp4", max_frames=50)
    assert n == 2


def test_evaluate_is_finite_and_deterministic():
    wcfg, _ = tiny_configs()
    a = evaluate(wcfg, random_policy, random_policy, jax.random.PRNGKey(3), 8)
    b = evaluate(wcfg, random_policy, random_policy, jax.random.PRNGKey(3), 8)
    assert a == b
    assert set(a) == {"catches_per_pred_step", "prey_lifetime", "mean_episode_len"}
    assert all(math.isfinite(v) for v in a.values())
    assert a["prey_lifetime"] > 0
    assert 0 < a["mean_episode_len"] <= wcfg.max_steps


def test_prey_lifetime_keeps_counting_after_predators_die_out():
    # Predators starve on step 1. Prey that outlive them must keep being counted,
    # or prey that starve predators look short-lived.
    wcfg, _ = tiny_configs()
    cfg = replace(wcfg, pred_start_energy=1.0)
    out = evaluate(cfg, random_policy, random_policy, jax.random.PRNGKey(3), 8)
    assert out["prey_lifetime"] > 5.0
    assert out["mean_episode_len"] < 3.0  # the episode itself still ended at extinction
