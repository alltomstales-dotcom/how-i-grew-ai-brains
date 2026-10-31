"""Stage 3b: Elo ratings for a run's saved brains."""
import json

import numpy as np

import scripts.league as league
import scripts.train as train
from simulations import evo


def test_fit_elo_recovers_rating_gaps():
    pred_true = np.array([1000.0, 1100.0, 1250.0])
    prey_true = np.array([1050.0, 1150.0])
    scores = 1 / (1 + 10 ** ((prey_true[None, :] - pred_true[:, None]) / 400))
    pred, prey = league.fit_elo(scores)
    np.testing.assert_allclose(np.diff(pred), np.diff(pred_true), atol=5)
    np.testing.assert_allclose(prey[1] - prey[0], 100, atol=5)
    np.testing.assert_allclose(pred[0] - prey[0], pred_true[0] - prey_true[0], atol=5)
    assert abs(np.concatenate([pred, prey]).mean() - 1000) < 1e-6


def test_fit_elo_survives_certain_outcomes():
    pred, prey = league.fit_elo(np.array([[0.0, 1.0], [1.0, 1.0]]))
    assert np.isfinite(pred).all() and np.isfinite(prey).all()


def test_progress_verdict():
    assert league.progress_verdict(np.array([900.0, 1000.0, 1050.0])) is True
    assert league.progress_verdict(np.array([1000.0, 1100.0, 1050.0])) is False  # +50 only
    assert league.progress_verdict(np.array([1000.0])) is False


def test_league_on_tiny_run(tmp_path):
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "l1",
                       "--updates", "3", "--ckpt-every", "1"]) == 0
    assert league.main(["--run-dir", str(tmp_path / "l1"), "--episodes", "2", "--max-brains", "3"]) == 0
    out = json.loads((tmp_path / "l1" / "league.json").read_text())
    assert out["updates"] == [1, 2, 3]
    assert len(out["pred_elo"]) == len(out["prey_elo"]) == 3
    assert np.asarray(out["scores"]).shape == (3, 3)
    assert isinstance(out["pred_progress"], bool) and isinstance(out["prey_progress"], bool)


def test_match_pool_gives_each_side_the_bodies_it_trained_with():
    prey_era = {"prey_genes": np.full((3, 3), 0.9), "prey_gen": np.ones(3), "pred_genes": np.zeros((2, 3)),
                "pred_gen": np.zeros(2), "plant_genes": np.full((5, 2), 0.7)}
    pred_era = {"prey_genes": np.zeros((4, 3)), "prey_gen": np.zeros(4), "pred_genes": np.full((6, 3), 0.1),
                "pred_gen": np.ones(6), "plant_genes": np.zeros((1, 2))}
    pool = league.match_pool(prey_era, pred_era)
    assert pool.prey_genes.shape == pool.pred_genes.shape == (league.POOL_ROWS, evo.ANIMAL_GENES)  # size and brain padded in
    w = np.asarray(pool.prey_w); assert w.sum() == 3 and np.allclose(np.asarray(pool.prey_genes)[w > 0, :3], 0.9)
    w = np.asarray(pool.pred_w); assert w.sum() == 6 and np.allclose(np.asarray(pool.pred_genes)[w > 0, :3], 0.1)
    w = np.asarray(pool.plant_w); assert w.sum() == 5 and np.allclose(np.asarray(pool.plant_genes)[w > 0], 0.7)


def test_league_on_tiny_evo_run_uses_evolved_bodies(tmp_path):
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "l2",
                       "--updates", "2", "--ckpt-every", "1"]) == 0
    assert league.main(["--run-dir", str(tmp_path / "l2"), "--episodes", "2"]) == 0
    assert json.loads((tmp_path / "l2" / "league.json").read_text())["bodies"] == "evolved"
