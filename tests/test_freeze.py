"""Stage 20b: freezing one species' bodies, kept checkpoints, and testing a kept checkpoint."""
import json

import numpy as np
import pytest

from simulations.worlds import get_world

COMMON = ["--tiny", "--world", "tiny_stealth", "--mode", "darwin", "--sigma", "0.03", "--ckpt-every", "1",
          "--world-set", "founder_noise=0", "--world-set", "plant_sigma=0.05", "--world-set", "gene_sigma=0.2"]


def genes(run, name="genes_latest.npz"):
    with np.load(run / "checkpoints" / name) as d:
        return {k: d[k] for k in d.files}


@pytest.mark.parametrize("which", ["prey", "pred"])
def test_only_the_frozen_species_keeps_the_founder_body(tmp_path, which):
    import scripts.train_indiv as train
    assert train.main(COMMON + ["--runs-dir", str(tmp_path), "--run-id", which, "--updates", "12", "--freeze-bodies", which, "--seed", "3"]) == 0
    g = genes(tmp_path / which)
    founder = np.asarray(get_world("tiny_stealth")[1]().founder_animal, np.float32)
    other = "pred" if which == "prey" else "prey"
    assert len(g[f"{which}_genes"]) > 0 and np.allclose(g[f"{which}_genes"], founder)
    assert len(g[f"{other}_genes"]) > 0 and not np.allclose(g[f"{other}_genes"], founder)  # the other species' bodies evolved


def test_freezing_both_gives_the_same_worlds_as_no_gene_mutation(tmp_path):
    import scripts.train_indiv as train
    base = COMMON + ["--runs-dir", str(tmp_path), "--updates", "6", "--seed", "1"]
    assert train.main(base + ["--run-id", "clamped", "--freeze-bodies", "both"]) == 0
    assert train.main(base + ["--run-id", "sigma0", "--world-set", "gene_sigma=0"]) == 0
    rows = [[json.loads(x) for x in (tmp_path / r / "metrics.jsonl").read_text().splitlines()] for r in ("clamped", "sigma0")]
    for a, b in zip(*rows):
        for k in ("catches_per_step", "pred_alive", "prey_alive", "pred_births_per_step"):
            assert np.isclose(a[k], b[k]), (k, a[k], b[k])
    for name in ("pop_latest.npz", "genes_latest.npz"):
        x, y = genes(tmp_path / "clamped", name), genes(tmp_path / "sigma0", name)
        assert x.keys() == y.keys() and all(np.allclose(x[k], y[k]) for k in x)


def test_kept_checkpoints_can_be_tested(tmp_path):
    import scripts.cross_indiv as cross
    import scripts.train_indiv as train
    assert train.main(COMMON + ["--runs-dir", str(tmp_path), "--run-id", "kept", "--updates", "4", "--keep-at", "2,4", "--freeze-bodies", "pred"]) == 0
    ck = tmp_path / "kept" / "checkpoints"
    for u in ("0002", "0004"):
        assert (ck / f"pop_u{u}.npz").exists() and (ck / f"genes_u{u}.npz").exists()
    assert not (ck / "pop_u0001.npz").exists()
    assert json.loads((tmp_path / "kept" / "config.json").read_text())["args"]["freeze_bodies"] == "pred"
    assert cross.main(["--hunters", str(tmp_path / "kept"), "--grazers", str(tmp_path / "kept"), "--episodes", "2", "--steps", "100",
                       "--hunter-ckpt", "pop_u0002.npz", "--out", "k.json"]) == 0
    assert json.loads((tmp_path / "kept" / "k.json").read_text())["hunter_ckpt"] == "pop_u0002.npz"
