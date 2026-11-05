"""Stage 23's scoring helpers (scripts/packs.py)."""
import json

import numpy as np

from scripts import packs


def _write(run, effort, n=200):
    run.mkdir()
    rows = []
    for u in range(1, n + 1):
        rows.append({"update": u, "pred_effort": effort + 0.001 * np.sin(u), "pred_effort_std": 0.05, "birth_blocked": 1.0,
                     "birth_eligible": 100.0, "pred_births": 50.0, "cov_ee": -2.0 + 0.1 * np.sin(u), "var_e": 1.0, "var_energy": 10.0,
                     "breed_dev_sum": -0.5, "pred_gen": u / 5, "hunter_steps": 1000.0, "pred_alive": 20.0, "catches": 30.0})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_matched_effort_uses_the_same_quantiles():
    a = packs.matched_effort(np.linspace(0, 1, 101), 4, 8, 0)
    b = packs.matched_effort(np.linspace(0, 0.5, 101), 4, 8, 0)
    np.testing.assert_allclose(b, a / 2, atol=1e-6)


def test_gate_and_bar1(tmp_path, capsys):
    for name, e in (("E", 0.30), ("P", 0.60), ("N", 0.50)):
        _write(tmp_path / name, e, 600)
    runs = [f"{a}={tmp_path / a}" for a in "EPN"]
    packs.main(["gate", *sum((["--run", r] for r in runs), []), "--out", str(tmp_path / "g.json")])
    g = json.loads((tmp_path / "g.json").read_text())
    assert g["criteria"]["1_free_slots"] and g["criteria"]["2_turnover"]
    assert g["arms"]["E"]["corr"] < 0 and g["arms"]["E"]["breed_dev"] < 0 and g["gate"] == "pass"
    packs.main(["bar1", *sum((["--run", r] for r in runs), []), "--out", str(tmp_path / "b.json")])
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["verdict"] == "pass" and abs(b["diffs"]["P-E"]["diff"] - 0.30) < 0.01
