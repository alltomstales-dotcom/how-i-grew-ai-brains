"""Stage 4 test: two runs compared at matched world-steps."""
import json

import scripts.compare as compare
import scripts.train as train


def test_verdict_passes_on_either_species():
    a = {"prey_lifetime": 11.9, "catches_per_pred_step": 0.010}
    b = {"prey_lifetime": 10.0, "catches_per_pred_step": 0.010}
    assert compare.verdict(a, b)["pass"] is False
    a["prey_lifetime"] = 12.0
    assert compare.verdict(a, b)["pass"] is True
    a = {"prey_lifetime": 10.0, "catches_per_pred_step": 0.013}
    out = compare.verdict(a, b)
    assert out["pass"] is True and abs(out["catch_ratio"] - 1.3) < 1e-9 and abs(out["prey_ratio"] - 1.0) < 1e-9


def test_window_is_last_third_of_the_shorter_run(tmp_path):
    for name, steps in (("a", [10, 20, 30, 40, 50, 60]), ("b", [10, 20, 30, 40, 50, 60, 70, 80, 90])):
        d = tmp_path / name
        (d / "checkpoints").mkdir(parents=True)
        (d / "metrics.jsonl").write_text("".join(json.dumps({"update": i + 1, "env_steps": s}) + "\n"
                                                 for i, s in enumerate(steps)))
        for i in range(len(steps)):
            (d / "checkpoints" / f"ckpt_{i + 1:06d}.msgpack").write_bytes(b"")
    lo, hi = compare.window([tmp_path / "a", tmp_path / "b"])
    assert (lo, hi) == (40, 60)
    picked = compare.checkpoints_in(tmp_path / "b", lo, hi)
    assert [p.name for p in picked] == ["ckpt_000004.msgpack", "ckpt_000005.msgpack", "ckpt_000006.msgpack"]


def test_compare_two_tiny_runs(tmp_path):
    common = ["--tiny", "--world", "seasons", "--runs-dir", str(tmp_path), "--updates", "3", "--ckpt-every", "1"]
    assert train.main(common + ["--run-id", "mem", "--memory", "8"]) == 0
    assert train.main(common + ["--run-id", "ff"]) == 0
    code = compare.main(["--run", str(tmp_path / "mem"), "--baseline", str(tmp_path / "ff"), "--episodes", "2"])
    out = json.loads((tmp_path / "mem" / "compare.json").read_text())
    assert code == (0 if out["pass"] else 1)
    assert out["baseline"].endswith("ff") and out["window"][0] <= out["window"][1]
    assert out["run_checkpoints"] and out["baseline_checkpoints"]
    for side in ("run_eval", "baseline_eval"):
        assert set(out[side]) >= {"prey_lifetime", "catches_per_pred_step"}
