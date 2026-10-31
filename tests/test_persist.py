"""Stage 10a: worlds that last."""
import json


def test_persist_runs_on_a_tiny_run(tmp_path):
    import scripts.persist as persist
    import scripts.train as train
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--updates", "2", "--world", "fatprey_fixed", "--run-id", "a"]) == 0
    assert persist.main(["--run-dir", str(tmp_path / "a"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "a" / "persist.json").read_text())
    assert isinstance(out["pass"], bool) and out["curve"][0]["step"] == 0
    assert persist.main(["--run-dir", str(tmp_path / "a"), "--episodes", "2", "--bot-grazers", "--seed-rate", "0.1"]) == 0
    assert persist.main(["--run-dir", str(tmp_path / "a"), "--episodes", "2", "--steps", "120"]) == 0
    assert json.loads((tmp_path / "a" / "persist.json").read_text())["curve"][-1]["step"] == 119


def test_the_sweep_runs_on_a_tiny_world(tmp_path):
    import scripts.lasting as lasting
    assert lasting.main(["--tiny", "--steps", "30", "--episodes", "2", "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "lasting.json").read_text())
    assert len(out) == 1 and isinstance(out[0]["pass"], bool)
