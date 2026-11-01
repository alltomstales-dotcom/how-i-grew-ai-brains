"""The phone viewer's live feed."""
import base64
import json

import numpy as np


def test_a_training_run_publishes_clips_a_phone_can_read(tmp_path):
    import scripts.train as train
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--updates", "3", "--world", "lasting_stealth", "--run-id", "a", "--live-every", "1", "--live-window", "8"]) == 0
    live = tmp_path / "a" / "live"
    index = json.loads((live / "index.json").read_text())
    assert index["update"] == 3 and len(index["history"]) == 3 and len(index["clips"]) == 3
    clip = json.loads((live / index["latest"]).read_text())
    assert clip["window"] == 8 and clip["steps"] == 48 and {"grazers", "hunters", "grazer_sight"} <= set(clip["stats"])
    p = clip["prey"]
    x = np.frombuffer(base64.b64decode(p["x"]), np.int16).reshape(clip["steps"], p["n"])
    assert x.shape == (48, p["n"])
    on = np.unpackbits(np.frombuffer(base64.b64decode(p["on"]), np.uint8))[: 48 * p["n"]].reshape(48, p["n"])
    assert ((x[on > 0] >= 0) & (x[on > 0] < clip["size"])).all()
    assert len(np.frombuffer(base64.b64decode(p["genes"]), np.uint8)) == p["n"] * 4
    assert not list(live.glob(".*.tmp"))


def test_old_clips_are_removed(tmp_path):
    import scripts.live as live
    import scripts.train as train
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--updates", str(live.KEEP + 3), "--world", "lasting_stealth", "--run-id", "a", "--live-every", "1", "--live-window", "8"]) == 0
    assert len(list((tmp_path / "a" / "live").glob("clip_*.json"))) == live.KEEP


def test_no_clips_trains_without_rendering(tmp_path):
    import scripts.train as train
    common = ["--tiny", "--runs-dir", str(tmp_path), "--updates", "2"]
    assert train.main([*common, "--world", "ambush", "--run-id", "n1", "--no-clips"]) == 0
    assert (tmp_path / "n1" / "metrics.jsonl").stat().st_size > 0
    assert not list((tmp_path / "n1" / "clips").glob("*.mp4"))

