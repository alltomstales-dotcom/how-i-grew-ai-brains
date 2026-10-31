"""The 3-D viewer's episode recording."""
import base64
import json

import numpy as np


def test_episode3d_records_a_tiny_run(tmp_path):
    import scripts.episode3d as ep
    import scripts.train as train
    assert train.main(["--tiny", "--world", "flank_tax", "--runs-dir", str(tmp_path), "--run-id", "e1", "--updates", "2"]) == 0
    for extra in ([], ["--evolved"]):
        assert ep.main(["--run-dir", str(tmp_path / "e1"), *extra]) == 0
        out = json.loads((tmp_path / "e1" / "episode3d.json").read_text())
        size, steps, animals = out["size"], out["steps"], out["animals"]
        assert out["world"] == "flank_tax" and out["flank"] and steps
        cells = np.unpackbits(np.frombuffer(base64.b64decode(steps[0]["plant"]), np.uint8))[: size * size]
        assert cells.shape == (size * size,) and "tough" in steps[0]
        for t, step in enumerate(steps):
            for kind in ("prey", "pred"):
                for uid, x, y, energy in step[kind]:
                    a = animals[str(uid)]
                    assert a["kind"] == kind and 0 <= x < size and 0 <= y < size
                    assert a["born"] <= t < a["died"] and 0 <= a["brain"] <= 1
        ids = [r[0] for r in steps[0]["prey"] + steps[0]["pred"]]
        assert len(ids) == len(set(ids))


def test_episode3d_records_the_stage1_world_and_untrained_brains(tmp_path):
    import scripts.episode3d as ep
    import scripts.train as train
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "p1", "--updates", "2"]) == 0  # predprey
    assert ep.main(["--run-dir", str(tmp_path / "p1")]) == 0
    assert ep.main(["--run-dir", str(tmp_path / "p1"), "--untrained"]) == 0
    for name in ("episode3d.json", "episode3d_untrained.json"):
        out = json.loads((tmp_path / "p1" / name).read_text())
        assert out["steps"] and all(a["kind"] in ("prey", "pred") for a in out["animals"].values())
    assert json.loads((tmp_path / "p1" / "episode3d_untrained.json").read_text())["update"] == 0
