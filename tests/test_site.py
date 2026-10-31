import json
import os
import time

import scripts.site as site


def make_run(root, name, world, rows, eval_result=None, age_s=0):
    run = root / name
    (run / "clips").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"world_name": world, "world": {}}))
    (run / "index.html").write_text("<html></html>")
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "clips.json").write_text(json.dumps([{"file": "clips/random.mp4", "update": 0, "label": "random"},
                                                {"file": "clips/clip_000002.mp4", "update": 2, "label": "u2"}]))
    if eval_result is not None:
        (run / "eval.json").write_text(json.dumps(eval_result))
    then = time.time() - age_s
    for f in run.iterdir():
        os.utime(f, (then, then))
    return run


def rows(n, **extra):
    return [{"update": i, "env_steps": 100 * i, "wall_s": 10.0 * i, "sps": 3000.0, "prey_alive": 30.0,
             "pred_alive": 7.0, "catches_per_step": 0.1, **extra} for i in range(1, n + 1)]


def test_build_site_lists_runs_newest_first(tmp_path):
    make_run(tmp_path, "old", "predprey", rows(3), {"predator_ratio": 3.4, "prey_ratio": 1.6,
                                                    "predator_pass": True, "prey_pass": True, "meta": {}},
             age_s=7200)
    make_run(tmp_path, "new", "evo", rows(5, plant_tough=0.5))
    (tmp_path / "not-a-run").mkdir()
    site.build_site(tmp_path)
    data = json.loads((tmp_path / "runs.json").read_text())
    assert [r["id"] for r in data["runs"]] == ["new", "old"]
    new, old = data["runs"]
    assert new["world"] == "evo" and new["status"] == "running"
    assert new["last"]["update"] == 5 and new["last"]["plant_tough"] == 0.5
    assert new["latest_clip"] == "clips/clip_000002.mp4"
    assert old["status"] == "ended" and old["eval"]["predator_pass"] is True
    assert "meta" not in old["eval"]
    assert (tmp_path / "index.html").read_text().startswith("<!doctype html>")
    assert json.loads((tmp_path / "roadmap.json").read_text())["stages"]


def test_last_row_reads_tail_and_skips_partial_line(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text("".join(json.dumps({"update": i, "pad": "x" * 200}) + "\n" for i in range(5000)) + '{"update": 50')
    assert site.last_row(p)["update"] == 4999
    assert site.last_row(tmp_path / "missing.jsonl") is None


def test_roadmap_json_is_well_formed():
    data = json.loads(site.ROADMAP.read_text())
    assert {"title", "updated", "summary", "explainer", "now", "stages", "lessons"} <= set(data)
    statuses = {s["status"] for s in data["stages"]}
    assert statuses <= {"done", "running", "next", "planned"}
    for s in data["stages"]:
        assert {"id", "name", "goal", "status", "built", "remaining", "coming"} <= set(s)


def test_done_marker_ends_a_fresh_run(tmp_path):
    run = make_run(tmp_path, "r", "evo", rows(2))
    (run / "done.json").write_text(json.dumps({"reason": "max_hours", "update": 2}))
    data = site.build_site(tmp_path) or json.loads((tmp_path / "runs.json").read_text())
    r = data["runs"][0]
    assert r["status"] == "ended" and r["learn"] is True


def test_league_result_is_summarised(tmp_path):
    run = make_run(tmp_path, "r", "evo", rows(2))
    (run / "league.json").write_text(json.dumps({"updates": [25, 50], "pred_elo": [950.0, 1100.0],
                                                 "prey_elo": [1000.0, 950.0], "pred_progress": True,
                                                 "prey_progress": False}))
    r = site.build_site(tmp_path)["runs"][0]
    assert r["league"] == {"pred_gain": 150.0, "prey_gain": -50.0, "pred_progress": True, "prey_progress": False}


def test_species_and_compare_results_are_summarised(tmp_path):
    run = make_run(tmp_path, "r", "niches", rows(2))
    split = {"animals": 50, "separation": 2.4, "minority": 0.3, "main_gene": "size", "centres": []}
    (run / "species.json").write_text(json.dumps({"snapshots": [{"update": 25, "prey": split, "pred": None}],
                                                  "pass": True}))
    (run / "compare.json").write_text(json.dumps({"baseline": "runs/ff", "prey_ratio": 1.3, "catch_ratio": 0.9,
                                                  "pass": True, "window": [1, 2]}))
    r = site.build_site(tmp_path)["runs"][0]
    assert r["species"] == {"pass": True, "prey": split, "pred": None}
    assert r["compare"] == {"baseline": "ff", "prey_ratio": 1.3, "catch_ratio": 0.9, "pass": True}


def test_chase_and_signal_results_are_summarised(tmp_path):
    run = make_run(tmp_path, "r", "signals", rows(2))
    (run / "chase.json").write_text(json.dumps({"update": 9, "chase_vs_trained_prey": 0.1, "trained_vs_trained_prey": 0.2,
                                                "ratio_vs_trained_prey": 0.5, "ratio_vs_random_prey": 0.7}))
    (run / "signals.json").write_text(json.dumps({"update": 9, "catches": 0.2, "catches_muted": 0.15, "drop": 0.25,
                                                  "pass": True, "prey": {"rate": 0.1, "mi_bits": 0.01, "table": []},
                                                  "pred": {"rate": 0.3, "mi_bits": 0.2, "table": []}}))
    r = site.build_site(tmp_path)["runs"][0]
    assert r["chase"] == {"ratio_vs_trained_prey": 0.5, "ratio_vs_random_prey": 0.7}
    assert r["signals"] == {"drop": 0.25, "pass": True, "prey_rate": 0.1, "pred_rate": 0.3,
                            "prey_mi_bits": 0.01, "pred_mi_bits": 0.2}


def test_evolved_body_results_are_summarised_too(tmp_path):
    run = make_run(tmp_path, "r", "flank", rows(2))
    (run / "chase_evolved.json").write_text(json.dumps({"bodies": "genes_latest.npz", "ratio_vs_trained_prey": 0.4,
                                                        "ratio_vs_random_prey": 0.6}))
    (run / "signals_evolved.json").write_text(json.dumps({"drop": 0.2, "pass": True, "prey": {"rate": 0.1, "mi_bits": 0.0},
                                                          "pred": {"rate": 0.3, "mi_bits": 0.1}}))
    r = site.build_site(tmp_path)["runs"][0]
    assert r["chase"] is None and r["chase_evolved"] == {"ratio_vs_trained_prey": 0.4, "ratio_vs_random_prey": 0.6}
    assert r["signals_evolved"]["drop"] == 0.2 and r["signals"] is None
