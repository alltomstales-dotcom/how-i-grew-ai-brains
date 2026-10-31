import json
import math

import scripts.eval as eval_script
import scripts.train as train


def test_check_memory_handles_missing_stats():
    ok, msg = train.check_memory(None, 12.0)
    assert ok and "unavailable" in msg
    ok, _ = train.check_memory({}, 12.0)
    assert ok


def test_check_memory_over_limit():
    ok, msg = train.check_memory({"peak_bytes_in_use": 13e9}, 12.0)
    assert not ok and "exceeds" in msg
    ok, _ = train.check_memory({"peak_bytes_in_use": 2e9}, 12.0)
    assert ok


def test_tiny_training_run_writes_outputs(tmp_path):
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "t1",
                     "--updates", "3", "--ckpt-every", "2"])
    assert rc == 0
    run = tmp_path / "t1"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == [1, 2, 3]
    assert all(math.isfinite(v) for r in rows for v in r.values())
    assert {"sps", "env_steps", "prey_alive", "catches_per_step"} <= set(rows[0])
    assert (run / "checkpoints" / "ckpt_000002.msgpack").exists()
    assert (run / "checkpoints" / "latest.msgpack").exists()
    clips = json.loads((run / "clips.json").read_text())
    assert [c["update"] for c in clips] == [0, 2, 3]
    assert all((run / c["file"]).exists() for c in clips)
    assert (run / "index.html").exists()
    runs = json.loads((tmp_path / "runs.json").read_text())["runs"]
    assert [r["id"] for r in runs] == ["t1"] and runs[0]["last"]["update"] == 3
    assert (tmp_path / "roadmap.json").exists()
    config = json.loads((run / "config.json").read_text())
    assert config["world"]["size"] == 12
    assert config["world_name"] == "predprey"


def test_existing_run_id_is_refused(tmp_path):
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "keep.txt").write_text("old run")
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "t1", "--updates", "1"])
    assert rc == 2
    assert (tmp_path / "t1" / "keep.txt").read_text() == "old run"


def test_bad_run_id_is_refused(tmp_path):
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "../escape", "--updates", "1"])
    assert rc == 2
    assert not (tmp_path.parent / "escape").exists()


def test_eval_script_on_tiny_run(tmp_path):
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "t2",
                       "--updates", "2", "--ckpt-every", "2"]) == 0
    rc = eval_script.main(["--run-dir", str(tmp_path / "t2"), "--episodes", "4"])
    assert rc in (0, 1)
    result = json.loads((tmp_path / "t2" / "eval.json").read_text())
    assert {"random_vs_random", "random_prey_vs_trained_pred", "trained_vs_trained",
            "predator_ratio", "prey_ratio", "predator_pass", "prey_pass"} <= set(result)
    assert isinstance(result["predator_pass"], bool)


def test_out_of_memory_exits_with_code_3(tmp_path, monkeypatch):
    real = train.make_trainer

    def failing_trainer(wcfg, tcfg, **kw):
        t = real(wcfg, tcfg, **kw)

        def update(runner):
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 5.13GiB")

        return t._replace(update=update)

    monkeypatch.setattr(train, "make_trainer", failing_trainer)
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "oom", "--updates", "2"])
    assert rc == 3


def test_tiny_evo_run_writes_gene_snapshots(tmp_path):
    import numpy as np

    from simulations import evo
    rc = train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "e1",
                     "--updates", "3", "--ckpt-every", "2"])
    assert rc == 0
    run = tmp_path / "e1"
    assert json.loads((run / "config.json").read_text())["world_name"] == "evo"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert {"prey_speed", "plant_tough", "plant_cover", "pred_gen"} <= set(rows[-1])
    for name in ("genes_000002.npz", "genes_latest.npz"):
        with np.load(run / "checkpoints" / name) as d:
            assert {"prey_genes", "prey_gen", "pred_genes", "pred_gen", "plant_genes"} <= set(d.files)
            assert d["prey_genes"].ndim == 2 and d["prey_genes"].shape[1] == evo.ANIMAL_GENES
            assert d["plant_genes"].shape[1] == 2
    clips = json.loads((run / "clips.json").read_text())
    assert [c["update"] for c in clips] == [0, 2, 3]


def test_eval_script_on_tiny_evo_run(tmp_path):
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "e2",
                       "--updates", "2", "--ckpt-every", "2"]) == 0
    rc = eval_script.main(["--run-dir", str(tmp_path / "e2"), "--episodes", "4"])
    assert rc in (0, 1)
    result = json.loads((tmp_path / "e2" / "eval.json").read_text())
    assert {"prey_evolved_share", "pred_evolved_share", "plant_shift", "evolved_pass", "plant_pass",
            "predator_ratio", "prey_ratio"} <= set(result)
    assert set(result["plant_shift"]) == {"spread", "tough"}
    assert set(result["single_gene_share"]) == {"speed", "sight", "breed", "size", "brain"}
    assert result["genes"].endswith("genes_latest.npz")
    rc = eval_script.main(["--run-dir", str(tmp_path / "e2"), "--episodes", "4",
                           "--checkpoint", str(tmp_path / "e2" / "checkpoints" / "ckpt_000002.msgpack")])
    result = json.loads((tmp_path / "e2" / "eval.json").read_text())
    assert result["genes"].endswith("genes_000002.npz")
    assert isinstance(result["evolved_pass"], bool) and isinstance(result["plant_pass"], bool)


def test_run_writes_done_marker(tmp_path):
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "d1", "--updates", "2"]) == 0
    done = json.loads((tmp_path / "d1" / "done.json").read_text())
    assert done["reason"] == "updates" and done["update"] == 2
    runs = json.loads((tmp_path / "runs.json").read_text())["runs"]
    assert runs[0]["status"] == "ended"


def test_runs_default_to_fifteen_minutes():
    args = train.parse_args([])
    assert args.max_hours == 0.25 and args.ckpt_every == 25 and args.no_learn is False
    assert train.parse_args(["--sample-frac", "0.35"]).sample_frac == 0.35


def test_no_learn_run_is_recorded_as_control(tmp_path):
    assert train.main(["--tiny", "--world", "evo", "--no-learn", "--runs-dir", str(tmp_path),
                       "--run-id", "c1", "--updates", "2"]) == 0
    config = json.loads((tmp_path / "c1" / "config.json").read_text())
    assert config["train"]["learn"] is False
    runs = json.loads((tmp_path / "runs.json").read_text())["runs"]
    assert runs[0]["learn"] is False


def test_learners_flag_is_recorded(tmp_path):
    assert train.main(["--tiny", "--world", "evo", "--learners", "pred", "--runs-dir", str(tmp_path),
                       "--run-id", "lp", "--updates", "1"]) == 0
    assert json.loads((tmp_path / "lp" / "config.json").read_text())["train"]["learners"] == "pred"


def test_out_of_memory_raised_late_by_async_dispatch_exits_with_code_3(tmp_path, monkeypatch):
    # On the GPU, JAX reported pp-evo-002's OOM when the metrics were fetched, not in update().
    real_get = train.jax.device_get

    def device_get(x):
        if isinstance(x, dict) and "prey_alive" in x:
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 11.73GiB")
        return real_get(x)

    monkeypatch.setattr(train.jax, "device_get", device_get)
    rc = train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "oom2", "--updates", "2"])
    assert rc == 3
    assert json.loads((tmp_path / "oom2" / "done.json").read_text())["reason"] == "out_of_memory"


def test_init_from_carries_an_evo_brain_into_the_terrain_world(tmp_path):
    from agents.checkpoint import load_checkpoint
    from agents.networks import ActorCritic, init_params
    import jax
    import numpy as np
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "src",
                       "--updates", "1", "--ckpt-every", "1"]) == 0
    src = tmp_path / "src" / "checkpoints" / "latest.msgpack"
    assert train.main(["--tiny", "--world", "terrain", "--runs-dir", str(tmp_path), "--run-id", "dst",
                       "--updates", "1", "--ckpt-every", "1", "--init-from", str(src), "--no-learn"]) == 0
    config = json.loads((tmp_path / "dst" / "config.json").read_text())
    assert config["args"]["init_from"] == str(src) and config["world_name"] == "terrain"
    # With learning off, the saved brain is exactly the source brain.
    template = init_params(ActorCritic(), jax.random.PRNGKey(0), 9, 10)
    new = load_checkpoint(tmp_path / "dst" / "checkpoints" / "latest.msgpack", template)
    old = load_checkpoint(src, template)
    for a, b in zip(jax.tree.leaves(new[:2]), jax.tree.leaves(old[:2])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_eval_script_on_tiny_terrain_run(tmp_path):
    assert train.main(["--tiny", "--world", "terrain", "--runs-dir", str(tmp_path), "--run-id", "t3",
                       "--updates", "1", "--ckpt-every", "1"]) == 0
    assert eval_script.main(["--run-dir", str(tmp_path / "t3"), "--episodes", "4"]) in (0, 1)
    result = json.loads((tmp_path / "t3" / "eval.json").read_text())
    assert {"trained", "random"} <= set(result["burrow"]) and isinstance(result["burrow_pass"], bool)


def test_league_training_run(tmp_path):
    assert train.main(["--tiny", "--world", "evo", "--runs-dir", str(tmp_path), "--run-id", "lg",
                       "--updates", "3", "--ckpt-every", "1", "--league", "2", "--num-envs", "8"]) == 0
    config = json.loads((tmp_path / "lg" / "config.json").read_text())
    assert config["train"]["league_size"] == 2
    rows = [json.loads(line) for line in (tmp_path / "lg" / "metrics.jsonl").read_text().splitlines()]
    assert "league_pred_catch" in rows[-1]


def test_set_overrides_world_config(tmp_path):
    assert train.main(["--tiny", "--world", "terrain", "--runs-dir", str(tmp_path), "--run-id", "s1",
                       "--updates", "1", "--set", "burrows=5", "--set", "bite_regrow=0.5"]) == 0
    world = json.loads((tmp_path / "s1" / "config.json").read_text())["world"]
    assert world["burrows"] == 5 and world["bite_regrow"] == 0.5


def test_set_rejects_unknown_fields(tmp_path):
    assert train.main(["--tiny", "--runs-dir", str(tmp_path), "--run-id", "s2", "--updates", "1",
                       "--set", "nonsense=1"]) == 2


def test_memory_run_with_clips_eval_and_league(tmp_path):
    import scripts.league as league
    assert train.main(["--tiny", "--world", "seasons", "--runs-dir", str(tmp_path), "--run-id", "m1",
                       "--updates", "2", "--ckpt-every", "1", "--memory", "8"]) == 0
    config = json.loads((tmp_path / "m1" / "config.json").read_text())
    assert config["train"]["memory"] == 8
    assert eval_script.main(["--run-dir", str(tmp_path / "m1"), "--episodes", "2"]) in (0, 1)
    assert league.main(["--run-dir", str(tmp_path / "m1"), "--episodes", "2"]) == 0
