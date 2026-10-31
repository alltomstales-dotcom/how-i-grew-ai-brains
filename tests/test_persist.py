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
    assert persist.main(["--run-dir", str(tmp_path / "a"), "--episodes", "2", "--bot-hunters"]) == 0
    assert persist.main(["--run-dir", str(tmp_path / "a"), "--episodes", "2", "--steps", "120"]) == 0
    assert json.loads((tmp_path / "a" / "persist.json").read_text())["curve"][-1]["step"] == 119


def test_big_meal_worlds_change_only_meals_and_the_cap():
    from simulations.worlds import get_world
    plain = get_world("lasting")[1]()
    for name, meal, cap in (("lasting_m100", 100.0, 200.0), ("lasting_m250", 250.0, 500.0)):
        c = get_world(name)[1]()
        assert (c.catch_energy, c.pred_energy_cap, c.seed_rate) == (meal, cap, plain.seed_rate) and c.pred_repro_energy == plain.pred_repro_energy


def test_costly_dodging_worlds_are_ambush_with_lasting_plants():
    from simulations.worlds import get_world
    a = get_world("ambush")[1]()
    for name, meal in (("lasting_ambush", a.catch_energy), ("lasting_ambush_m100", 100.0)):
        c = get_world(name)[1]()
        assert c.seed_rate == 0.2 and c.move_cost == a.move_cost > 0 and c.thickets == a.thickets > 0 and c.catch_energy == meal


def test_the_sweep_runs_on_a_tiny_world(tmp_path):
    import scripts.lasting as lasting
    assert lasting.main(["--tiny", "--steps", "30", "--episodes", "2", "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "lasting.json").read_text())
    assert len(out) == 1 and isinstance(out[0]["pass"], bool)
