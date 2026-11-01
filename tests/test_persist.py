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
    assert persist.main(["--run-dir", str(tmp_path / "a"), "--episodes", "2", "--force-pred", "breed=0.1,sight=0.7"]) == 0
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


def test_speed_costs_sprint_hunters_nothing_and_grazers_as_before():
    import jax
    import jax.numpy as jnp
    from simulations import evo
    QUIET = dict(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2, seed_rate=0.0, die_rate=0.0, germinate_rate=0.0, birth_cost=0.0, gene_sigma=0.0, sight_cost=0.0)
    still = jnp.zeros(4, jnp.int32)
    def spend(cfg):
        s = evo.make_state(cfg, prey=[(1, 1, 30.0, 1.0, 0.5, 1.0)], pred=[(8, 8, 30.0, 1.0, 0.5, 1.0)])
        r = evo.advance(cfg, jax.random.PRNGKey(0), s, still, still).state
        return 30.0 - float(r.prey.energy[0]), 30.0 - float(r.pred.energy[0])
    plain, sprint = spend(evo.EvoConfig(**QUIET)), spend(evo.EvoConfig(**QUIET, pred_speed_cost=0.0))
    assert plain[0] == sprint[0] and abs(plain[1] - sprint[1] - 1.0) < 1e-5


def test_a_grazer_sees_a_camouflaged_hunter_only_from_the_next_cell():
    from dataclasses import replace
    from simulations import evo
    base = evo.EvoConfig(size=12, max_prey=4, max_pred=4, init_prey=2, init_pred=2)
    genes = (0.0, 1.0, 0.5)  # speed, sight 1 gives the widest view, breed
    def hunters_seen(cfg, x):
        s = evo.make_state(cfg, prey=[(5, 5, 20.0, *genes)], pred=[(x, 5, 30.0, *genes)])
        prey_obs, pred_obs = evo.observe(cfg, s)
        return float(prey_obs[0, ..., 3].sum()), float(pred_obs[0, ..., 2].sum())
    stealth = replace(base, stealth_radius=1)
    assert hunters_seen(base, 8)[0] == 1.0 and hunters_seen(stealth, 8)[0] == 0.0
    assert hunters_seen(stealth, 6)[0] == 1.0
    assert hunters_seen(stealth, 8)[1] == 1.0  # hunters still see grazers


def test_the_roomy_world_only_changes_the_slots():
    from simulations.worlds import get_world
    a, b = get_world("lasting_stealth")[1](), get_world("lasting_room")[1]()
    assert (b.max_prey, b.max_pred) == (a.max_prey * 3, a.max_pred * 3) and (b.stealth_radius, b.seed_rate, b.catch_energy) == (a.stealth_radius, a.seed_rate, a.catch_energy)


def test_the_three_size_arms_differ_only_as_intended():
    from simulations.worlds import get_world
    base, cheap, nocatch = (get_world(n)[1]() for n in ("bulk_base", "bulk_cheap", "bulk_nocatch"))
    assert base.sizes and base.catch_scale == 0.25 and base.size_cost == 0.5 and base.size_catch and base.stealth_radius == 1
    assert cheap.size_cost == 0.1 and cheap.size_catch
    assert not nocatch.size_catch and nocatch.size_cost == 0.5


def test_the_sweep_runs_on_a_tiny_world(tmp_path):
    import scripts.lasting as lasting
    assert lasting.main(["--tiny", "--steps", "30", "--episodes", "2", "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "lasting.json").read_text())
    assert len(out) == 1 and isinstance(out[0]["pass"], bool)


def test_transplant_swaps_only_the_focal_brain(tmp_path):
    import json
    import scripts.train as train
    import scripts.transplant as transplant
    common = ["--tiny", "--runs-dir", str(tmp_path), "--updates", "2", "--world", "lasting", "--no-clips", "--ckpt-every", "1"]
    for run, seed in (("a", "0"), ("b", "1"), ("c", "2")):
        assert train.main([*common, "--run-id", run, "--seed", seed]) == 0
    assert transplant.main(["--target", str(tmp_path / "a"), "--same", str(tmp_path / "b"), "--other", str(tmp_path / "c"), "--episodes", "2"]) == 0
    out = json.loads((tmp_path / "a" / "transplant.json").read_text())
    assert out["same"] == "b" and out["other"] == "c" and out["hunter_ratio"] >= 0 and out["grazer_ratio"] >= 0
    assert set(out) >= {"home", "hunter_same", "hunter_other", "grazer_same", "grazer_other"}

