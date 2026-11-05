"""Dynasties (scripts.lineage): uids assigned at birth survive slot reuse, parents link up, and line moments are right."""
import json

import jax
import numpy as np
import pytest

from scripts import lineage as L
from simulations import evo


def test_a_reused_slot_gets_a_new_uid_and_the_right_parent():
    tr = L.Tracer()
    tr.start("prey", np.array([True, True, True, False]))  # uids 0, 1, 2 in slots 0, 1, 2
    # step 1: slot 2 is eaten and slot 0's child takes it in the same step; slot 1's child goes to the empty slot 3
    tr.step(1, "prey", parent=np.array([0, 1, 0, 1]), was_alive=np.array([True, True, True, False]),
            alive=np.array([True, True, True, True]), eaten=np.array([False, False, True, False]))
    a = tr.animals
    assert a[2]["died"] == 1 and a[2]["cause"] == "eaten"
    assert [x["uid"] for x in a[3:]] == [3, 4]
    assert (a[3]["slot"], a[3]["parent_uid"], a[3]["born"]) == (2, 0, 1)
    assert (a[4]["slot"], a[4]["parent_uid"]) == (3, 1)
    assert list(tr.slots["prey"]) == [0, 1, 3, 4]
    # step 2: slot 0 starves, its grandchild is born into slot 0 from slot 2 (uid 3)
    tr.step(2, "prey", parent=np.array([2, 1, 2, 3]), was_alive=np.ones(4, bool), alive=np.ones(4, bool),
            eaten=np.zeros(4, bool))
    assert a[0]["died"] == 2 and a[0]["cause"] == "starved"
    assert (a[5]["slot"], a[5]["parent_uid"]) == (0, 3)
    assert list(tr.slots["prey"]) == [5, 1, 3, 4]
    # step 3: two die with no births
    tr.step(3, "prey", parent=np.arange(4), was_alive=np.ones(4, bool), alive=np.array([False, True, False, True]))
    assert a[5]["died"] == 3 and a[3]["died"] == 3 and a[5]["cause"] == "died"
    assert list(tr.slots["prey"]) == [-1, 1, -1, 4]


def test_a_birth_from_an_empty_slot_is_refused():
    tr = L.Tracer()
    tr.start("pred", np.array([True, False, False]))
    with pytest.raises(ValueError):
        tr.step(1, "pred", parent=np.array([0, 2, 2]), was_alive=np.array([True, False, False]), alive=np.array([True, True, False]))


TINY = evo.ContinentConfig(size=12, max_prey=12, max_pred=6, init_prey=8, init_pred=4, max_steps=10_000,
                           prey_repro_energy=6.0, pred_repro_energy=20.0, catch_energy=25.0)


def test_uids_follow_a_real_world_through_slot_reuse():
    shim, exact = L._extended(evo)
    assert exact
    step = jax.jit(lambda k, s, a, b: shim.advance(TINY, k, s, a, b))
    key = jax.random.PRNGKey(5)
    s = evo.reset(TINY, key)
    tr = L.Tracer()
    for kind in L.SPECIES:
        tr.start(kind, getattr(s, kind).alive)
    reused, eaten_total, catches = 0, 0, 0
    for t in range(1, 301):
        key, k1, k2, k3 = jax.random.split(key, 4)
        pa = jax.random.randint(k1, (TINY.max_prey,), 0, evo.NUM_ACTIONS)
        qa = jax.random.randint(k2, (TINY.max_pred,), 0, evo.NUM_ACTIONS)
        was = {k: np.asarray(getattr(s, k).alive) for k in L.SPECIES}
        res = step(k3, s, pa, qa)
        info = jax.device_get(res.info)
        before = tr.slots["prey"].copy()
        tr.step(t, "prey", info.prey_parent, was["prey"], info.x_prey_alive, info.x_prey_eaten)
        tr.step(t, "pred", info.pred_parent, was["pred"], info.x_pred_alive)
        born = info.prey_parent != np.arange(TINY.max_prey)
        reused += int((born & was["prey"]).sum())  # a newborn in a slot whose animal died this very step
        assert all(before[i] != tr.slots["prey"][i] for i in np.nonzero(born)[0])
        eaten_total += int(info.x_prey_eaten.sum())
        catches += int(info.catches)
        s = res.state
        if not (np.asarray(s.prey.alive).any() and np.asarray(s.pred.alive).any()):
            break
    assert reused > 0 and eaten_total == catches
    a = tr.animals
    assert [x["uid"] for x in a] == list(range(len(a)))
    assert sum(x["cause"] == "eaten" for x in a) == catches
    kids = [x for x in a if x["parent_uid"] is not None]
    assert kids
    for x in kids:
        p = a[x["parent_uid"]]
        assert p["species"] == x["species"] and p["born"] < x["born"]
        assert p["died"] is None or p["died"] > x["born"]  # the parent was alive when the child was born


def _animal(uid, parent, born, died, kind="prey"):
    return {"uid": uid, "parent_uid": parent, "species": kind, "born": born, "died": died, "cause": None, "slot": uid}


def test_line_moments_rise_fall_and_extinction():
    # Founders 0 and 1, each 50% at step 0. Line 0 grows to 4 of 5 (80%) at step 20, then dies out at 60;
    # line 1 grows to 3 of 4 (75%) at step 3, is cut to its founder at 20, and comes back to take over.
    animals = [_animal(0, None, 0, 40), _animal(1, None, 0, None),
               _animal(2, 0, 5, 50), _animal(3, 0, 10, 45), _animal(4, 2, 15, 60),
               _animal(5, 1, 2, 20), _animal(6, 1, 3, 20), _animal(7, 1, 70, None), _animal(8, 7, 80, None)]
    rows = {r["founder"]: r for r in L.analyse(animals, 100, every=10)}
    a, b = rows[0], rows[1]
    assert (a["descendants"], a["generations"], a["children"]) == (3, 2, 2)
    m = {(x["kind"], x.get("pct")): x["step"] for x in a["moments"]}
    assert ("rise", 50) not in m and m[("rise", 80)] == 20  # it starts at 50%, so it doesn't "rise" past 50
    assert (a["peak_pct"], a["peak_step"]) == (80.0, 20) and ("takeover", None) not in m
    # from 60 to 69 only line 1's founder lives, under the floor of 2, so shares don't count there: the fall shows at 70
    assert m[("fall", 0.0)] == 70 and a["extinct_step"] == 60 and m[("extinction", None)] == 60
    last = next(x for x in a["moments"] if x["kind"] == "last_survivor")
    assert (last["step"], last["uid"]) == (50, 4)  # from step 50 only #4 is left
    assert a["arc"] == "rise_fall" and a["drama"] == 30.0  # up 30 points, then down 80: the smaller counts
    mb = {x["kind"]: x for x in b["moments"]}
    assert (mb["near_extinction"]["step"], mb["near_extinction"]["members"]) == (20, 1)
    assert (mb["comeback"]["step"], mb["comeback"]["pct"]) == (80, 100.0)  # back with 3 living, not at 60 with only its founder
    assert mb["takeover"]["step"] == 70  # 100% with at least two living
    assert b["extinct_step"] is None and b["final_pct"] == 100.0
    # comeback: from 75% down to 20%, then up to 100% (55 points); takeover: 50% to 100%, counted at half
    assert b["drama_parts"] == {"rise_fall": 0.0, "comeback": 55.0, "takeover": 25.0}
    assert b["arc"] == "comeback" and b["drama"] == 80.0
    oldest = next(x for x in b["moments"] if x["kind"] == "longest_lived")
    assert oldest["uid"] == 1 and oldest["alive_at_end"]
    # the series is (step, percent, members), every 10 steps plus the moments' steps
    assert b["series"][0] == [0, 50.0, 1, 2] and [100, 100.0, 3, 3] in b["series"]  # step, percent, members, species alive
    assert (b["peak_step"], b["share_floor"]) == (70, 2)


def test_rivals_dying_is_not_a_comeback():
    # Line 0 grows to 3, falls to its founder at step 10; line 1 dies at 20, so line 0 is 100% of a species of one.
    animals = [_animal(0, None, 0, None), _animal(1, None, 0, 20), _animal(2, 0, 1, 10), _animal(3, 0, 2, 10)]
    a = {r["founder"]: r for r in L.analyse(animals, 50)}[0]
    kinds = [m["kind"] for m in a["moments"]]
    assert "comeback" not in kinds and a["drama_parts"]["comeback"] == 0.0
    assert a["final_pct"] == 100.0 and "takeover" not in kinds  # a takeover needs two living
    assert not any(m["kind"] == "rise" for m in a["moments"]) and a["peak_pct"] == 75.0  # 3 of 4 at step 2; later 100% of a species of one is not a population


def test_a_line_that_dies_with_its_species_is_flagged():
    animals = [_animal(0, None, 0, 30, "pred"), _animal(1, None, 0, 10, "pred"), _animal(2, 0, 5, 40, "pred"),
               _animal(3, None, 0, None)]
    rows = {r["founder"]: r for r in L.analyse(animals, 50)}
    assert rows[0]["extinct_step"] == 40 and rows[0]["fell_with_species"]
    assert not rows[1]["fell_with_species"]
    ext = next(m for m in rows[0]["moments"] if m["kind"] == "extinction")
    assert ext["with_species"] and "last of the hunters" in ext["text"]


def test_trace_a_tiny_individual_run_end_to_end(tmp_path):
    import scripts.train as train
    import scripts.train_indiv as ti
    assert train.main(["--tiny", "--world", "tiny_stealth", "--runs-dir", str(tmp_path), "--run-id", "f", "--updates", "1", "--no-clips"]) == 0
    assert ti.main(["--tiny", "--world", "tiny_stealth", "--mode", "darwin", "--founders", str(tmp_path / "f"),
                    "--runs-dir", str(tmp_path), "--run-id", "i", "--updates", "2", "--ckpt-every", "1"]) == 0
    out = tmp_path / "lineage"
    assert L.main(["--run-dir", str(tmp_path / "i"), "--out-dir", str(out), "--episodes", "2", "--steps", "60", "--replays", "1"]) == 0
    d = json.loads((out / "i" / "lineage.json").read_text())
    assert d["cause_of_death"] == "exact" and len(d["episodes_traced"]) == 2
    ep = d["episodes_traced"][0]
    uids = {a["uid"] for a in ep["animals"]}
    assert all(a["parent_uid"] is None or a["parent_uid"] in uids for a in ep["animals"])
    assert {r["founder"] for r in ep["founders"]} == {a["uid"] for a in ep["animals"] if a["parent_uid"] is None}
    rep = json.loads((out / "i" / "episode3d.json").read_text())
    assert len(rep["steps"]) == 61
    for t, f in enumerate(rep["steps"]):
        for kind in ("prey", "pred"):
            for uid, x, y, _e in f[kind]:
                a = rep["animals"][str(uid)]
                assert a["kind"] == kind and a["born"] <= t < a["died"] and str(a["line"]) in rep["lines"]
    assert (out / "lines.md").read_text().startswith("# Dynasties")
    # a shared-brain run: every animal plays the run's one brain, and lines are lines of bodies
    assert L.main(["--run-dir", str(tmp_path / "f"), "--out-dir", str(out), "--episodes", "1", "--steps", "30", "--replays", "1"]) == 0
    d = json.loads((out / "f" / "lineage.json").read_text())
    assert d["run_kind"] == "shared" and d["cause_of_death"] == "exact"
    assert len(json.loads((out / "f" / "episode3d.json").read_text())["steps"]) == 31
