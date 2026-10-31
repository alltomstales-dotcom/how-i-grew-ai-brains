"""Stage 11a: animals crossing between continents."""
import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from agents.ppo import make_trainer, tiny_configs
from scripts.transit import Transit
from simulations import evo
from simulations.worlds import get_world


def runner_for(me, cfg, tcfg):
    return make_trainer(cfg, tcfg, world=evo).init(jax.random.PRNGKey(me))


def put_at_edge(runner, cfg):
    p = runner.env_state.prey
    pos = np.array(p.pos); pos[..., 0] = cfg.size - 1
    return runner._replace(env_state=runner.env_state._replace(prey=p._replace(pos=pos)))


def test_animals_leave_one_continent_and_arrive_with_their_genes_and_lineage(tmp_path):
    cfg, tcfg = tiny_configs("meadow")
    cfg = replace(cfg, size=12)
    a, b = runner_for(0, cfg, tcfg), runner_for(1, cfg, tcfg)
    ta = Transit(tmp_path, 0, 1, 1.0, 2, cfg, evo)
    tb = Transit(tmp_path, 1, 0, 1.0, 2, cfg, evo)
    a = put_at_edge(a, cfg)
    before = np.asarray(a.env_state.prey.alive).sum()
    genes = np.array(a.env_state.prey.genes)[np.asarray(a.env_state.prey.alive)]
    a2, sa = ta.exchange(a, 1)
    assert sa["prey_sent"] == before and np.asarray(a2.env_state.prey.alive).sum() == 0
    assert list((tmp_path / "to-1").glob("*.npz"))
    alive_b = np.asarray(b.env_state.prey.alive).sum()
    b = b._replace(env_state=b.env_state._replace(prey=b.env_state.prey._replace(pos=np.zeros_like(b.env_state.prey.pos))))
    b2, sb = tb.exchange(b, 1)
    p = b2.env_state.prey
    alive, tag = np.asarray(p.alive), np.asarray(p.tag)
    assert sb["prey_received"] > 0 and alive.sum() == alive_b + sb["prey_received"]
    assert (tag[alive] == 1).sum() == sb["prey_received"]  # continent 0's lineage is tag 1
    arrived = np.asarray(p.genes)[alive & (tag == 1)]
    assert all((np.isclose(genes, g).all(1)).any() for g in arrived)
    assert (np.asarray(p.pos)[alive & (tag == 1)][:, 0] < 2).all()  # the west edge
    assert not list((tmp_path / "to-1").glob("*.npz"))
    assert sb["prey_immigrant_share"] > 0


def test_a_lost_cause_is_not_fatal_a_broken_file_waits(tmp_path):
    cfg, tcfg = tiny_configs("meadow")
    cfg = replace(cfg, size=12)
    r = runner_for(0, cfg, tcfg)
    t = Transit(tmp_path, 0, 1, 0.0, 2, cfg, evo)
    (tmp_path / "to-0" / "2-0000001.npz").write_bytes(b"half a file")
    r2, s = t.exchange(r, 1)
    assert s["prey_received"] == 0 and (tmp_path / "to-0" / "2-0000001.npz").exists()
    assert np.asarray(r2.prey_obs).shape == np.asarray(r.prey_obs).shape


def test_climates_differ_only_as_intended():
    m, f, t = (get_world(n)[1]() for n in ("meadow", "forest", "tundra"))
    assert (m.size, m.max_prey, m.max_pred) == (256, 2730, 910) and m.stealth_radius == 1 and m.seed_rate == 0.2
    assert f.move_cost == 0.5 and f.thickets == 0.25 and m.thickets == 0 and t.winter_len == 60 and m.winter_len == 0
    assert {m.climate, f.climate, t.climate} == {"meadow", "forest", "tundra"}


def test_training_runs_with_crossing(tmp_path):
    import scripts.train as train
    common = ["--tiny", "--runs-dir", str(tmp_path), "--updates", "2", "--world", "tundra", "--transit-dir", str(tmp_path / "x"), "--transit-prob", "1"]
    assert train.main([*common, "--run-id", "a", "--transit-me", "0", "--transit-to", "1"]) == 0
    assert train.main([*common, "--run-id", "b", "--transit-me", "1", "--transit-to", "0"]) == 0
    rows = [json.loads(l) for l in open(tmp_path / "b" / "transit.jsonl")]
    assert len(rows) == 2 and {"prey_sent", "pred_received", "pred_immigrant_share", "prey_speed"} <= set(rows[0])
