"""Stage 21b studio clip (scripts/oracle_clip.py): the call field, the shared start and stream, and the moment finder."""
import dataclasses

import jax
import numpy as np

from agents.networks import init_params
from scripts import oracle_clip as oc
from simulations import evo
from simulations.worlds import get_world, make_net, obs_channels

STEPS = 30


def _setup():
    world, cls = get_world("alarm")
    cfg = dataclasses.replace(cls(), size=16, max_prey=12, max_pred=4, init_prey=10, init_pred=3, max_steps=STEPS,
                              oracle_call=True, signal_cost=0.3, kin_share=0.5, stealth_radius=1)
    net = make_net(world, cfg)
    ch = obs_channels(world, cfg)
    params = init_params(net, jax.random.PRNGKey(1), cfg.view, ch)
    return cfg, net, params, ch


def _deaf(params, channels):
    z = lambda x: x.at[:, :, channels - 1, :].set(0.0) if x.ndim == 4 and x.shape[2] == channels else x
    return jax.tree.map(z, params)


def _play(cfg, net, prey_p, pred_p, mode):
    play, exact = oc.make_play(cfg, net, prey_p, pred_p, STEPS)
    return oc.trace(*jax.device_get(play(np.array([0, 1], np.uint32), np.int32(mode))), 0, STEPS, exact)


def test_the_call_field_is_the_oracle_bit_row_by_row():
    cfg, net, p, _ = _setup()
    tr, view, sim, mismatched = _play(cfg, net, p, p, 0)
    assert mismatched == 0 and len(view) == STEPS + 1
    calls = 0
    for f, v in zip(sim, view):
        assert len(v["prey_call"]) == len(v["prey"])
        d = np.abs(oc._wrap(f["pos"][:, None, :] - f["pred_pos"][None, :, :], cfg.size)).max(-1)
        adj = f["alive"] & ((d <= 1) & f["pred_alive"][None, :]).any(-1)
        slots = np.nonzero(f["uid"] >= 0)[0]
        assert [r[0] for r in v["prey"]] == [int(f["uid"][s]) for s in slots]
        if v is not view[-1]:
            assert v["prey_call"] == [int(adj[s]) for s in slots]
            calls += sum(v["prey_call"])
    assert calls > 0


def test_muting_changes_nothing_for_listeners_with_no_call_weights():
    cfg, net, p, ch = _setup()
    deaf = _deaf(p, ch)
    a, b = _play(cfg, net, deaf, p, 0), _play(cfg, net, deaf, p, 1)
    assert a[1] == b[1]  # same start, same stream: identical frames when hearing can't matter


def _frame(prey, pred, uid, act, call=None):
    n = len(uid)
    return {"uid": np.array(uid), "pos": np.array(prey), "alive": np.array(uid) >= 0, "act": np.array(act),
            "call": np.zeros(n, bool) if call is None else np.array(call), "pred_pos": np.array(pred),
            "pred_alive": np.ones(len(pred), bool)}


def test_a_moment_needs_a_quiet_listener_two_to_four_cells_off_stepping_away():
    # caller 0 at (5, 5) beside the hunter at (6, 5). grazer 1 three cells off steps away (-x); grazer 2 one cell off
    # (too close); grazer 3 is 4 cells off but steps toward it.
    from simulations.predprey import DELTAS
    left = next(i for i, d in enumerate(np.asarray(DELTAS)) if tuple(d) == (-1, 0))
    right = next(i for i, d in enumerate(np.asarray(DELTAS)) if tuple(d) == (1, 0))
    prey = [(5, 5), (2, 5), (4, 5), (8, 9)]
    f0 = _frame(prey, [(6, 5)], [0, 1, 2, 3], [0, 0, 0, 0], call=[True, False, False, False])
    f1 = _frame(prey, [(6, 5)], [0, 1, 2, 3], [0, left, 0, left])
    f2 = _frame(prey, [(6, 5)], [0, 1, 2, 3], [0, 0, 0, 0])
    got = oc.heard_calls([f0, f1, f2], 16, 4)
    assert got == [{"step": 0, "heard_step": 1, "caller": 0, "listeners": [{"uid": 1, "distance": 3}]}]
    assert oc._move(right, np.array([3, 0])) == "toward" and oc._move(0, np.array([3, 0])) == "still"


def test_identities_follow_an_animal_across_versions_that_number_newborns_differently():
    real = [{"uid": 0, "parent_uid": None, "species": "prey", "slot": 0, "born": 0},
            {"uid": 1, "parent_uid": 0, "species": "prey", "slot": 3, "born": 9},
            {"uid": 2, "parent_uid": 0, "species": "prey", "slot": 4, "born": 12}]
    muted = [real[0], {**real[2], "uid": 1}, {**real[1], "uid": 2}]
    a, b = oc.identities(real), oc.identities(muted)
    assert a[1] == b[2] and a[2] == b[1]
