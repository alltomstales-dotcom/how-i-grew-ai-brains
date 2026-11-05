"""Stage 21 tests: alarm calls and eavesdropping. Spec: docs/specs/2026-11-04-alarm-calls-design.md

  python -m scripts.alarm eval --grazers runs/A --hunters runs/B --out DIR --tag NAME [--conds real:real,mute:real]
         [--eavesdrop] [--probe runs/C ...] [--episodes 256] [--seed 12345]
  python -m scripts.alarm clip --grazers runs/A --hunters runs/B --out DIR --tag NAME [--eavesdrop] [--seed S] [--mute prey|pred]
  python -m scripts.alarm score --phase A|B|C ...   (see score_* below)

`eval` plays frozen grazers against frozen hunters in 256 worlds for 500 steps with no restarts, once per reception
condition, from the same starting states and random streams. A condition is PREY:PRED, what grazers and hunters hear:
real (last step's calls), mute (nothing) or shuffle (the same number of calls, moved among living grazers of the same
world, step, local density and energy; positions and counts are kept, the link to danger is broken). Callers still
decide and pay in every condition. Each writes DIR/NAME_<prey>-<pred>.npz with per-world counts, so scoring can
bootstrap over worlds and pair conditions world by world.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.checkpoint import load_brains
from agents.networks import init_params
from scripts.eval import EVAL_SEED, load_world_config
from simulations import evo
from simulations import predprey as pp
from simulations.predprey import NUM_ACTIONS
from simulations.worlds import make_net, obs_channels

MODES = {"real": 0, "mute": 1, "shuffle": 2}
DENSITY_CUTS = (1, 3, 6)  # other living grazers within hearing: 0, 1-2, 3-5, 6+
ENERGY_BINS = 3  # thirds of the energy cap
NULLS = 4  # within-stratum permutations of the calls per step, for the information null
RISK = 50  # steps
BOOT = 1000


# ---------------------------------------------------------------- loading


def eval_world(run: Path, eavesdrop: bool, sets=()):
    """The alarm world a run trained in, with reception left to the test: grazers not deaf, hunters hearing or not."""
    from scripts.train import apply_overrides
    world, wcfg = load_world_config(json.loads((run / "config.json").read_text()))
    wcfg = dataclasses.replace(wcfg, prey_deaf=False, eavesdrop=eavesdrop)
    return world, apply_overrides(wcfg, list(sets))


def brain(run: Path, who: str, net, wcfg, ckpt: str = "latest.msgpack"):
    """One species' brain from a run, into this world's shape (5-action brains widened, extra channels zero)."""
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(evo, wcfg))
    prey, pred, meta = load_brains(run / "checkpoints" / ckpt, template)
    return (prey if who == "prey" else pred), meta.get("update")


# ---------------------------------------------------------------- the shuffled-reception test


def strata(cfg, prey: evo.Species) -> jax.Array:
    """(N,) int: each living grazer's stratum, from how many other living grazers are within hearing and its energy.
    Empty slots get their own stratum."""
    d = jnp.abs(prey.pos[:, None, :] - prey.pos[None, :, :])
    d = jnp.minimum(d, cfg.size - d).max(-1)
    n = prey.alive.shape[0]
    dens = ((d <= cfg.view // 2) & prey.alive[None, :] & ~jnp.eye(n, dtype=bool)).sum(1)
    dbin = jnp.searchsorted(jnp.asarray(DENSITY_CUTS), dens, side="right")
    ebin = jnp.clip(jnp.floor(ENERGY_BINS * prey.energy / cfg.prey_energy_cap), 0, ENERGY_BINS - 1).astype(jnp.int32)
    return jnp.where(prey.alive, dbin * ENERGY_BINS + ebin, 99)


def shuffle_within(key, groups, bits) -> jax.Array:
    """bits permuted at random within each group (same count per group). Sorting by group then a random key lists each
    group's members in one block, in both orders, so block positions match."""
    k1, k2 = jax.random.split(key)
    g = groups.astype(jnp.float32)
    o1 = jnp.argsort(g + 0.5 * jax.random.uniform(k1, g.shape))
    o2 = jnp.argsort(g + 0.5 * jax.random.uniform(k2, g.shape))
    return jnp.zeros_like(bits).at[o2].set(bits[o1])


# ---------------------------------------------------------------- rollouts


def make_rollout(cfg, net, episodes: int, n_probes: int):
    """A jitted rollout of `episodes` worlds for cfg.max_steps steps, given (prey params, pred params, stacked probe
    params, key, prey mode, pred mode). Per-world sums come back in a dict, plus per-step slot arrays for the risk."""
    R = cfg.view // 2
    T = cfg.max_steps
    deltas = pp.DELTAS

    def one_world_extras(state, calls_now, prey_a, k):
        """Per-world quantities that need slot-by-slot work: strata, real calls heard, hunter adjacency, null tables,
        and where listeners move relative to the nearest caller."""
        prey, pred = state.prey, state.pred
        n = prey.alive.shape[0]
        st = strata(cfg, prey)
        adj = evo.hunter_adjacent(cfg, prey, pred)
        delta = prey.pos[None, :, :] - prey.pos[:, None, :]  # (listener, caller) offset, wrapped to [-size/2, size/2)
        delta = (delta + cfg.size // 2) % cfg.size - cfg.size // 2
        d = jnp.abs(delta).max(-1)
        callers = state.prey_signal & prey.alive
        hears = (d <= R) & callers[None, :] & ~jnp.eye(n, dtype=bool)
        heard = hears.sum(1)
        listener = prey.alive & ~adj & (heard > 0)
        nearest = jnp.argmin(jnp.where(hears, d, 10 * cfg.size), axis=1)
        off = delta[jnp.arange(n), nearest]  # toward the nearest caller
        mv = deltas[prey_a % NUM_ACTIONS]
        dot = (off * mv).sum(-1)
        away, toward = listener & (dot < 0), listener & (dot > 0)
        cell = adj.astype(jnp.int32) * 2 + calls_now.astype(jnp.int32)
        live = prey.alive.astype(jnp.float32)
        table = jnp.zeros(4, jnp.float32).at[cell].add(live)
        nulls = []
        for i in range(NULLS):
            sc = shuffle_within(jax.random.fold_in(k, i), st, calls_now)
            nulls.append(jnp.zeros(4, jnp.float32).at[adj.astype(jnp.int32) * 2 + sc.astype(jnp.int32)].add(live))
        return adj, listener, table, jnp.stack(nulls), away.sum(), toward.sum(), st

    def run(prey_params, pred_params, probe_params, key, prey_mode, pred_mode):
        k_reset, k_steps = jax.random.split(key)
        state = jax.vmap(partial(evo.reset, cfg))(jax.random.split(k_reset, episodes))
        zero = jnp.zeros(episodes, jnp.float32)
        names = ("grazer_steps", "deaths", "eaten", "births", "calls", "hunter_steps", "catches", "away", "toward",
                 "listener_steps", "nn_sum", "nn_count", "probe_adj_n", "probe_far_n")
        acc0 = {k: zero for k in names}
        acc0.update(table=jnp.zeros((episodes, 4)), nulls=jnp.zeros((episodes, NULLS, 4)),
                    probe_adj=jnp.zeros((episodes, n_probes)), probe_far=jnp.zeros((episodes, n_probes)))

        def body(carry, k):
            state, acc = carry
            kp, kq, kw, ks, kx = jax.random.split(k, 5)
            real = state.prey_signal
            shuffled = jax.vmap(lambda kk, s: shuffle_within(kk, strata(cfg, s.prey), s.prey_signal & s.prey.alive))(
                jax.random.split(ks, episodes), state)
            heard_by = lambda mode: jnp.where(mode == 0, real, jnp.where(mode == 1, jnp.zeros_like(real), shuffled))
            prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state, heard_by(prey_mode), heard_by(pred_mode))
            prey_logits = net.apply(prey_params, prey_obs)[0]
            pred_logits = net.apply(pred_params, pred_obs)[0]
            prey_a = jax.random.categorical(kp, prey_logits)
            pred_a = jax.random.categorical(kq, pred_logits)
            calls_now = (prey_a >= NUM_ACTIONS) & state.prey.alive
            if cfg.oracle_call:  # Stage 21b: the call made is the oracle's, not the brain's
                calls_now = jax.vmap(lambda st: evo.hunter_adjacent(cfg, st.prey, st.pred))(state)
            adj, listener, table, nulls, away, toward, _ = jax.vmap(one_world_extras)(
                state, calls_now, prey_a, jax.random.split(kx, episodes))
            probe_p = jax.vmap(lambda p: jax.nn.softmax(net.apply(p, prey_obs)[0].astype(jnp.float32))[..., NUM_ACTIONS:].sum(-1))(probe_params)
            live = state.prey.alive
            far = live & ~adj
            res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(kw, episodes), state, prey_a, pred_a)
            info = res.info
            n = live.shape[1]
            died = live & (~res.state.prey.alive | (info.prey_parent != jnp.arange(n)[None, :]))
            nn_sum, nn_count = jax.vmap(lambda s: pp.nearest_dist(s.prey, s.prey, cfg.size, exclude_self=True))(state)
            f = lambda x: x.astype(jnp.float32)
            acc = {
                "grazer_steps": acc["grazer_steps"] + f(live.sum(1)),
                "deaths": acc["deaths"] + f(died.sum(1)),
                "eaten": acc["eaten"] + f(info.catches),
                "births": acc["births"] + f(info.prey_births),
                "calls": acc["calls"] + f(calls_now.sum(1)),
                "hunter_steps": acc["hunter_steps"] + f(state.pred.alive.sum(1)),
                "catches": acc["catches"] + f(info.catches),
                "away": acc["away"] + f(away),
                "toward": acc["toward"] + f(toward),
                "listener_steps": acc["listener_steps"] + f(listener.sum(1)),
                "nn_sum": acc["nn_sum"] + f(nn_sum),
                "nn_count": acc["nn_count"] + f(nn_count),
                "table": acc["table"] + table,
                "nulls": acc["nulls"] + nulls,
                "probe_adj": acc["probe_adj"] + jnp.einsum("pen,en->ep", probe_p, f(adj)),
                "probe_far": acc["probe_far"] + jnp.einsum("pen,en->ep", probe_p, f(far)),
                "probe_adj_n": acc["probe_adj_n"] + f(adj.sum(1)),
                "probe_far_n": acc["probe_far_n"] + f(far.sum(1)),
            }
            quiet = live & ~adj & ~listener
            return (res.state, acc), (listener, quiet, far, died)

        (state, acc), slots = jax.lax.scan(body, (state, acc0), jax.random.split(k_steps, T))
        acc["end_prey"] = state.prey.alive.sum(1).astype(jnp.float32)
        acc["end_pred"] = state.pred.alive.sum(1).astype(jnp.float32)
        return acc, slots

    return jax.jit(run)


def risk_counts(listener, quiet, far, died, horizon: int = RISK) -> dict:
    """Per world: grazer-steps of each kind (listener, quiet: neither near a hunter nor hearing a call, far: no hunter
    near) up to T - horizon, and how many of them were followed by that grazer's death within `horizon` steps (the step
    itself included). A slot's occupant lives until it dies, so its next death is that animal's."""
    T, E, N = died.shape
    nxt = np.full((E, N), 1 << 30, np.int64)
    out = {k: np.zeros(E) for k in ("listener_n", "listener_d", "quiet_n", "quiet_d", "far_n", "far_d")}
    for t in range(T - 1, -1, -1):
        nxt = np.where(died[t], t, nxt)
        if t > T - horizon:
            continue
        soon = (nxt - t) < horizon
        for name, mask in (("listener", listener[t]), ("quiet", quiet[t]), ("far", far[t])):
            out[f"{name}_n"] += mask.sum(1)
            out[f"{name}_d"] += (mask & soon).sum(1)
    return out


def cmd_eval(args) -> int:
    grazers, hunters = Path(args.grazers), Path(args.hunters)
    world, cfg = eval_world(Path(args.world_run or args.grazers), args.eavesdrop, args.set)
    cfg = dataclasses.replace(cfg, max_steps=args.steps)
    net = make_net(world, cfg)
    prey_p, prey_u = brain(grazers, "prey", net, cfg, args.grazer_ckpt)
    pred_p, pred_u = brain(hunters, "pred", net, cfg, args.hunter_ckpt)
    probes = [brain(Path(p), "prey", net, cfg)[0] for p in args.probe] or [prey_p]
    probe_stack = jax.tree.map(lambda *x: jnp.stack(x), *probes)
    run = make_rollout(cfg, net, args.episodes, len(probes))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cond in args.conds.split(","):
        pm, qm = cond.split(":")
        acc, slots = run(prey_p, pred_p, probe_stack, jax.random.PRNGKey(args.seed), jnp.int32(MODES[pm]), jnp.int32(MODES[qm]))
        acc = {k: np.asarray(v) for k, v in jax.device_get(acc).items()}
        acc.update(risk_counts(*(np.asarray(x) for x in jax.device_get(slots))))
        meta = {"grazers": grazers.name, "hunters": hunters.name, "grazer_update": prey_u, "hunter_update": pred_u,
                "prey_hear": pm, "pred_hear": qm, "eavesdrop": args.eavesdrop, "seed": args.seed, "episodes": args.episodes,
                "probes": [Path(p).name for p in args.probe], "signal_cost": cfg.signal_cost, "steps": cfg.max_steps}
        path = out / f"{args.tag}_{pm}-{qm}.npz"
        np.savez_compressed(path, meta=json.dumps(meta), **acc)
        s = summary(acc)
        print(json.dumps({"file": str(path), **meta, **{k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v]) for k, v in s.items()}}), flush=True)
    return 0


# ---------------------------------------------------------------- stage 21b: one fixed listener cohort
# Spec docs/specs/2026-11-05-oracle-call-design.md. A reference rollout (real reception) picks listeners at
# branch steps t = 10, 20, ..., 450; from the reference state at t the world is played 50 steps under each reception,
# with the reference's own random stream, so the same animals in the same states are compared.

BRANCH_EVERY = 10
CONDS = ("real", "mute", "shuffle")


def make_step(cfg, net):
    """One step of E worlds, drawing randomness exactly as `make_rollout` does. Returns the next state and per-slot
    masks: listener (by real calls), hunter adjacent, died, and whether a listener moved away from the nearest call."""
    R = cfg.view // 2

    def listeners(state):
        def one(st):
            prey, pred = st.prey, st.pred
            n = prey.alive.shape[0]
            adj = evo.hunter_adjacent(cfg, prey, pred)
            delta = prey.pos[None, :, :] - prey.pos[:, None, :]
            delta = (delta + cfg.size // 2) % cfg.size - cfg.size // 2
            d = jnp.abs(delta).max(-1)
            hears = (d <= R) & (st.prey_signal & prey.alive)[None, :] & ~jnp.eye(n, dtype=bool)
            listener = prey.alive & ~adj & (hears.sum(1) > 0)
            nearest = jnp.argmin(jnp.where(hears, d, 10 * cfg.size), axis=1)
            return listener, adj, delta[jnp.arange(n), nearest]
        return jax.vmap(one)(state)

    def step(prey_params, pred_params, state, k, prey_mode, pred_mode):
        E = state.prey.alive.shape[0]
        kp, kq, kw, ks, kx = jax.random.split(k, 5)
        real = state.prey_signal
        shuffled = jax.vmap(lambda kk, st: shuffle_within(kk, strata(cfg, st.prey), st.prey_signal & st.prey.alive))(
            jax.random.split(ks, E), state)
        heard_by = lambda mode: jnp.where(mode == 0, real, jnp.where(mode == 1, jnp.zeros_like(real), shuffled))
        prey_obs, pred_obs = jax.vmap(partial(evo.observe, cfg))(state, heard_by(prey_mode), heard_by(pred_mode))
        prey_a = jax.random.categorical(kp, net.apply(prey_params, prey_obs)[0])
        pred_a = jax.random.categorical(kq, net.apply(pred_params, pred_obs)[0])
        listener, adj, off = listeners(state)
        away = listener & ((off * pp.DELTAS[prey_a % NUM_ACTIONS]).sum(-1) < 0)
        live = state.prey.alive
        res = jax.vmap(partial(evo.advance, cfg))(jax.random.split(kw, E), state, prey_a, pred_a)
        n = live.shape[1]
        died = live & (~res.state.prey.alive | (res.info.prey_parent != jnp.arange(n)[None, :]))
        return res.state, {"listener": listener, "adj": adj, "died": died, "away": away,
                           "hunters": state.pred.alive.sum(1), "catches": res.info.catches}

    return step


def make_cohort(cfg, net, episodes: int):
    """(reference segment, branch) functions, both jitted.
    segment(prey p, pred p, state, keys (S, 2)): S reference steps; per-step per-world counts.
    branch(prey p, pred p, states (B, E, ...), keys (B, RISK, 2), cohort (B, E, N), prey mode): per branch and world,
    how many of the cohort died within RISK steps and how many moved away from the nearest call on the first step."""
    step = make_step(cfg, net)
    f = lambda x: x.astype(jnp.float32)

    def segment(prey_p, pred_p, state, keys):
        def body(st, k):
            nxt, m = step(prey_p, pred_p, st, k, jnp.int32(0), jnp.int32(0))
            return nxt, {"grazers": f(st.prey.alive.sum(1)), "hunters": f(m["hunters"]), "adj": f(m["adj"].sum(1)),
                         "listeners": f(m["listener"].sum(1)), "deaths": f(m["died"].sum(1)), "catches": f(m["catches"])}
        return jax.lax.scan(body, state, keys)

    def branch(prey_p, pred_p, states, keys, cohort, prey_mode):
        def one(state, ks, coh):
            def body(carry, k):
                st, dead, first = carry
                nxt, m = step(prey_p, pred_p, st, k, prey_mode, jnp.int32(0))
                away = jnp.where(first, (m["away"] & coh).sum(1), 0)
                return (nxt, dead | m["died"], jnp.array(False)), away
            (_, dead, _), away = jax.lax.scan(body, (state, jnp.zeros_like(coh), jnp.array(True)), ks)
            return f((dead & coh).sum(1)), f(away.sum(0))
        return jax.vmap(one)(states, keys, cohort)

    return jax.jit(segment), jax.jit(branch)


def cmd_cohort(args) -> int:
    grazers, hunters = Path(args.grazers), Path(args.hunters)
    world, cfg = eval_world(Path(args.world_run or args.grazers), False, args.set)
    cfg = dataclasses.replace(cfg, max_steps=args.steps)
    net = make_net(world, cfg)
    prey_p, prey_u = brain(grazers, "prey", net, cfg, args.grazer_ckpt)
    pred_p, pred_u = brain(hunters, "pred", net, cfg, args.hunter_ckpt)
    E, T = args.episodes, cfg.max_steps
    segment, branch = make_cohort(cfg, net, E)
    step = make_step(cfg, net)
    k_reset, k_steps = jax.random.split(jax.random.PRNGKey(args.seed))  # the same split as `eval`
    keys = jax.random.split(k_steps, T)
    state = jax.vmap(partial(evo.reset, cfg))(jax.random.split(k_reset, E))
    ts = list(range(BRANCH_EVERY, T - RISK + 1, BRANCH_EVERY))
    saved, ref, t = {}, [], 0
    for stop in ts + [T]:  # the reference, saving the state at each branch step
        if stop > t:
            state, per = segment(prey_p, pred_p, state, keys[t:stop])
            ref.append(jax.device_get(per))
            t = stop
        if stop in ts:
            saved[stop] = state
    ref = {k: np.concatenate([np.asarray(r[k]) for r in ref], 0) for k in ref[0]}  # (T, E)
    listen = jax.jit(lambda st: make_step_listeners(cfg, st))
    out = {"t": np.array(ts), **{f"ref_{k}": v for k, v in ref.items()}}
    n = np.zeros((E, len(ts)))
    hunters_t = np.zeros((E, len(ts)))
    res = {c: (np.zeros((E, len(ts))), np.zeros((E, len(ts)))) for c in CONDS}
    for i in range(0, len(ts), args.chunk):
        idx = list(range(i, min(i + args.chunk, len(ts))))
        sts = jax.tree.map(lambda *x: jnp.stack(x), *[saved[ts[j]] for j in idx])
        bkeys = jnp.stack([keys[ts[j]:ts[j] + RISK] for j in idx])
        coh = jax.vmap(listen)(sts)
        n[:, idx] = np.asarray(coh.sum(-1)).T
        hunters_t[:, idx] = np.asarray(sts.pred.alive.sum(-1)).T
        for c in CONDS:
            dead, away = branch(prey_p, pred_p, sts, bkeys, coh, jnp.int32(MODES[c]))
            res[c][0][:, idx] = np.asarray(dead).T
            res[c][1][:, idx] = np.asarray(away).T
    out.update(cohort_n=n, hunters_t=hunters_t)
    for c in CONDS:
        out[f"dead_{c}"], out[f"away_{c}"] = res[c]
    meta = {"grazers": grazers.name, "hunters": hunters.name, "grazer_update": prey_u, "hunter_update": pred_u,
            "seed": args.seed, "episodes": E, "steps": T, "risk": RISK, "branch_every": BRANCH_EVERY,
            "oracle_call": cfg.oracle_call, "signal_cost": cfg.signal_cost, "kin_share": cfg.kin_share}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.tag}_cohort.npz"
    np.savez_compressed(path, meta=json.dumps(meta), **out)
    s = cohort_summary(out)
    print(json.dumps({"file": str(path), **meta, **{k: round(v, 5) for k, v in s.items()}}), flush=True)
    return 0


def make_step_listeners(cfg, state):
    """The cohort at a branch step: living grazers with no hunter within 1 cell that hear a real call. (E, N) bool."""
    R = cfg.view // 2

    def one(st):
        prey = st.prey
        n = prey.alive.shape[0]
        adj = evo.hunter_adjacent(cfg, prey, st.pred)
        d = jnp.abs(prey.pos[None, :, :] - prey.pos[:, None, :])
        d = jnp.minimum(d, cfg.size - d).max(-1)
        hears = (d <= R) & (st.prey_signal & prey.alive)[None, :] & ~jnp.eye(n, dtype=bool)
        return prey.alive & ~adj & (hears.sum(1) > 0)
    return jax.vmap(one)(state)


def cohort_mask(d, which: str = "all"):
    """(E, B) bool: the branch steps a summary counts. all; early (t <= 150); hunters (a hunter alive at t)."""
    m = np.ones_like(d["cohort_n"], bool)
    if which == "early":
        m &= (np.asarray(d["t"]) <= 150)[None, :]
    elif which == "hunters":
        m &= d["hunters_t"] > 0
    return m


def stat_cohort_ratio(d, a: str, b: str, which: str = "all"):
    """Fixed-cohort 50-step death risk under reception a over reception b (same cohort, so a ratio of deaths)."""
    m = cohort_mask(d, which)
    da, db = (d[f"dead_{a}"] * m).sum(1), (d[f"dead_{b}"] * m).sum(1)

    def f(idx):
        x, y = da[idx].sum(), db[idx].sum()
        return x / y if y > 0 else np.nan
    return f


def cohort_summary(d, which: str = "all") -> dict:
    m = cohort_mask(d, which)
    n = float((d["cohort_n"] * m).sum())
    out = {"cohort": n}
    for c in CONDS:
        out[f"risk_{c}"] = _div((d[f"dead_{c}"] * m).sum(), n)
        out[f"away_{c}"] = _div((d[f"away_{c}"] * m).sum(), n)
    out["real_over_muted"] = _div(out["risk_real"], out["risk_mute"])
    out["real_over_shuffled"] = _div(out["risk_real"], out["risk_shuffle"])
    return out


def prevalence(d, bin_steps: int = 50) -> list:
    """Danger prevalence in the reference rollout, per bin of steps: hunters alive per world, the share of grazer-steps
    with a hunter adjacent (the call rate under the oracle), listener-steps per world and deaths per grazer-step."""
    T = d["ref_grazers"].shape[0]
    rows = []
    for a in range(0, T, bin_steps):
        sl = slice(a, a + bin_steps)
        g, steps = d["ref_grazers"][sl].sum(), d["ref_grazers"][sl].shape[0]
        rows.append({"steps": f"{a}-{a + bin_steps - 1}",
                     "hunters_per_world": float(d["ref_hunters"][sl].mean()),
                     "adjacent_share": _div(d["ref_adj"][sl].sum(), g),
                     "listener_steps_per_world": float(d["ref_listeners"][sl].sum(0).mean()),
                     "deaths_per_grazer_step": _div(d["ref_deaths"][sl].sum(), g),
                     "catches_per_world_step": float(d["ref_catches"][sl].sum() / (d["ref_catches"].shape[1] * steps))})
    return rows


def score_O(d, seed, opp, spread):
    """Stage 21b's one bar on O-s<seed> against opp, plus the reported windows and stage 21's measure if present."""
    c = load(Path(d) / f"O-s{seed}_vs_{opp}_cohort.npz")
    n = c["cohort_n"].shape[0]
    est = {w: {**cohort_summary(c, w),
               "real_over_muted": _est(stat_cohort_ratio(c, "real", "mute", w), n),
               "real_over_shuffled": _est(stat_cohort_ratio(c, "real", "shuffle", w), n)}
           for w in ("all", "early", "hunters")}
    verdicts = [verdict(est["all"]["real_over_muted"], 0.9, "below", 1.0, spread.get("cohort_mute", 0)),
                verdict(est["all"]["real_over_shuffled"], 0.9, "below", 1.0, spread.get("cohort_shuffle", 0))]
    out = {"bar": {"verdicts": verdicts, **est["all"]}, "early": est["early"], "hunters_present": est["hunters"],
           "prevalence": prevalence(c)}
    files = [Path(d) / f"O-s{seed}_vs_{opp}_{x}.npz" for x in ("real-real", "mute-real", "shuffle-real")]
    if all(p.exists() for p in files):  # stage 21's measure, each condition in its own rollout
        a1, m1, h1 = (load(p) for p in files)
        out["stage21_measure"] = {
            "real_over_muted": _est(stat_ratio(a1, "listener_d", "listener_n", m1, "listener_d", "listener_n"), n),
            "real_over_shuffled": _est(stat_ratio(a1, "listener_d", "listener_n", h1, "listener_d", "listener_n"), n),
            "as_trained": summary(a1), "muted": summary(m1), "shuffled": summary(h1)}
    return out


# ---------------------------------------------------------------- statistics


def mutual_information(table) -> float:
    """Bits, from a 2 x 2 table of counts flattened as [adj * 2 + call]."""
    t = np.asarray(table, np.float64).reshape(2, 2)
    n = t.sum()
    if n == 0:
        return 0.0
    p = t / n
    outer = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    nz = p > 0
    return float(max((p[nz] * np.log2(p[nz] / outer[nz])).sum(), 0.0))


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def summary(d, idx=None) -> dict:
    """Pooled numbers over worlds `idx` (all by default) from one condition's per-world counts."""
    g = (lambda k: d[k].sum(0)) if idx is None else (lambda k: d[k][idx].sum(0))
    t = g("table")
    nulls = g("nulls")
    p_adj = _div(t[3], t[2] + t[3])
    p_none = _div(t[1], t[0] + t[1])
    mi = mutual_information(t)
    mi_null = float(np.mean([mutual_information(nulls[i]) for i in range(nulls.shape[0])]))
    return {
        "mi_bits": mi, "mi_null": mi_null, "mi_excess": mi - mi_null,
        "p_call_adj": p_adj, "p_call_none": p_none, "call_ratio": _div(p_adj, p_none),
        "call_rate": _div(g("calls"), g("grazer_steps")),
        "deaths_per_grazer_step": _div(g("deaths"), g("grazer_steps")),
        "eaten_per_grazer_step": _div(g("eaten"), g("grazer_steps")),
        "births_per_grazer_step": _div(g("births"), g("grazer_steps")),
        "starved_per_grazer_step": _div(g("deaths") - g("eaten"), g("grazer_steps")),
        "catches_per_hunter_step": _div(g("catches"), g("hunter_steps")),
        "catches": float(g("catches")),
        "listener_risk": _div(g("listener_d"), g("listener_n")),
        "quiet_risk": _div(g("quiet_d"), g("quiet_n")),
        "far_risk": _div(g("far_d"), g("far_n")),
        "listener_steps": float(g("listener_n")),
        "away_share": _div(g("away"), g("listener_steps")), "toward_share": _div(g("toward"), g("listener_steps")),
        "nn_dist": _div(g("nn_sum"), g("nn_count")),
        "probe_call_adj": [_div(a, g("probe_adj_n")) for a in np.atleast_1d(g("probe_adj"))],
        "probe_call_far": [_div(a, g("probe_far_n")) for a in np.atleast_1d(g("probe_far"))],
        "prey_extinct": float((d["end_prey"] if idx is None else d["end_prey"][idx]).__eq__(0).mean()),
        "pred_extinct": float((d["end_pred"] if idx is None else d["end_pred"][idx]).__eq__(0).mean()),
    }


def load(path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def boot(stat, n_worlds: int, b: int = BOOT, seed: int = 0):
    """stat(idx) -> float; point estimate on all worlds and a 95% percentile interval over worlds resampled together
    (the same resampled worlds in every condition, since conditions share starting states)."""
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n_worlds))
    draws = np.array([stat(rng.integers(0, n_worlds, n_worlds)) for _ in range(b)])
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws.size else (np.nan, np.nan))
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def stat_ratio(d1, num1, den1, d2, num2, den2):
    """(sum num1 / sum den1 in d1) / (sum num2 / sum den2 in d2), over worlds idx."""
    def f(idx):
        a = d1[num1][idx].sum() / max(d1[den1][idx].sum(), 1e-12)
        b = d2[num2][idx].sum() / max(d2[den2][idx].sum(), 1e-12)
        return a / b if b > 0 else np.nan
    return f


def stat_mi_excess(d):
    def f(idx):
        t = d["table"][idx].sum(0)
        nulls = d["nulls"][idx].sum(0)
        return mutual_information(t) - np.mean([mutual_information(x) for x in nulls])
    return f


def stat_call_ratio(d):
    def f(idx):
        t = d["table"][idx].sum(0)
        a, b = t[3] / max(t[2] + t[3], 1e-12), t[1] / max(t[0] + t[1], 1e-12)
        return a / b if b > 0 else np.nan
    return f


def stat_probe_ratio(d, i, j):
    """Danger-conditioned calling of probe i over probe j, on the same states."""
    def f(idx):
        a, b = d["probe_adj"][idx, i].sum(), d["probe_adj"][idx, j].sum()
        return a / b if b > 0 else np.nan
    return f


def verdict(est, threshold, direction, null, spread=0.0):
    """Pass: the point at or beyond the threshold, and the 95% interval entirely on the effect's side of `null`
    (1 for ratios, 0 for differences). A miss within 2 eval spreads of the threshold is 'no result', not a fail."""
    p, lo, hi = est["point"], est["lo"], est["hi"]
    if direction == "above":
        side, ok_ci, close = p >= threshold, lo > null, threshold - p <= 2 * spread
    else:
        side, ok_ci, close = p <= threshold, hi < null, p - threshold <= 2 * spread
    if side and ok_ci:
        return "pass"
    if not side and close:
        return "no result"
    return "fail"


# ---------------------------------------------------------------- clips


def cmd_clip(args) -> int:
    """One world, played twice from the same saved starting state and random stream: as trained, and with reception
    muted (--mute prey or pred). Writes the episode3d format plus each step's callers, and the moments the spec names."""
    from scripts.episode3d import Tracker, _bits, _bytes
    grazers, hunters = Path(args.grazers), Path(args.hunters)
    world, cfg = eval_world(Path(args.world_run or args.grazers), args.eavesdrop, args.set)
    net = make_net(world, cfg)
    prey_p, _ = brain(grazers, "prey", net, cfg)
    pred_p, _ = brain(hunters, "pred", net, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def step(state, k, prey_mode, pred_mode):
        kp, kq, kw = jax.random.split(k, 3)
        real = state.prey_signal
        zero = jnp.zeros_like(real)
        prey_obs, pred_obs = evo.observe(cfg, state, jnp.where(prey_mode == 1, zero, real), jnp.where(pred_mode == 1, zero, real))
        prey_a = jax.random.categorical(kp, net.apply(prey_p, prey_obs)[0])
        pred_a = jax.random.categorical(kq, net.apply(pred_p, pred_obs)[0])
        res = evo.advance(cfg, kw, state, prey_a, pred_a)
        return res.state, prey_a, pred_a, res.info.catches

    results = {}
    for label, modes in (("trained", (0, 0)), (f"muted_{args.mute}", (1, 0) if args.mute == "prey" else (0, 1))):
        k_reset, key = jax.random.split(jax.random.PRNGKey(args.seed))
        state = evo.reset(cfg, k_reset)
        animals: dict = {}
        prey_t, pred_t = Tracker("prey", animals), Tracker("pred", animals)
        steps, frames = [], []
        for t, k in enumerate(jax.random.split(key, cfg.max_steps)):
            s = jax.device_get(state)
            prey_rows, pred_rows = prey_t.frame(t, s.prey), pred_t.frame(t, s.pred)
            state, prey_a, pred_a, caught = step(state, k, jnp.int32(modes[0]), jnp.int32(modes[1]))
            prey_a, pred_a = np.asarray(prey_a), np.asarray(pred_a)
            uid_of = {slot: uid for slot, uid in prey_t.slot_id.items()}
            calls = [uid_of[i] for i in np.nonzero((prey_a >= NUM_ACTIONS) & np.asarray(s.prey.alive))[0] if i in uid_of]
            row = {"prey": prey_rows, "pred": pred_rows, "plant": _bits(np.asarray(s.plant)), "call": calls}
            if t % 50 == 0:
                row["tough"] = _bytes(np.asarray(s.plant_genes)[..., 1])
            steps.append(row)
            frames.append({"pos": np.asarray(s.prey.pos), "alive": np.asarray(s.prey.alive), "act": prey_a,
                           "pred_pos": np.asarray(s.pred.pos), "pred_alive": np.asarray(s.pred.alive), "pred_act": pred_a,
                           "pred_energy": np.asarray(s.pred.energy),
                           "uid": dict(uid_of), "pred_uid": dict(pred_t.slot_id)})
            if not (np.asarray(state.prey.alive).any() and np.asarray(state.pred.alive).any()):
                break
        for a in animals.values():
            a["died"] = len(steps) if a["died"] is None else a["died"]
        episode = {"size": cfg.size, "steps": steps, "animals": animals, "terrain": _bytes(np.zeros((cfg.size, cfg.size))),
                   "flank": False, "pack_size": 0, "stealth": int(cfg.stealth_radius), "climate": "", "winter_len": 0,
                   "world": "alarm", "grazers": grazers.name, "hunters": hunters.name, "seed": args.seed, "condition": label,
                   "eavesdrop": args.eavesdrop, "calls_per_step": "step.call lists the ids of grazers that called on that step"}
        results[label] = (episode, frames)

    moments = find_moments(cfg, results["trained"][1], results["trained"][0]["animals"])
    for label, (episode, _) in results.items():
        episode["moments"] = moments
        path = out_dir / f"{args.tag}_{label}.json"
        path.write_text(json.dumps(episode, separators=(",", ":")))
        print(f"{path}: {len(episode['steps'])} steps, {len(moments['warning'])} warning moments, "
              f"{len(moments['eavesdrop'])} eavesdrop moments", flush=True)
    return 0


def find_moments(cfg, frames, animals) -> dict:
    """Spec moments in a trained episode. warning: a grazer next to a hunter calls at step t, and a grazer 2 to 4
    cells from it, with no hunter within a cell, moves away from it at t + 1. eavesdrop: a hunter with no grazer within
    its sight (3) moves toward a grazer that called last step 4 cells away, and catches within 5 steps."""
    size, R = cfg.size, cfg.view // 2

    def wrap(v):
        return (v + size // 2) % size - size // 2

    warning, eaves = [], []
    for t in range(len(frames) - 6):
        f, g = frames[t], frames[t + 1]
        pos, alive, act = f["pos"], f["alive"], f["act"]
        qpos, qalive = f["pred_pos"], f["pred_alive"]
        callers = np.nonzero(alive & (act >= NUM_ACTIONS))[0]
        for i in callers:
            dq = np.abs(wrap(qpos - pos[i])).max(-1)
            if not (qalive & (dq <= 1)).any():
                continue
            for j in np.nonzero(g["alive"])[0]:
                if j == i or j not in f["uid"] or f["uid"].get(j) != g["uid"].get(j):
                    continue
                off = wrap(g["pos"][i] - g["pos"][j]) if g["alive"][i] else wrap(pos[i] - g["pos"][j])
                dist = np.abs(off).max()
                if not 2 <= dist <= R:
                    continue
                if (g["pred_alive"] & (np.abs(wrap(g["pred_pos"] - g["pos"][j])).max(-1) <= 1)).any():
                    continue
                mv = pp.DELTAS[int(g["act"][j]) % NUM_ACTIONS]
                if int((np.asarray(mv) * off).sum()) < 0:
                    warning.append({"step": t, "caller": f["uid"].get(int(i)), "listener": g["uid"].get(int(j)), "distance": int(dist)})
        sig = alive & (act >= NUM_ACTIONS)  # calls made at t, heard at t + 1
        for h in np.nonzero(g["pred_alive"])[0]:
            dg = np.abs(wrap(g["pos"] - g["pred_pos"][h])).max(-1)
            if (g["alive"] & (dg <= 3)).any():
                continue
            loud = np.nonzero(sig & g["alive"] & (dg == R))[0]
            if not loud.size:
                continue
            mv = np.asarray(pp.DELTAS[int(g["pred_act"][h]) % NUM_ACTIONS])
            off = wrap(g["pos"][loud[0]] - g["pred_pos"][h])
            if int((mv * off).sum()) <= 0:
                continue
            uid = g["pred_uid"].get(int(h))
            if any(_ate(frames, t + 1 + k, h, uid) for k in range(5)):
                eaves.append({"step": t + 1, "hunter": uid, "grazer": g["uid"].get(int(loud[0])), "distance": int(R)})
    return {"warning": warning[:200], "eavesdrop": eaves[:200]}


def _ate(frames, t, h, uid) -> bool:
    """The same hunter (slot h, id uid) gained more than a step's upkeep between steps t and t + 1: it ate."""
    if t + 1 >= len(frames):
        return False
    a, b = frames[t], frames[t + 1]
    if a["pred_uid"].get(int(h)) != uid or b["pred_uid"].get(int(h)) != uid:
        return False
    return bool(b["pred_energy"][h] - a["pred_energy"][h] > 5)


# ---------------------------------------------------------------- scoring
# File names: DIR/<tag>_<prey>-<pred>.npz from `eval`. Tags used by the stage:
#   A: A{0,1,2}-s<seed>_vs_<opp>   (conds real:real, mute:real, shuffle:real; A2's as-trained condition is mute:real)
#   B: B{1,0}-s<seed>_vs_<opp>     (B1 with --eavesdrop: real:real, real:mute, real:shuffle; B0 deaf: real:real)
#   C: C{1,0}-s<seed>_vs_<opp> and A1-s<seed>_vs_<opp> (all --eavesdrop real:real; the A1 file carries probes C1, C0)


def _get(d, tag, cond):
    return load(Path(d) / f"{tag}_{cond}.npz")


def _est(stat, n):
    return boot(stat, n)


def score_A(d, seed, opp, spread):
    a1, m1, h1 = (_get(d, f"A1-s{seed}_vs_{opp}", c) for c in ("real-real", "mute-real", "shuffle-real"))
    a2 = _get(d, f"A2-s{seed}_vs_{opp}", "mute-real")
    a0 = _get(d, f"A0-s{seed}_vs_{opp}", "real-real")
    n = len(a1["grazer_steps"])
    s1 = summary(a1)
    mi = {"bits": s1["mi_bits"], "null": s1["mi_null"], "excess": _est(stat_mi_excess(a1), n)}
    bar1a = "pass" if (mi["bits"] >= 0.05 and mi["excess"]["lo"] > 0) else (
        "no result" if 0.05 - mi["bits"] <= 2 * spread.get("mi_bits", 0) else "fail")
    ratio_ = _est(stat_call_ratio(a1), n)
    bar1b = verdict(ratio_, 2.0, "above", 1.0, spread.get("call_ratio", 0))
    rm = _est(stat_ratio(a1, "listener_d", "listener_n", m1, "listener_d", "listener_n"), n)
    rs = _est(stat_ratio(a1, "listener_d", "listener_n", h1, "listener_d", "listener_n"), n)
    r2 = _est(stat_ratio(a1, "deaths", "grazer_steps", a2, "deaths", "grazer_steps"), n)
    bar2 = [verdict(rm, 0.9, "below", 1.0, spread.get("risk_mute", 0)), verdict(rs, 0.9, "below", 1.0, spread.get("risk_shuffle", 0)),
            verdict(r2, 0.9, "below", 1.0, spread.get("deaths_ratio", 0))]
    return {
        "bar1": {"mi": mi, "mi_verdict": bar1a, "call_ratio": ratio_, "call_ratio_verdict": bar1b,
                 "p_call_adj": s1["p_call_adj"], "p_call_none": s1["p_call_none"]},
        "bar2": {"risk_real_over_muted": rm, "risk_real_over_shuffled": rs, "A1_deaths_over_A2": r2, "verdicts": bar2,
                 "listener_risk": {"real": s1["listener_risk"], "muted": summary(m1)["listener_risk"], "shuffled": summary(h1)["listener_risk"]}},
        "reported": {name: summary(x) for name, x in (("A1", a1), ("A1_muted", m1), ("A1_shuffled", h1), ("A2", a2), ("A0", a0))},
    }


def score_B(d, seed, opp, spread):
    b1, b1m, b1s = (_get(d, f"B1-s{seed}_vs_{opp}", c) for c in ("real-real", "real-mute", "real-shuffle"))
    b0 = _get(d, f"B0-s{seed}_vs_{opp}", "real-real")
    n = len(b1["grazer_steps"])
    c = ("catches", "hunter_steps")
    gain = _est(stat_ratio(b1, *c, b0, *c), n)
    mute = _est(stat_ratio(b1m, *c, b1, *c), n)
    shuf = _est(stat_ratio(b1s, *c, b1, *c), n)
    return {"bar3": {"B1_over_B0": gain, "muted_over_real": mute, "shuffled_over_real": shuf,
                     "verdicts": [verdict(gain, 1.15, "above", 1.0, spread.get("catch_ratio", 0)),
                                  verdict(mute, 0.9, "below", 1.0, spread.get("catch_ratio", 0)),
                                  verdict(shuf, 0.9, "below", 1.0, spread.get("catch_ratio", 0))]},
            "reported": {name: summary(x) for name, x in (("B1", b1), ("B1_muted", b1m), ("B1_shuffled", b1s), ("B0", b0))}}


def score_C(d, seed, opp, spread):
    c1, c0, a1 = (_get(d, f"{t}-s{seed}_vs_{opp}", "real-real") for t in ("C1", "C0", "A1"))
    n = len(c1["grazer_steps"])
    probe = _est(stat_probe_ratio(a1, 0, 1), n)
    onpol = _est(lambda idx: stat_call_ratio_adj(c1, idx) / stat_call_ratio_adj(c0, idx), n)
    deaths = _est(stat_ratio(c1, "deaths", "grazer_steps", a1, "deaths", "grazer_steps"), n)
    sa = summary(a1)
    return {"bar4": {"danger_calling_C1_over_C0_same_states": probe, "danger_calling_C1_over_C0_own_rollouts": onpol,
                     "C1_deaths_over_A1": deaths,
                     "probe_p_call_adj": {"C1": sa["probe_call_adj"][0], "C0": sa["probe_call_adj"][1]},
                     "probe_p_call_far": {"C1": sa["probe_call_far"][0], "C0": sa["probe_call_far"][1]},
                     "verdicts": [verdict(probe, 0.8, "below", 1.0, spread.get("probe_ratio", 0)),
                                  verdict(deaths, 0.9, "below", 1.0, spread.get("deaths_ratio", 0))]},
            "reported": {name: summary(x) for name, x in (("C1", c1), ("C0", c0), ("A1", a1))}}


def stat_call_ratio_adj(d, idx):
    t = d["table"][idx].sum(0)
    return t[3] / max(t[2] + t[3], 1e-12)


def cmd_score(args) -> int:
    spread = json.loads(Path(args.spread).read_text()) if args.spread else {}
    fn = {"A": score_A, "B": score_B, "C": score_C, "O": score_O}[args.phase]
    out = {"phase": args.phase, "opp": args.opp, "spread": spread, "seeds": {}}
    for seed in args.seeds:
        opp = args.opp.format(seed=seed)
        out["seeds"][str(seed)] = fn(args.dir, seed, opp, spread)
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "clip", "cohort"):
        q = sub.add_parser(name)
        q.add_argument("--grazers", required=True)
        q.add_argument("--hunters", required=True)
        q.add_argument("--world-run", default=None, help="run whose world config to test in (default: --grazers)")
        q.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
        q.add_argument("--eavesdrop", action="store_true")
        q.add_argument("--out", required=True)
        q.add_argument("--tag", required=True)
        q.add_argument("--seed", type=int, default=EVAL_SEED)
    e = sub.choices["eval"]
    e.add_argument("--conds", default="real:real")
    e.add_argument("--probe", action="append", default=[])
    e.add_argument("--episodes", type=int, default=256)
    e.add_argument("--steps", type=int, default=500)
    e.add_argument("--grazer-ckpt", default="latest.msgpack")
    e.add_argument("--hunter-ckpt", default="latest.msgpack")
    co = sub.choices["cohort"]  # Stage 21b: one fixed listener cohort, branched under each reception
    co.add_argument("--episodes", type=int, default=256)
    co.add_argument("--steps", type=int, default=500)
    co.add_argument("--chunk", type=int, default=3, help="branch steps played at once (x episodes worlds)")
    co.add_argument("--grazer-ckpt", default="latest.msgpack")
    co.add_argument("--hunter-ckpt", default="latest.msgpack")
    c = sub.choices["clip"]
    c.add_argument("--mute", choices=("prey", "pred"), default="prey")
    sc = sub.add_parser("score")
    sc.add_argument("--phase", choices=("A", "B", "C", "O"), required=True)
    sc.add_argument("--dir", required=True)
    sc.add_argument("--seeds", type=int, nargs="+", required=True)
    sc.add_argument("--opp", required=True, help="opponent tag, e.g. st0 or A1-s{seed}")
    sc.add_argument("--spread", default=None, help="JSON of eval spreads per statistic")
    sc.add_argument("--out", default=None)
    args = p.parse_args(argv)
    return {"eval": cmd_eval, "clip": cmd_clip, "cohort": cmd_cohort, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
