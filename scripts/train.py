"""One-command training: python -m scripts.train --run-id NAME"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np

from agents.checkpoint import load_brains, save_checkpoint
from agents.policies import network_policy, random_policy
from agents.ppo import TrainConfig, make_trainer, tiny_configs
from simulations.worlds import WORLDS, get_world
from scripts.site import build_site
from viz.render import write_clip

DASHBOARD = Path(__file__).resolve().parents[1] / "viz" / "dashboard" / "index.html"
CLIP_SEED = 2026
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def check_memory(stats, limit_gb: float) -> tuple[bool, str]:
    """Judge jax Device.memory_stats() output against the memory budget."""
    if not stats or "peak_bytes_in_use" not in stats:
        return True, "device memory stats unavailable; relying on XLA_PYTHON_CLIENT_MEM_FRACTION cap"
    peak = stats["peak_bytes_in_use"] / 1e9
    if peak > limit_gb:
        return False, f"peak device memory {peak:.2f} GB exceeds limit {limit_gb} GB"
    return True, f"peak device memory {peak:.2f} GB (limit {limit_gb} GB)"


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def finish(run_dir: Path, reason: str, update: int, env_steps: int, wall_s: float) -> None:
    """Mark the run as ended, so the dashboard stops calling it running."""
    write_json_atomic(run_dir / "done.json", {"reason": reason, "update": update, "env_steps": env_steps,
                                              "wall_s": wall_s, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_runs_index(run_dir.parent)


def write_runs_index(runs_dir: Path) -> None:
    """Refresh the dashboard home page (runs list and roadmap)."""
    build_site(runs_dir)


def apply_overrides(wcfg, pairs: list[str]):
    """FIELD=VALUE strings onto a frozen world config, cast to each field's current type."""
    fields = {f.name for f in __import__("dataclasses").fields(wcfg)}
    changes = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if name not in fields:
            raise ValueError(f"unknown world setting {name!r}")
        kind = type(getattr(wcfg, name))
        changes[name] = (value.lower() in ("1", "true", "yes")) if kind is bool else kind(value)
    return replace(wcfg, **changes)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--updates", type=int, default=100_000)
    p.add_argument("--max-hours", type=float, default=0.25)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--mem-limit-gb", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--world", choices=sorted(WORLDS), default="predprey")
    p.add_argument("--tiny", action="store_true", help="tiny world and batch, for tests")
    p.add_argument("--no-learn", action="store_true", help="control run: brains never learn")
    p.add_argument("--learners", choices=("both", "prey", "pred"), default="both",
                   help="which species learns; the other keeps the brain it starts with")
    p.add_argument("--init-from", default=None, help="checkpoint to start both brains from")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="override a world setting, e.g. --set burrows=64 (repeatable)")
    p.add_argument("--memory", type=int, default=0, help="GRU size for brains with memory (0 = feed-forward)")
    p.add_argument("--league", type=int, default=0, help="past brains kept to train against (0 = none)")
    p.add_argument("--league-frac", type=float, default=0.25, help="worlds per side facing a past brain")
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="share of agent-slots each update learns from, living animals first")
    p.add_argument("--no-clips", action="store_true", help="skip the checkpoint clips: a 256 x 256 map holds about 10 GB of frames for each")
    p.add_argument("--live-every", type=int, default=0, help="publish a clip for the phone viewer every this many updates; 0 is off")
    p.add_argument("--live-window", type=int, default=96, help="cells per side of the part of the map a clip shows")
    p.add_argument("--transit-displace", action="store_true", help="a migrant reaching a full world replaces a resident")
    p.add_argument("--min-update-seconds", type=float, default=0.0, help="wait so an update takes at least this long, to keep trainers on different machines in step")
    p.add_argument("--transit-dir", help="Stage 11a: folder for animals crossing between continents")
    p.add_argument("--transit-me", type=int, default=0, help="this continent's number")
    p.add_argument("--transit-to", type=int, default=0, help="the continent to the east")
    p.add_argument("--transit-prob", type=float, default=0.2, help="chance per update that an animal at the east edge crosses")
    p.add_argument("--transit-strip", type=int, default=2, help="width in cells of the edge animals cross from and arrive at")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not RUN_ID.fullmatch(args.run_id):
        print(f"invalid --run-id {args.run_id!r}: use letters, digits, '-' and '_'", file=sys.stderr)
        return 2
    world, config_cls = get_world(args.world)
    if args.tiny:
        wcfg, tcfg = tiny_configs(args.world)
    else:
        wcfg, tcfg = config_cls(), TrainConfig()
    try:
        wcfg = apply_overrides(wcfg, args.set)
    except (ValueError, TypeError) as e:
        print(f"--set: {e}", file=sys.stderr)
        return 2
    tcfg = replace(tcfg, seed=args.seed, learn=not args.no_learn, learners=args.learners, sample_frac=args.sample_frac,
                   league_size=args.league, league_frac=args.league_frac, memory=args.memory)
    if args.num_envs:
        tcfg = replace(tcfg, num_envs=args.num_envs)

    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_id
    if run_dir.exists():
        print(f"run directory {run_dir} already exists; pick another --run-id", file=sys.stderr)
        return 2
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "clips").mkdir()
    shutil.copy(DASHBOARD, run_dir / "index.html")
    write_json_atomic(run_dir / "config.json", {"world_name": args.world, "world": asdict(wcfg),
                                                "train": asdict(tcfg), "args": vars(args)})
    write_runs_index(runs_dir)
    print(f"run {run_dir}  devices {jax.devices()}", flush=True)

    clips: list[dict] = []

    evolving = hasattr(world, "gene_pool")

    def add_clip(name, update, env_steps, prey_pi, pred_pi, label, init_state=None):
        if args.no_clips:
            return
        frames = write_clip(wcfg, prey_pi, pred_pi, jax.random.PRNGKey(CLIP_SEED),
                            run_dir / "clips" / name, label=label, world=world, init_state=init_state)
        clips.append({"file": f"clips/{name}", "update": update, "env_steps": env_steps,
                      "frames": frames, "label": label})
        write_json_atomic(run_dir / "clips.json", clips)

    add_clip("random.mp4", 0, 0, random_policy, random_policy, "random policies")

    trainer = make_trainer(wcfg, tcfg, world=world)
    runner = trainer.init(jax.random.PRNGKey(tcfg.seed))
    if args.init_from:  # carried-over brains; the optimiser starts fresh
        prey_p, pred_p, meta = load_brains(args.init_from, runner.prey.params)
        runner = runner._replace(prey=runner.prey._replace(params=prey_p),
                                 pred=runner.pred._replace(params=pred_p))
        print(f"brains from {args.init_from} (update {meta.get('update')})", flush=True)
    steps_per_update = tcfg.num_envs * tcfg.rollout_len
    crossing = None
    if args.transit_dir:  # Stage 11a: animals cross to the next continent between updates
        from scripts.transit import Transit, log_row
        crossing = Transit(args.transit_dir, args.transit_me, args.transit_to, args.transit_prob, args.transit_strip, wcfg, world, tcfg.seed, args.transit_displace)
        crossing_log = open(run_dir / "transit.jsonl", "a")
    feed = None
    if args.live_every:  # short clips for the phone viewer, while this run trains
        from scripts.live import LiveFeed
        feed = LiveFeed(run_dir, wcfg, world, trainer.net, args.live_every, seed=tcfg.seed, window=args.live_window)
    start = time.time()
    env_steps = 0
    reason, update = "updates", 0
    with open(run_dir / "metrics.jsonl", "a") as log:
        for update in range(1, args.updates + 1):
            t0 = time.time()
            try:
                runner, metrics = trainer.update(runner)
                # Dispatch is async, so an OOM inside update() can surface here.
                metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            except Exception as e:  # XLA raises its own runtime error type for OOM
                if "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                print(f"out of device memory at update {update}; try a smaller --num-envs: {e}",
                      file=sys.stderr, flush=True)
                finish(run_dir, "out_of_memory", update - 1, env_steps, time.time() - start)
                return 3
            dt = max(time.time() - t0, 1e-9)
            env_steps += steps_per_update
            row = {"update": update, "env_steps": env_steps, "sps": steps_per_update / dt,
                   "wall_s": time.time() - start, **metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if crossing is not None:
                runner, crossed = crossing.exchange(runner, update)
                log_row(crossing_log, update, crossed)
            if feed is not None:
                try:
                    feed.publish(runner, update, crossed if crossing is not None else None)
                except OSError as e:  # a viewer's feed is never worth stopping a run for
                    print(f"live feed: {e}", file=sys.stderr, flush=True)

            if update == 1:
                ok, msg = check_memory(jax.devices()[0].memory_stats(), args.mem_limit_gb)
                print(msg, flush=True)
                if not ok:
                    finish(run_dir, "memory_limit", update, env_steps, time.time() - start)
                    return 3

            reason = "updates" if update == args.updates else (
                "max_hours" if time.time() - start > args.max_hours * 3600 else None)
            last = reason is not None
            if update % args.ckpt_every == 0 or last:
                ckpt = run_dir / "checkpoints" / f"ckpt_{update:06d}.msgpack"
                meta = {"update": update, "env_steps": env_steps}
                save_checkpoint(ckpt, runner.prey.params, runner.pred.params, meta)
                runner = trainer.add_to_league(runner)  # a no-op without --league
                shutil.copy(ckpt, run_dir / "checkpoints" / "latest.msgpack")
                clip_start = None
                if evolving:
                    pool = world.gene_pool(runner.env_state)
                    genes = run_dir / "checkpoints" / f"genes_{update:06d}.npz"
                    np.savez_compressed(genes, **world.pool_to_arrays(jax.device_get(pool), seed=update))
                    shutil.copy(genes, run_dir / "checkpoints" / "genes_latest.npz")
                    clip_start = world.reset(wcfg, jax.random.PRNGKey(CLIP_SEED), pool)
                add_clip(f"clip_{update:06d}.mp4", update, env_steps,
                         network_policy(trainer.net, runner.prey.params),
                         network_policy(trainer.net, runner.pred.params), f"update {update}", clip_start)
                write_runs_index(runs_dir)
                print(f"update {update}  env_steps {env_steps:,}  sps {row['sps']:,.0f}  "
                      f"prey {metrics['prey_alive']:.1f}  pred {metrics['pred_alive']:.1f}  "
                      f"catches/step {metrics['catches_per_step']:.3f}", flush=True)
            if last:
                break
            if args.min_update_seconds:
                time.sleep(max(0.0, args.min_update_seconds - (time.time() - t0)))
    finish(run_dir, reason, update, env_steps, time.time() - start)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
