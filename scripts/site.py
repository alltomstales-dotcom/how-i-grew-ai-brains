"""Build the dashboard home page: python -m scripts.site --runs-dir runs [--every 60]

Copies the home page and roadmap into the runs directory and writes runs.json,
a small summary of every run (last metrics row, eval verdicts, latest clip), so
the home page never has to download a whole metrics file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DASH_DIR = Path(__file__).resolve().parents[1] / "viz" / "dashboard"
HOME = DASH_DIR / "home.html"
ROADMAP = DASH_DIR / "roadmap.json"
VIEWER = DASH_DIR / "world3d.html"  # the 3-D replay, with its three.js files in vendor/
TRI = DASH_DIR / "tri.html"  # three replays side by side
LIVE = DASH_DIR / "live.html"  # the phone viewer of runs in progress
RUNNING_WITHIN_S = 15 * 60  # a run whose metrics changed this recently counts as running
LAST_KEYS = ("update", "env_steps", "wall_s", "sps", "prey_alive", "pred_alive", "catches_per_step",
             "plant_cover", "prey_speed", "prey_sight", "prey_breed", "pred_speed", "pred_sight",
             "pred_breed", "prey_size", "pred_size", "prey_brain", "pred_brain", "plant_spread", "plant_tough", "prey_gen", "pred_gen",
             "burrow_use_near", "burrow_use_far")
EVAL_KEYS = ("predator_ratio", "prey_ratio", "predator_pass", "prey_pass", "prey_evolved_share",
             "pred_evolved_share", "single_gene_share", "evolved_pass", "plant_shift", "plant_pass",
             "burrow", "burrow_pass")


def last_row(path: Path, chunk: int = 65536):
    """The last complete JSON line of a JSONL file, read from its tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - chunk, 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_summary(run_dir: Path, now: float) -> dict:
    config = _read_json(run_dir / "config.json", {})
    metrics = run_dir / "metrics.jsonl"
    last = last_row(metrics) or {}
    updated = metrics.stat().st_mtime if metrics.exists() else (run_dir / "config.json").stat().st_mtime
    clips = _read_json(run_dir / "clips.json", [])
    evaluation = _read_json(run_dir / "eval.json")
    league = _read_json(run_dir / "league.json")
    species = _read_json(run_dir / "species.json")
    compare = _read_json(run_dir / "compare.json")
    chase, chase_evolved = (_read_json(run_dir / f) for f in ("chase.json", "chase_evolved.json"))
    signals, signals_evolved = (_read_json(run_dir / f) for f in ("signals.json", "signals_evolved.json"))
    return {
        "id": run_dir.name,
        "world": config.get("world_name", "predprey"),
        "started": _iso((run_dir / "config.json").stat().st_mtime),
        "updated": _iso(updated),
        "status": "ended" if (run_dir / "done.json").exists() or now - updated >= RUNNING_WITHIN_S else "running",
        "learn": config.get("train", {}).get("learn", True),
        "done": _read_json(run_dir / "done.json"),
        "last": {k: last[k] for k in LAST_KEYS if k in last},
        "eval": {k: evaluation[k] for k in EVAL_KEYS if k in evaluation} if evaluation else None,
        "latest_clip": clips[-1]["file"] if clips else None,
        "has_3d": (run_dir / "episode3d.json").exists(),
        "league": {"pred_gain": league["pred_elo"][-1] - league["pred_elo"][0],
                   "prey_gain": league["prey_elo"][-1] - league["prey_elo"][0],
                   "pred_progress": league["pred_progress"], "prey_progress": league["prey_progress"]}
        if league else None,
        "species": {"pass": species["pass"], "prey": species["snapshots"][-1]["prey"],
                    "pred": species["snapshots"][-1]["pred"]} if species and species.get("snapshots") else None,
        "compare": {"baseline": Path(compare["baseline"]).name,
                    **{k: compare[k] for k in ("prey_ratio", "catch_ratio", "pass")}} if compare else None,
        "chase": _chase_summary(chase),
        "chase_evolved": _chase_summary(chase_evolved),
        "signals": _signal_summary(signals),
        "signals_evolved": _signal_summary(signals_evolved),
    }


def _chase_summary(chase):
    return {k: chase[k] for k in ("ratio_vs_trained_prey", "ratio_vs_random_prey")} if chase else None


def _signal_summary(signals):
    if not signals:
        return None
    return {"drop": signals["drop"], "pass": signals["pass"],
            **{f"{who}_{k}": signals[who][k] for who in ("prey", "pred") for k in ("rate", "mi_bits")}}


def build_site(runs_dir: Path) -> dict:
    runs_dir = Path(runs_dir)
    now = time.time()
    runs = [run_summary(p, now) for p in runs_dir.iterdir() if (p / "config.json").exists()]
    runs.sort(key=lambda r: r["started"], reverse=True)
    data = {"generated": _iso(now), "runs": runs}
    tmp = runs_dir / f"runs.json.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(runs_dir / "runs.json")
    for src in (HOME, ROADMAP):
        dst = runs_dir / ("index.html" if src == HOME else src.name)
        part = dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.tmp")  # trainers in containers can share this folder
        shutil.copy(src, part)
        part.replace(dst)
    for src in (VIEWER, TRI, LIVE, *sorted((DASH_DIR / "vendor").glob("*.js"))):
        dst = runs_dir / src.relative_to(DASH_DIR)
        if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
            continue  # vendor files are large and rarely change
        dst.parent.mkdir(exist_ok=True)
        part = dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.tmp")  # trainers in containers can share this folder
        shutil.copy(src, part)
        part.replace(dst)
    return data


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--every", type=float, default=0, help="rebuild every N seconds (0 = once)")
    args = p.parse_args(argv)
    while True:
        if not args.every:
            build_site(Path(args.runs_dir))
            return 0
        try:
            build_site(Path(args.runs_dir))
        except Exception as e:  # keep the loop alive; a run directory may be half-written
            print(f"site build failed: {e}", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
