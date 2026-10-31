"""Two runs compared at matched world-steps:
python -m scripts.compare --run runs/pp-mem-001 --baseline runs/pp-mem-ff

The window is the last third of the shorter run's world-steps. Every checkpoint either run saved
inside it is evaluated with its own prey against its own predators (fixed seed), and the
results are averaged per run. Pass: the run's prey live >= 1.2x as long as the baseline's, or its
predators catch >= 1.2x as often. Written to <run>/compare.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jax
import numpy as np

from agents.checkpoint import load_brains
from agents.evaluate import evaluate
from agents.networks import init_params
from simulations.worlds import make_net, obs_channels
from agents.policies import network_policy
from scripts.eval import EVAL_SEED, load_world_config

TARGET = 1.2
KEYS = ("prey_lifetime", "catches_per_pred_step")


def steps_by_update(run_dir: Path) -> dict[int, int]:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return {int(r["update"]): int(r["env_steps"]) for r in rows}


def window(run_dirs: list[Path]) -> tuple[int, int]:
    hi = min(max(steps_by_update(d).values()) for d in run_dirs)
    return hi * 2 // 3, hi


def checkpoints_in(run_dir: Path, lo: int, hi: int) -> list[Path]:
    steps = steps_by_update(run_dir)
    out = []
    for c in sorted((run_dir / "checkpoints").glob("ckpt_*.msgpack")):
        s = steps.get(int(re.search(r"(\d+)", c.name).group(1)))
        if s is not None and lo <= s <= hi:
            out.append(c)
    return out


def verdict(run: dict, baseline: dict) -> dict:
    prey = run["prey_lifetime"] / max(baseline["prey_lifetime"], 1e-9)
    catch = run["catches_per_pred_step"] / max(baseline["catches_per_pred_step"], 1e-9)
    return {"prey_ratio": prey, "catch_ratio": catch, "pass": bool(prey >= TARGET or catch >= TARGET)}


def evaluate_run(run_dir: Path, ckpts: list[Path], episodes: int) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    results = []
    for c in ckpts:
        prey, pred, _ = load_brains(c, template)
        results.append(evaluate(wcfg, network_policy(net, prey), network_policy(net, pred),
                                jax.random.PRNGKey(EVAL_SEED), episodes, world))
    return {k: float(np.mean([r[k] for r in results])) for k in KEYS}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--episodes", type=int, default=128)
    args = p.parse_args(argv)

    run, base = Path(args.run), Path(args.baseline)
    lo, hi = window([run, base])
    run_ckpts, base_ckpts = checkpoints_in(run, lo, hi), checkpoints_in(base, lo, hi)
    if not run_ckpts or not base_ckpts:
        print(f"no checkpoints in the window {lo:,}..{hi:,} world-steps (run {len(run_ckpts)}, "
              f"baseline {len(base_ckpts)})")
        return 2
    run_eval, base_eval = evaluate_run(run, run_ckpts, args.episodes), evaluate_run(base, base_ckpts, args.episodes)
    out = {"baseline": str(base), "window": [lo, hi], "episodes": args.episodes,
           "run_checkpoints": [c.name for c in run_ckpts], "baseline_checkpoints": [c.name for c in base_ckpts],
           "run_eval": run_eval, "baseline_eval": base_eval, **verdict(run_eval, base_eval)}
    (run / "compare.json").write_text(json.dumps(out, indent=1))
    print(f"window {lo:,}..{hi:,} world-steps; {len(run_ckpts)} vs {len(base_ckpts)} checkpoints")
    print(f"prey lifetime {run_eval['prey_lifetime']:.1f} vs {base_eval['prey_lifetime']:.1f} "
          f"({out['prey_ratio']:.2f}x); catches/pred-step {run_eval['catches_per_pred_step']:.4f} vs "
          f"{base_eval['catches_per_pred_step']:.4f} ({out['catch_ratio']:.2f}x); target {TARGET}x -> "
          f"{'PASS' if out['pass'] else 'FAIL'}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
