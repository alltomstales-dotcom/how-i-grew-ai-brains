"""Learned predators against a hand-written chase rule: python -m scripts.chase --run-dir runs/NAME

The chase rule steps toward the nearest prey it can see. Both kinds of predator play the run's
latest prey brain and random prey (fixed seed, founder bodies). If the chaser catches far more
often than the learned predators, the learned ones stopped short of what the world allows; if not,
the predators are near the world's ceiling. Written to <run>/chase.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from agents.checkpoint import load_brains
from agents.evaluate import evaluate
from agents.networks import init_params
from simulations.worlds import make_net, obs_channels
from agents.policies import chase_policy, network_policy, random_policy
from scripts.eval import EVAL_SEED, load_world_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--evolved", action="store_true",
                   help="start from the run's latest evolved bodies (genes snapshot) instead of founders")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(run_dir / "checkpoints" / "latest.msgpack", template)
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    chaser = chase_policy(world.PREY_CHANNEL)
    key = jax.random.PRNGKey(EVAL_SEED)
    pool, bodies = None, "founders"
    if args.evolved:
        from scripts.league import evolved_pool
        pool, bodies = evolved_pool(run_dir)
        if pool is None:
            print(f"no gene snapshots in {run_dir}")
            return 1

    def catches(prey, pred):
        return evaluate(wcfg, prey, pred, key, args.episodes, world, pool=pool)["catches_per_pred_step"]

    out = {
        "update": meta.get("update"),
        "bodies": bodies,
        "chase_vs_random_prey": catches(random_policy, chaser),
        "trained_vs_random_prey": catches(random_policy, pred_pi),
        "chase_vs_trained_prey": catches(prey_pi, chaser),
        "trained_vs_trained_prey": catches(prey_pi, pred_pi),
    }
    out["ratio_vs_random_prey"] = out["chase_vs_random_prey"] / max(out["trained_vs_random_prey"], 1e-9)
    out["ratio_vs_trained_prey"] = out["chase_vs_trained_prey"] / max(out["trained_vs_trained_prey"], 1e-9)
    (run_dir / ("chase_evolved.json" if args.evolved else "chase.json")).write_text(json.dumps(out, indent=1))
    print(f"bodies: {bodies}")
    for who in ("random", "trained"):
        print(f"vs {who} prey: chaser {out[f'chase_vs_{who}_prey']:.4f}, learned {out[f'trained_vs_{who}_prey']:.4f} "
              f"catches per predator-step (chaser {out[f'ratio_vs_{who}_prey']:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
