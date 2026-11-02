"""Hunters from one shared-brain run against grazers from another: python -m scripts.cross_shared --hunters runs/A --grazers runs/B [--random]

Each side brings its own genes and brain, which may differ in size. --random: hunters act at random, with --hunters' genes.
Written to <grazers>/crossshared_<hunters>[_random].json. Spec: docs/specs/2026-11-02-mealworm-design.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from agents.checkpoint import load_brains
from agents.evaluate import evaluate
from agents.networks import init_params
from agents.policies import network_policy, random_policy
from scripts.eval import EVAL_SEED, load_world_config
from scripts.league import evolved_pool
from simulations.worlds import make_net, obs_channels


def brains(run: Path):
    config = json.loads((run / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey, pred = load_brains(run / "checkpoints" / "latest.msgpack", template)[:2]
    return world, wcfg, network_policy(net, prey), network_policy(net, pred)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hunters", required=True)
    p.add_argument("--grazers", required=True)
    p.add_argument("--random", action="store_true")
    p.add_argument("--episodes", type=int, default=64)
    args = p.parse_args(argv)
    hunters, grazers = Path(args.hunters), Path(args.grazers)
    world, wcfg, prey_pi, _ = brains(grazers)
    _, _, _, pred_pi = brains(hunters)
    pool_h, _ = evolved_pool(hunters)
    pool_g, _ = evolved_pool(grazers)
    pool = pool_g._replace(pred_genes=pool_h.pred_genes, pred_gen=pool_h.pred_gen, pred_w=pool_h.pred_w)
    r = evaluate(wcfg, prey_pi, random_policy if args.random else pred_pi, jax.random.PRNGKey(EVAL_SEED), args.episodes, world, pool=pool)
    out = {"hunters": hunters.name, "grazers": grazers.name, "random": args.random, "catches_per_hunter_step": float(r["catches_per_pred_step"])}
    suffix = "_random" if args.random else ""
    (grazers / f"crossshared_{hunters.name}{suffix}.json").write_text(json.dumps(out, indent=1))
    print(f"{'random ' if args.random else ''}{hunters.name} against {grazers.name}: {out['catches_per_hunter_step']:.4f} catches per hunter-step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
