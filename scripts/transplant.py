"""Are brains better at their own climate? python -m scripts.transplant --target runs/A --same runs/B --other runs/C --own-other ...

Every evaluation uses the target run's world, gene pool and partner brain; only the focal species' brain changes. --same is a
foreign run from the target's climate, --other a foreign run from another climate, so both are foreign and climate is the
difference. Written to <target>/transplant.json. Spec: docs/specs/2026-11-01-transplant-design.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from agents.checkpoint import load_brains
from agents.evaluate import evaluate
from agents.networks import init_params
from agents.policies import network_policy
from scripts.eval import EVAL_SEED, load_world_config
from scripts.league import evolved_pool
from simulations.worlds import make_net, obs_channels


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--same", required=True, help="a foreign run from the target's climate")
    p.add_argument("--other", required=True, help="a foreign run from another climate")
    p.add_argument("--episodes", type=int, default=32)
    args = p.parse_args(argv)

    target = Path(args.target)
    config = json.loads((target / "config.json").read_text())
    world, wcfg = load_world_config(config)
    net = make_net(world, wcfg, config["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    brains = lambda run: load_brains(Path(run) / "checkpoints" / "latest.msgpack", template)[:2]
    home_prey, home_pred = brains(target)
    same_prey, same_pred = brains(args.same)
    other_prey, other_pred = brains(args.other)
    pool, bodies = evolved_pool(target)
    if pool is None:
        print(f"no gene snapshots in {target}")
        return 1
    key = jax.random.PRNGKey(EVAL_SEED)

    def run(prey, pred):
        r = evaluate(wcfg, network_policy(net, prey), network_policy(net, pred), key, args.episodes, world, pool=pool)
        return {k: float(r[k]) for k in ("catches_per_pred_step", "prey_lifetime")}

    out = {"target": target.name, "same": Path(args.same).name, "other": Path(args.other).name, "bodies": bodies,
           "home": run(home_prey, home_pred),
           "hunter_same": run(home_prey, same_pred), "hunter_other": run(home_prey, other_pred),
           "grazer_same": run(same_prey, home_pred), "grazer_other": run(other_prey, home_pred)}
    catches = lambda k: out[k]["catches_per_pred_step"]
    out["hunter_ratio"] = catches("hunter_same") / max(catches("hunter_other"), 1e-9)
    out["grazer_ratio"] = catches("grazer_other") / max(catches("grazer_same"), 1e-9)  # hunters catch fewer of the better grazer
    (target / "transplant.json").write_text(json.dumps(out, indent=1))
    print(f"{target.name}: hunters home {catches('home'):.4f}, same-climate {catches('hunter_same'):.4f}, other-climate {catches('hunter_other'):.4f}, ratio {out['hunter_ratio']:.2f}x")
    print(f"{target.name}: grazers same-climate {catches('grazer_same'):.4f}, other-climate {catches('grazer_other'):.4f} against them, ratio {out['grazer_ratio']:.2f}x; lifetime {out['grazer_same']['prey_lifetime']:.1f}, {out['grazer_other']['prey_lifetime']:.1f}, home {out['home']['prey_lifetime']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
