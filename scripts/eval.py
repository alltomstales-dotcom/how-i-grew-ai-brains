"""Trained-vs-random evaluation: python -m scripts.eval --run-dir runs/<id>"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import numpy as np

from agents.checkpoint import load_brains
from agents.evaluate import burrow_use, evaluate, lineage_share
from agents.networks import init_params
from agents.policies import network_policy, random_policy
from simulations.worlds import get_world, make_net, obs_channels

EVAL_SEED = 12345
PREDATOR_TARGET = 2.0  # trained predators catch >= 2x as often as random ones (vs random prey)
PREY_TARGET = 1.5  # trained prey live >= 1.5x as long as random prey (vs trained predators)
EVOLVED_TARGET = 0.60  # evolved bodies hold >= 60% of living-agent-steps against founders (prey or predators)
PLANT_TARGET = 0.15  # at least one plant gene's mean moved this far from the founders


def load_world_config(config: dict):
    """The run's world module and config. JSON turns tuples into lists; turn them back."""
    world, config_cls = get_world(config.get("world_name", "predprey"))
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in config["world"].items()}
    if "founder_animal" in fields:  # runs saved before the size or brain gene get founder values for them
        from simulations.evo import ANIMAL_GENES, FOUNDER_SIZE
        f = fields["founder_animal"]
        fields["founder_animal"] = tuple(f) + (FOUNDER_SIZE,) * (ANIMAL_GENES - len(f))
    # Fields newer than the run take the default of the oldest class that declares them (a new
    # flag's "off"), not the registered world's, so a run evaluates as it was trained.
    family = [c for c in reversed(config_cls.__mro__) if dataclasses.is_dataclass(c)]
    for name in {f.name for f in dataclasses.fields(config_cls)} - fields.keys():
        oldest = next(c for c in family if name in {f.name for f in dataclasses.fields(c)})
        default = next(f.default for f in dataclasses.fields(oldest) if f.name == name)
        if default is not dataclasses.MISSING:
            fields[name] = default
    return world, config_cls(**fields)


BURROW_TARGET = 1.5  # trained prey hide under threat, and hide selectively, 1.5x as much as random prey


def burrow_verdict(trained: dict, random: dict) -> bool:
    """Trained prey must hide when threatened more than random prey do, and hide
    selectively (near / far), not just sit in burrows all the time."""
    if trained["near"] is None or trained["ratio"] is None:
        return False
    r_near = random["near"] or 0.0
    r_ratio = random["ratio"] if random["ratio"] is not None else 1.0
    return bool(trained["near"] >= BURROW_TARGET * max(r_near, 1e-3) and trained["ratio"] >= BURROW_TARGET * r_ratio)


def evolved_verdict(share: dict) -> bool:
    """Pass when any species with data reaches the target (None = it died out before scoring)."""
    return any(v is not None and v >= EVOLVED_TARGET for v in share.values())


def _share(v) -> str:
    return "none survived" if v is None else f"{v:.2f}"


def evolution_result(world, wcfg, prey_pi, pred_pi, key, episodes, genes_path) -> dict:
    with np.load(genes_path) as d:
        arrays = {k: d[k] for k in d.files}
    pool = world.pool_from_arrays(arrays)
    share = lineage_share(wcfg, prey_pi, pred_pi, key, episodes, pool)
    # One evolved gene at a time: shows which trait carries the advantage.
    single = {name: lineage_share(wcfg, prey_pi, pred_pi, key, episodes, pool, genes=(i,))
              for i, name in enumerate(world.GENE_NAMES["animal"])}
    plants = arrays["plant_genes"].reshape(-1, len(wcfg.founder_plant))
    shift = {name: float(abs(plants[:, i].mean() - wcfg.founder_plant[i])) if len(plants) else 0.0
             for i, name in enumerate(world.GENE_NAMES["plant"])}
    animals = {sp: {name: float(arrays[f"{sp}_genes"][:, i].mean()) if len(arrays[f"{sp}_genes"]) else None
                    for i, name in enumerate(world.GENE_NAMES["animal"])} for sp in ("prey", "pred")}
    return {
        "genes": str(genes_path),
        "prey_evolved_share": share["prey"],
        "pred_evolved_share": share["pred"],
        "single_gene_share": single,
        "plant_shift": shift,
        "evolved_gene_means": animals,
        "evolved_pass": evolved_verdict(share),
        "plant_pass": bool(max(shift.values()) >= PLANT_TARGET),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=256)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    world, wcfg = load_world_config(json.loads((run_dir / "config.json").read_text()))
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "latest.msgpack"
    net = make_net(world, wcfg, json.loads((run_dir / "config.json").read_text())["train"].get("memory", 0))
    template = init_params(net, jax.random.PRNGKey(0), wcfg.view, obs_channels(world, wcfg))
    prey_params, pred_params, meta = load_brains(ckpt, template)  # pads brains saved before terrain
    prey_pi, pred_pi = network_policy(net, prey_params), network_policy(net, pred_params)
    key = jax.random.PRNGKey(EVAL_SEED)

    rr = evaluate(wcfg, random_policy, random_policy, key, args.episodes, world)
    rt = evaluate(wcfg, random_policy, pred_pi, key, args.episodes, world)
    tt = evaluate(wcfg, prey_pi, pred_pi, key, args.episodes, world)
    predator_ratio = rt["catches_per_pred_step"] / max(rr["catches_per_pred_step"], 1e-9)
    prey_ratio = tt["prey_lifetime"] / max(rt["prey_lifetime"], 1e-9)
    result = {
        "checkpoint": str(ckpt),
        "meta": meta,
        "episodes": args.episodes,
        "random_vs_random": rr,
        "random_prey_vs_trained_pred": rt,
        "trained_vs_trained": tt,
        "predator_ratio": predator_ratio,
        "prey_ratio": prey_ratio,
        "predator_pass": bool(predator_ratio >= PREDATOR_TARGET),
        "prey_pass": bool(prey_ratio >= PREY_TARGET),
    }
    if hasattr(world, "gene_pool"):
        result["brain_ratios_note"] = ("predator_ratio and prey_ratio use founder bodies, not the evolved "
                                       "ones the brains trained with; not comparable with predprey runs")
        matched = ckpt.parent / ckpt.name.replace("ckpt_", "genes_").replace(".msgpack", ".npz")
        genes = matched if matched.exists() else ckpt.parent / "genes_latest.npz"
        result.update(evolution_result(world, wcfg, prey_pi, pred_pi, key, args.episodes, genes))
    if getattr(wcfg, "burrows", 0):
        trained = burrow_use(wcfg, prey_pi, pred_pi, key, args.episodes)
        rand = burrow_use(wcfg, random_policy, pred_pi, key, args.episodes)
        result["burrow"] = {"trained": trained, "random": rand}
        result["burrow_pass"] = burrow_verdict(trained, rand)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=1))
    if "burrow" in result:
        f = lambda v: "-" if v is None else f"{v:.3f}"
        for who in ("trained", "random"):
            b = result["burrow"][who]
            print(f"burrows ({who} prey): in a burrow {f(b['near'])} when a predator is near, "
                  f"{f(b['far'])} when not (ratio {f(b['ratio'])})")
        print(f"burrow test (target {BURROW_TARGET}x random) -> {'PASS' if result['burrow_pass'] else 'FAIL'}")
    if "evolved_pass" in result:
        print(f"evolved bodies: prey {_share(result['prey_evolved_share'])}, predators "
              f"{_share(result['pred_evolved_share'])} share vs founders (target {EVOLVED_TARGET}) -> "
              f"{'PASS' if result['evolved_pass'] else 'FAIL'}")
        for name, sh in result["single_gene_share"].items():
            print(f"  {name} alone: prey {_share(sh['prey'])}, predators {_share(sh['pred'])}")
        shift = ", ".join(f"{k} {v:.2f}" for k, v in result["plant_shift"].items())
        print(f"plants: gene shift {shift} (target {PLANT_TARGET}) -> {'PASS' if result['plant_pass'] else 'FAIL'}")
        return 0 if result["evolved_pass"] and result["plant_pass"] and result.get("burrow_pass", True) else 1
    print(f"predators: {predator_ratio:.2f}x random catch rate "
          f"(target {PREDATOR_TARGET}x) -> {'PASS' if result['predator_pass'] else 'FAIL'}")
    print(f"prey: {prey_ratio:.2f}x random lifetime vs trained predators "
          f"(target {PREY_TARGET}x) -> {'PASS' if result['prey_pass'] else 'FAIL'}")
    return 0 if result["predator_pass"] and result["prey_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
