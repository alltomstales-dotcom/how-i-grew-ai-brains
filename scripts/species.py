"""Has a species split in two by size? python -m scripts.species --run-dir runs/NAME

For each saved gene snapshot (checkpoints/genes_*.npz) and each animal species, a two-Gaussian
mixture is fitted to the size gene. It gives Ashman's D, D = sqrt(2) |m1 - m2| / sqrt(s1^2 + s2^2),
the smaller group's share and the gap between the group means. (Hard 2-means clusters alone would
score a single bell curve above D = 2.) Each group's spread is floored at one mutation step: resets
copy genomes from the pool, so a snapshot holds clumps of clones, and a curve fitted to one clump
would otherwise be needle-thin. 2-means on all four genes is reported alongside, to show which gene
differs most. Pass: in each of the last three snapshots some species has D >= 2, a minority share
>= 0.2 and a size gap >= 0.3 (more than the catch margin, the gap at which size changes who can eat you).
Written to <run>/species.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from simulations.evo import ANIMAL_GENES, GENE_NAMES, pad_genes

SEPARATION_TARGET = 2.0
MINORITY_TARGET = 0.2
GAP_TARGET = 0.3  # above catch_margin (0.25): a no-learning probe reached 0.24 by chance (see spec)
DIET_SEPARATION, DIET_GAP = 4.0, 0.6  # stage 9d: drift alone reached D 2.66 and gap 0.49 on the diet gene (see spec)
SD_FLOOR = 0.05  # EvoConfig.gene_sigma: one mutation step
MIN_ANIMALS = 20
LAST = 3


def two_means(x: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Centres (2, d) and labels; starts from the two points farthest apart along the main axis."""
    centred = x - x.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    proj = centred @ axis
    c = np.stack([x[proj.argmin()], x[proj.argmax()]])
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), 1)
        new = np.stack([x[labels == k].mean(0) if (labels == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    return c, labels


def mixture_1d(z: np.ndarray, labels: np.ndarray, floor: float = SD_FLOOR, iters: int = 200):
    """EM for two Gaussians on z, started from a split given as labels. Returns (means, sds, weights)."""
    m = np.array([z[labels == k].mean() if (labels == k).any() else z.mean() for k in range(2)])
    s = np.array([max(z[labels == k].std() if (labels == k).sum() > 1 else z.std(), floor) for k in range(2)])
    w = np.array([(labels == k).mean() for k in range(2)]).clip(1e-3, 1)
    for _ in range(iters):
        dens = w / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((z[:, None] - m) / s) ** 2)
        r = dens / np.maximum(dens.sum(1, keepdims=True), 1e-300)
        n = r.sum(0) + 1e-12
        w = n / len(z)
        m = (r * z[:, None]).sum(0) / n
        s = np.maximum(np.sqrt((r * (z[:, None] - m) ** 2).sum(0) / n), floor)
    return m, s, w


def split(genes: np.ndarray, gene: str = "size") -> dict:
    x = np.asarray(genes, np.float64)
    c, _ = two_means(x)
    size = x[:, GENE_NAMES["animal"].index(gene)]
    m, s, w = mixture_1d(size, (size > size.mean()).astype(int))
    return {
        "animals": len(x),
        "separation": float(np.sqrt(2) * abs(m[1] - m[0]) / np.sqrt(s[0] ** 2 + s[1] ** 2)),
        "minority": float(w.min()),
        "size_gap": float(abs(m[1] - m[0])),
        "size_means": sorted(m.round(3).tolist()),
        "main_gene": GENE_NAMES["animal"][int(np.abs(c[1] - c[0]).argmax())],
        "centres": c.round(3).tolist(),
    }


def snapshot(path: Path, gene: str = "size") -> dict:
    out = {"update": int(re.search(r"(\d+)", path.name).group(1)), "prey": None, "pred": None}
    with np.load(path) as d:
        for name in ("prey", "pred"):
            if gene == "diet" and name == "prey":  # only hunters use the diet gene; in grazers it just drifts
                continue
            genes = pad_genes(d[f"{name}_genes"], ANIMAL_GENES)
            out[name] = split(genes, gene) if len(genes) >= MIN_ANIMALS else None
    return out


def split_found(s: dict | None, gene: str = "size") -> bool:
    sep, gap = (DIET_SEPARATION, DIET_GAP) if gene == "diet" else (SEPARATION_TARGET, GAP_TARGET)
    return bool(s and s["separation"] >= sep and s["minority"] >= MINORITY_TARGET and s["size_gap"] >= gap)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gene", default="size", choices=("size", "diet"), help="diet: hunters only (stage 9d)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    paths = sorted((run_dir / "checkpoints").glob("genes_[0-9]*.npz"),
                   key=lambda q: int(re.search(r"(\d+)", q.name).group(1)))
    if not paths:
        print(f"no gene snapshots in {run_dir}")
        return 2
    snaps = [snapshot(q, args.gene) for q in paths]
    last = snaps[-LAST:]
    passed = len(last) == LAST and all(split_found(s["prey"], args.gene) or split_found(s["pred"], args.gene) for s in last)
    (run_dir / ("species.json" if args.gene == "size" else f"species_{args.gene}.json")).write_text(json.dumps({"snapshots": snaps, "pass": passed}, indent=1))
    f = lambda s: "-" if s is None else (f"{args.gene} D {s['separation']:.2f} minority {s['minority']:.2f} "
                                         f"gap {s['size_gap']:.2f}; most different gene {s['main_gene']}")
    for s in snaps:
        print(f"u{s['update']}: prey {f(s['prey'])}; predators {f(s['pred'])}")
    sep, gap = (DIET_SEPARATION, DIET_GAP) if args.gene == "diet" else (SEPARATION_TARGET, GAP_TARGET)
    print(f"{args.gene} split (D >= {sep}, minority >= {MINORITY_TARGET}, gap >= {gap}, "
          f"last {LAST} snapshots) "
          f"-> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
