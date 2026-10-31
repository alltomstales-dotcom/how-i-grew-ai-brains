#!/usr/bin/env python3
"""Build the static GitHub Pages site into _site/: python tools/build_pages.py [--out DIR]

site/ holds the landing page and the replays. The viewer itself is not copied into the repo twice:
viz/dashboard/world3d.html and its vendor/ folder go into watch/ here, next to the replays and
runs.json it fetches by relative path. No server or private dashboard is involved.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(out: Path, root: Path = ROOT) -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(root / "site", out)
    dash = root / "viz" / "dashboard"
    shutil.copy(dash / "world3d.html", out / "watch" / "index.html")
    shutil.copy(dash / "tri.html", out / "watch" / "tri.html")
    shutil.copytree(dash / "vendor", out / "watch" / "vendor")
    shutil.copy(root / "videos.json", out / "videos.json")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "_site")
    args = p.parse_args(argv)
    print(f"built {build(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
