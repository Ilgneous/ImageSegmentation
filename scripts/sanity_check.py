"""Data sanity checks before training: save show_sample / show_augmented figures for a few
images, and scan every image for size mismatches or unexpected trimap values.

    python scripts/sanity_check.py                       # 10 random train images -> debug/
    python scripts/sanity_check.py --n 25 --split test
    python scripts/sanity_check.py --names Abyssinian_1 beagle_12
    python scripts/sanity_check.py --scan                # also check every image in the split

Run from anywhere. Paths are resolved relative to the repo root.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from src import viz  # noqa: E402
from src.dataset import OxfordPetSeg  # noqa: E402


def scan(ds: OxfordPetSeg) -> list[tuple[str, str]]:
    problems = []
    for i in range(len(ds)):
        img, tri = ds.load_raw(i)
        name = ds.records[i].name
        if img.shape[:2] != tri.shape[:2]:
            problems.append((name, f"size mismatch image {img.shape[:2]} vs trimap {tri.shape[:2]}"))
        values = set(np.unique(tri).tolist())
        if not values <= {1, 2, 3}:
            problems.append((name, f"unexpected trimap values {sorted(values)}"))
        if 1 not in values:
            problems.append((name, "no pet pixels in trimap"))
        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1}/{len(ds)}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "oxford-iiit-pet"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=10, help="random images to plot")
    ap.add_argument("--names", nargs="*", help="plot these images instead of random ones")
    ap.add_argument("--aug", type=int, default=8, help="augmentation passes per image")
    ap.add_argument("--out", default=str(ROOT / "debug"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan", action="store_true", help="check every image in the split")
    a = ap.parse_args()

    ds = OxfordPetSeg(a.root, split=a.split)
    print(f"{a.split}: {len(ds)} images in {a.root}")
    out = Path(a.out)

    idxs = [ds.index_of(n) for n in a.names] if a.names else \
        random.Random(a.seed).sample(range(len(ds)), min(a.n, len(ds)))
    for idx in idxs:
        name = ds.records[idx].name
        viz.save_figure(viz.show_sample(ds, idx), out / f"{name}_sample.png")
        viz.save_figure(viz.show_augmented(ds, idx, n=a.aug), out / f"{name}_augmented.png")
        print(f"  wrote {name}")
    print(f"figures in {out}")

    if a.scan:
        print(f"scanning all {len(ds)} images ...")
        problems = scan(ds)
        if problems:
            report = out / f"problems_{a.split}.txt"
            report.write_text("\n".join(f"{n}\t{p}" for n, p in problems) + "\n")
            print(f"{len(problems)} problems, listed in {report}:")
            for n, p in problems[:10]:
                print(f"  {n}: {p}")
            print("look at any of them with: python scripts/sanity_check.py --names <name>")
        else:
            print("no problems found")


if __name__ == "__main__":
    main()
