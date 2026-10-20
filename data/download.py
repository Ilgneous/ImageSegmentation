"""Download the Oxford-IIIT Pet dataset into data/oxford-iiit-pet/.

Default source is torchvision's downloader (official VGG tarballs, md5-checked).
Kaggle is supported for mirrors that keep the original layout:

    python data/download.py
    python data/download.py --source kaggle --slug <owner>/<dataset>

Afterwards the folder must contain images/, annotations/trimaps/, annotations/trainval.txt
and annotations/test.txt. The script verifies that and prints split sizes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "oxford-iiit-pet"


def from_torchvision() -> None:
    from torchvision.datasets import OxfordIIITPet

    # torchvision writes to <root>/oxford-iiit-pet/{images,annotations}
    OxfordIIITPet(root=str(ROOT), split="trainval", target_types="segmentation", download=True)


def from_kaggle(slug: str) -> None:
    import kagglehub  # pip install kagglehub, and put your API token in ~/.kaggle/kaggle.json

    path = Path(kagglehub.dataset_download(slug))
    # Mirrors nest the folders differently, so search for the two folders we need.
    images = next(p for p in path.rglob("images") if p.is_dir() and any(p.glob("*.jpg")))
    annotations = next(p for p in path.rglob("annotations") if (p / "trimaps").is_dir())
    DEST.mkdir(parents=True, exist_ok=True)
    for src, name in ((images, "images"), (annotations, "annotations")):
        if not (DEST / name).exists():
            shutil.copytree(src, DEST / name)


def verify() -> None:
    need = [DEST / "images", DEST / "annotations" / "trimaps",
            DEST / "annotations" / "trainval.txt", DEST / "annotations" / "test.txt"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit("Dataset incomplete, missing:\n  " + "\n  ".join(missing))
    sys.path.insert(0, str(ROOT.parent))
    from src.dataset import make_splits

    splits = make_splits(DEST)
    n_img = len(list((DEST / "images").glob("*.jpg")))
    print(f"OK: {n_img} jpgs in {DEST}")
    print("split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["torchvision", "kaggle"], default="torchvision")
    ap.add_argument("--slug", help="Kaggle dataset slug, required with --source kaggle")
    args = ap.parse_args()
    if args.source == "kaggle":
        if not args.slug:
            ap.error("--slug is required with --source kaggle")
        from_kaggle(args.slug)
    else:
        from_torchvision()
    verify()
