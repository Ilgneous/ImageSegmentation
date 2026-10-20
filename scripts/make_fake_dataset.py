"""Write a tiny synthetic dataset in the exact Oxford-IIIT Pet layout.

Used by the tests and for smoke-testing the whole pipeline (train, evaluate, viz, app)
in seconds on a CPU, before spending GPU time on the real data.

    python scripts/make_fake_dataset.py --out data/fake-pet --n 120

Each "pet" is a textured blob (an ellipse plus a head and ears) on a cluttered background.
Trimaps follow Oxford's convention: 1 pet, 2 background, 3 a few-pixel boundary band.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# (breed, species_id) with Oxford's casing convention: cats capitalized, dogs lowercase.
BREEDS = [("Abyssinian", 1), ("Bengal", 1), ("Persian", 1),
          ("beagle", 2), ("pug", 2), ("samoyed", 2)]


def make_pet(rng: np.random.Generator, h: int, w: int):
    img = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 6)
    for _ in range(6):  # background clutter
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.rectangle(img, p1, p2, c, -1)
    img = cv2.addWeighted(img, 0.6, cv2.GaussianBlur(img, (0, 0), 3), 0.4, 0)

    fg = np.zeros((h, w), np.uint8)
    cx, cy = int(w * rng.uniform(0.35, 0.65)), int(h * rng.uniform(0.4, 0.65))
    ax, ay = int(w * rng.uniform(0.15, 0.3)), int(h * rng.uniform(0.12, 0.25))
    cv2.ellipse(fg, (cx, cy), (ax, ay), float(rng.uniform(-30, 30)), 0, 360, 1, -1)
    hx, hy, hr = cx + int(ax * 0.8), cy - int(ay * 0.9), max(6, int(min(ax, ay) * 0.6))
    cv2.circle(fg, (hx, hy), hr, 1, -1)
    ears = np.array([[hx - hr, hy - hr // 3], [hx - hr // 2, hy - int(hr * 1.6)], [hx, hy - hr // 2]])
    cv2.fillPoly(fg, [ears], 1)
    cv2.fillPoly(fg, [ears + [hr, 0]], 1)

    fur = np.array(rng.integers(40, 230, 3), dtype=np.float32)
    noise = rng.normal(0, 25, size=(h, w, 1)).astype(np.float32)
    pet = np.clip(fur + noise, 0, 255).astype(np.uint8)
    img[fg == 1] = pet[fg == 1]

    k = np.ones((5, 5), np.uint8)
    band = cv2.dilate(fg, k) - cv2.erode(fg, k)
    trimap = np.full((h, w), 2, np.uint8)
    trimap[fg == 1] = 1
    trimap[band == 1] = 3
    return img, trimap


def main(out: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations" / "trimaps").mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        breed, species = BREEDS[i % len(BREEDS)]
        class_id = BREEDS.index((breed, species)) + 1
        name = f"{breed}_{i // len(BREEDS) + 1}"
        h, w = int(rng.integers(220, 400)), int(rng.integers(220, 400))
        img, trimap = make_pet(rng, h, w)
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "annotations" / "trimaps" / f"{name}.png"), trimap)
        lines.append(f"{name} {class_id} {species} {class_id}")
    header = "#Image CLASS-ID SPECIES BREED ID\n"
    split = int(len(lines) * 0.6)
    (out / "annotations" / "trainval.txt").write_text("\n".join(lines[:split]) + "\n")
    (out / "annotations" / "test.txt").write_text("\n".join(lines[split:]) + "\n")
    (out / "annotations" / "list.txt").write_text(header + "\n".join(lines) + "\n")
    print(f"wrote {n} samples to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/fake-pet"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.n, a.seed)
