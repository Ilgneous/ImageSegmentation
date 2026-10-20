"""Oxford-IIIT Pet segmentation dataset.

Expected layout (what torchvision and the official tarballs produce):

    <root>/images/Abyssinian_1.jpg
    <root>/annotations/trimaps/Abyssinian_1.png      values: 1 pet, 2 background, 3 boundary
    <root>/annotations/trainval.txt                  "Image CLASS-ID SPECIES BREED-ID" per line
    <root>/annotations/test.txt

Masks used everywhere else in the repo are uint8 with three values:
    0 = background, 1 = pet, IGNORE_INDEX (255) = excluded from loss and metric.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRIMAP_PET, TRIMAP_BG, TRIMAP_BOUNDARY = 1, 2, 3
BOUNDARY_MODES = ("ignore", "fg", "bg")
SPECIES_NAMES = {1: "cat", 2: "dog"}


# --------------------------------------------------------------------------- trimaps
def convert_trimap(trimap: np.ndarray, boundary_mode: str = "ignore") -> np.ndarray:
    """Oxford trimap (1/2/3) -> training mask (0/1/255).

    boundary_mode decides what happens to class 3 (the uncertain outline):
        ignore: excluded from loss and metric (default, most defensible)
        fg:     counted as pet
        bg:     counted as background
    """
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}")
    mask = np.zeros(trimap.shape, dtype=np.uint8)
    mask[trimap == TRIMAP_PET] = 1
    boundary = trimap == TRIMAP_BOUNDARY
    mask[boundary] = {"ignore": IGNORE_INDEX, "fg": 1, "bg": 0}[boundary_mode]
    return mask


# --------------------------------------------------------------------------- transforms
def get_transforms(train: bool, size: int = 256, grayscale_p: float = 0.0) -> A.Compose:
    """Geometric + photometric augmentation. Normalization is NOT included here so the
    visualization tools can display exactly what the model sees before normalizing.

    Albumentations applies the same geometric transform to image and mask, and uses
    nearest-neighbor interpolation for the mask, so labels never become fractional.
    """
    if not train:
        return A.Compose([A.Resize(size, size)])
    ops = [
        A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), translate_percent=(-0.05, 0.05), p=0.5,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=IGNORE_INDEX),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03, p=0.7),
    ]
    if grayscale_p > 0:
        ops.append(A.ToGray(num_output_channels=3, p=grayscale_p))
    return A.Compose(ops)


def normalize_image(image: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> float tensor 3xHxW with ImageNet normalization."""
    x = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def denormalize_image(x: torch.Tensor) -> np.ndarray:
    """Inverse of normalize_image, for viewing tensors that came out of the loader."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1).mul(255).byte()
    return img.permute(1, 2, 0).cpu().numpy()


# --------------------------------------------------------------------------- split files
@dataclass(frozen=True)
class Record:
    name: str        # e.g. "Abyssinian_1"
    breed: str       # e.g. "Abyssinian"
    species: str     # "cat" or "dog"


def breed_from_name(name: str) -> str:
    return name.rsplit("_", 1)[0]


def read_split_file(path: Path) -> list[Record]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        name, species_id = parts[0], int(parts[2])
        records.append(Record(name, breed_from_name(name), SPECIES_NAMES[species_id]))
    return records


def make_splits(root: str | Path, val_fraction: float = 0.1, seed: int = 0) -> dict[str, list[Record]]:
    """train/val carved from the official trainval list (stratified by breed), test is the
    official test list. Deterministic for a given seed, so every model sees identical splits.
    """
    ann = Path(root) / "annotations"
    trainval = read_split_file(ann / "trainval.txt")
    test = read_split_file(ann / "test.txt")

    by_breed: dict[str, list[Record]] = defaultdict(list)
    for r in trainval:
        by_breed[r.breed].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for breed in sorted(by_breed):
        items = sorted(by_breed[breed], key=lambda r: r.name)
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return {"train": sorted(train, key=lambda r: r.name),
            "val": sorted(val, key=lambda r: r.name),
            "test": test}


# --------------------------------------------------------------------------- dataset
class OxfordPetSeg(Dataset):
    """Returns dicts: image (3xHxW float, normalized), mask (HxW long, 0/1/255), name, breed, species."""

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 256,
                 boundary_mode: str = "ignore", transform: A.Compose | None = None,
                 val_fraction: float = 0.1, seed: int = 0, grayscale_p: float = 0.0):
        self.root = Path(root)
        self.split = split
        self.boundary_mode = boundary_mode
        self.records = self._filter_existing(make_splits(self.root, val_fraction, seed)[split])
        self.transform = transform if transform is not None else get_transforms(
            train=(split == "train"), size=image_size, grayscale_p=grayscale_p)

    def _filter_existing(self, records: list[Record]) -> list[Record]:
        keep = [r for r in records if self.image_path(r).exists() and self.trimap_path(r).exists()]
        dropped = len(records) - len(keep)
        if dropped:
            print(f"[OxfordPetSeg] {self.split}: skipped {dropped} records with a missing image or trimap")
        return keep

    def image_path(self, r: Record) -> Path:
        return self.root / "images" / f"{r.name}.jpg"

    def trimap_path(self, r: Record) -> Path:
        return self.root / "annotations" / "trimaps" / f"{r.name}.png"

    def __len__(self) -> int:
        return len(self.records)

    def index_of(self, name: str) -> int:
        for i, r in enumerate(self.records):
            if r.name == name:
                return i
        raise KeyError(name)

    def load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Original-resolution (image uint8 HxWx3, trimap uint8 HxW with values 1/2/3)."""
        r = self.records[idx]
        # A few dataset files are grayscale or RGBA, so always force RGB.
        image = np.array(Image.open(self.image_path(r)).convert("RGB"))
        trimap = np.array(Image.open(self.trimap_path(r)))
        if trimap.ndim == 3:
            trimap = trimap[..., 0]
        return image, trimap

    def load_pair(self, idx: int, transform: A.Compose | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(image uint8, mask 0/1/255) after the given transform (default: this split's), before normalization."""
        image, trimap = self.load_raw(idx)
        mask = convert_trimap(trimap, self.boundary_mode)
        t = transform if transform is not None else self.transform
        out = t(image=image, mask=mask)
        return out["image"], out["mask"]

    def __getitem__(self, idx: int) -> dict:
        image, mask = self.load_pair(idx)
        r = self.records[idx]
        return {
            "image": normalize_image(image),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "name": r.name,
            "breed": r.breed,
            "species": r.species,
        }
