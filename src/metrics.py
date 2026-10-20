"""Segmentation metrics. Pixels equal to IGNORE_INDEX never count toward any score.

Dice = 2|P∩G| / (|P| + |G|),  IoU = |P∩G| / |P∪G|,  IoU = D / (2 - D).
Per-image scores are averaged over images (each image counts equally, regardless of pet size).
"""
from __future__ import annotations

import numpy as np
import torch

from .dataset import IGNORE_INDEX

EPS = 1e-7


def _stats(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """pred, target: (B,H,W). pred is binary {0,1}. Returns per-image tp, fp, fn as floats."""
    valid = target != ignore_index
    p = (pred > 0) & valid
    g = (target == 1) & valid
    dims = tuple(range(1, pred.ndim))
    tp = (p & g).sum(dims).float()
    fp = (p & ~g).sum(dims).float()
    fn = (~p & g).sum(dims).float()
    return tp, fp, fn


def dice_per_image(pred, target, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Binary pred vs target (both (B,H,W) or (H,W)). Empty pred and empty target scores 1."""
    pred, target = _as_batched_tensor(pred), _as_batched_tensor(target)
    tp, fp, fn = _stats(pred, target, ignore_index)
    return (2 * tp + EPS) / (2 * tp + fp + fn + EPS)


def iou_per_image(pred, target, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    pred, target = _as_batched_tensor(pred), _as_batched_tensor(target)
    tp, fp, fn = _stats(pred, target, ignore_index)
    return (tp + EPS) / (tp + fp + fn + EPS)


def pixel_accuracy_per_image(pred, target, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Reported only to show why it is misleading (all-background scores ~0.7 on this dataset)."""
    pred, target = _as_batched_tensor(pred), _as_batched_tensor(target)
    valid = target != ignore_index
    correct = ((pred > 0).long() == (target == 1).long()) & valid
    dims = tuple(range(1, pred.ndim))
    return correct.sum(dims).float() / valid.sum(dims).clamp(min=1).float()


def _as_batched_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 4:  # (B,1,H,W)
        x = x[:, 0]
    return x


class RunningSegMetrics:
    """Accumulates per-image Dice/IoU across batches. Feed logits or probabilities."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.dice: list[float] = []
        self.iou: list[float] = []

    @torch.no_grad()
    def update(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (_as_batched_tensor(probs) >= self.threshold).long()
        d = dice_per_image(pred, target)
        self.dice.extend(d.tolist())
        self.iou.extend(iou_per_image(pred, target).tolist())
        return d

    def summary(self) -> dict[str, float]:
        if not self.dice:
            return {"dice": float("nan"), "iou": float("nan")}
        return {"dice": float(np.mean(self.dice)), "iou": float(np.mean(self.iou))}
