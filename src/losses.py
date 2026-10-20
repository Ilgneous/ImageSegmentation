"""Soft Dice + BCE, both masked by IGNORE_INDEX.

Soft Dice replaces the thresholded prediction with sigmoid probabilities so the metric
becomes differentiable. That is what lets the same quantity be both the training loss
and the comparison metric. BCE is added because soft Dice gradients are noisy early in
training when predictions are near 0.5 everywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX,
                   smooth: float = 1.0) -> torch.Tensor:
    """logits (B,1,H,W), target (B,H,W) with 0/1/ignore. Mean over images of 1 - soft Dice."""
    probs = torch.sigmoid(logits[:, 0])
    valid = (target != ignore_index).float()
    g = (target == 1).float() * valid
    p = probs * valid
    inter = (p * g).sum((1, 2))
    denom = p.sum((1, 2)) + g.sum((1, 2))
    dice = (2 * inter + smooth) / (denom + smooth)
    return (1 - dice).mean()


def masked_bce(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    valid = (target != ignore_index).float()
    g = (target == 1).float()
    loss = F.binary_cross_entropy_with_logits(logits[:, 0], g, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * soft_dice_loss(logits, target) + self.bce_weight * masked_bce(logits, target)
