"""SegFormer (Xie et al., 2021): hierarchical Mix Transformer encoder + all-MLP decoder.

The HF head predicts at 1/4 resolution, so logits are upsampled back to the input size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerBinary(nn.Module):
    def __init__(self, checkpoint: str = "nvidia/mit-b0", pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint, num_labels=1, ignore_mismatched_sizes=True)
        else:
            cfg = SegformerConfig.from_pretrained(checkpoint) if "/" in checkpoint else SegformerConfig()
            cfg.num_labels = 1
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits           # (B,1,H/4,W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_segformer(checkpoint: str = "nvidia/mit-b0", pretrained: bool = True, **_) -> nn.Module:
    return SegFormerBinary(checkpoint, pretrained)
