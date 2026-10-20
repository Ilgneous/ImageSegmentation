"""Frozen DINOv3 backbone + linear (or tiny conv) head on the patch tokens.

Tests how good general self-supervised features are with almost no task training: only the
head is trained. With patch size 16 a 256x256 input gives a 16x16 token grid, so the head
predicts at 1/16 resolution and is bilinearly upsampled. Expect soft boundaries, and use the
error map / boundary profile to show it.

The DINOv3 weights on Hugging Face are gated: accept the license on the model page, then
`huggingface-cli login`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoProbe(nn.Module):
    def __init__(self, checkpoint: str = "facebook/dinov3-vits16-pretrain-lvd1689m", head: str = "linear",
                 pretrained: bool = True, freeze: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.backbone = AutoModel.from_pretrained(checkpoint)
        else:
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint))
        cfg = self.backbone.config
        self.patch = getattr(cfg, "patch_size", 16)
        self.n_special = 1 + getattr(cfg, "num_register_tokens", 0)  # CLS + registers come first
        dim = cfg.hidden_size
        self.freeze = freeze
        if freeze:
            self.backbone.requires_grad_(False).eval()
        if head == "linear":
            self.head = nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, 1, 1))
        elif head == "conv":
            self.head = nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, 256, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 1, 1))
        else:
            raise ValueError(head)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # keep the frozen backbone in inference mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        with torch.set_grad_enabled(not self.freeze and self.training):
            tokens = self.backbone(pixel_values=x).last_hidden_state   # (B, special + h*w, D)
        grid = tokens[:, self.n_special:, :].transpose(1, 2).reshape(x.shape[0], -1, h, w)
        logits = self.head(grid)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_dino_linear(checkpoint: str = "facebook/dinov3-vits16-pretrain-lvd1689m", head: str = "linear",
                      pretrained: bool = True, freeze: bool = True, **_) -> nn.Module:
    return DinoProbe(checkpoint, head, pretrained, freeze)
