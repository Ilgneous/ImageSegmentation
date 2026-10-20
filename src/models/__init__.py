"""Model registry. Every model maps a normalized (B,3,H,W) batch to (B,1,H,W) logits."""
from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

from .unet import UNet


def _unet_scratch(base: int = 32, **_) -> nn.Module:
    return UNet(base=base)


def _lazy(module: str, fn: str):
    def build(**kw):
        import importlib

        return getattr(importlib.import_module(f"{__name__}.{module}"), fn)(**kw)
    return build


REGISTRY = {
    "unet_scratch": _unet_scratch,
    "unet_resnet34": _lazy("smp_models", "build_unet_resnet34"),
    "deeplabv3plus": _lazy("smp_models", "build_deeplabv3plus"),
    "segformer": _lazy("segformer", "build_segformer"),
    "dino_linear": _lazy("dino_probe", "build_dino_linear"),
}


def build_model(cfg: Mapping) -> nn.Module:
    """cfg: {'name': <registry key>, ...kwargs for that builder}."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}, choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**cfg)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
