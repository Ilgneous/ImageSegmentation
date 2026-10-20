"""Pretrained-encoder CNNs from segmentation_models_pytorch.

unet_resnet34   same U-Net topology as the scratch model, ImageNet ResNet34 encoder.
                Comparing it with unet_scratch isolates the effect of pretraining.
deeplabv3plus   atrous ResNet encoder + ASPP + light decoder (Chen et al., 2018).
                Comparing it with unet_resnet34 isolates decoder design.
"""
from __future__ import annotations

import torch.nn as nn


def build_unet_resnet34(encoder: str = "resnet34", pretrained: bool = True, **_) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def build_deeplabv3plus(encoder: str = "resnet50", pretrained: bool = True, **_) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                             in_channels=3, classes=1)
