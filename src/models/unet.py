"""U-Net written from scratch (Ronneberger et al., 2015), the no-pretraining baseline.

Differences from the 2015 paper, all standard modern choices:
    - padded 3x3 convs, so output size == input size (no cropping of skip connections)
    - BatchNorm after every conv
    - bilinear upsampling + conv instead of transposed conv (avoids checkerboard artifacts)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.MaxPool2d(2), DoubleConv(cin, cout))


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Encoder widths base * (1, 2, 4, 8, 16). base=32 gives ~7.8M params, base=64 the paper's ~31M."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base: int = 32):
        super().__init__()
        w = [base * m for m in (1, 2, 4, 8, 16)]
        self.inc = DoubleConv(in_channels, w[0])
        self.downs = nn.ModuleList([Down(w[i], w[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([Up(w[i + 1], w[i], w[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(w[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        y = skips.pop()
        for up in self.ups:
            y = up(y, skips.pop())
        return self.head(y)
