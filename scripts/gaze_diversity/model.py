"""Frozen EyePoseTinyCNN-v1 architecture."""

from __future__ import annotations

import torch
from torch import nn


class EyePoseTinyCNN(nn.Module):
    """Small eye-image plus head-pose gaze regressor frozen in protocol v1."""

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(98, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, image: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(image).flatten(1)
        return self.regressor(torch.cat((encoded, pose), dim=1))


def count_parameters(model: nn.Module) -> int:
    """Return the exact number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())
