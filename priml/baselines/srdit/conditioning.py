"""Reusable diffusion timestep and class conditioning."""

from __future__ import annotations

import math

from torch import Tensor, nn

import torch


def timestep_embedding(t: Tensor, width: int = 256, period: int = 10_000) -> Tensor:
    """Sinusoidal embedding for fractional diffusion times."""
    half = width // 2
    frequencies = torch.exp(
        -math.log(period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    angles = t.float()[:, None] * frequencies[None]
    result = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if width % 2:
        result = torch.cat((result, torch.zeros_like(result[:, :1])), dim=-1)
    return result


class TimestepEmbedder(nn.Module):
    """Project a sinusoidal timestep into the model's conditioning width."""

    def __init__(self, channels: int, frequency_channels: int = 256) -> None:
        super().__init__()
        self.frequency_channels = frequency_channels
        self.mlp = nn.Sequential(
            nn.Linear(frequency_channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed one time per batch example."""
        return self.mlp(timestep_embedding(t, self.frequency_channels).to(t.dtype))


class ClassEmbedder(nn.Module):
    """Class conditioning with one extra unconditional label."""

    def __init__(self, num_classes: int, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.num_classes = num_classes
        self.dropout = dropout
        self.embedding_table = nn.Embedding(num_classes + 1, channels)

    def forward(self, labels: Tensor, *, force_drop: Tensor | None = None) -> Tensor:
        """Embed labels, optionally replacing them with the null class."""
        if force_drop is not None:
            drop = force_drop.bool()
        elif self.training and self.dropout:
            drop = torch.rand(labels.shape, device=labels.device) < self.dropout
        else:
            drop = torch.zeros_like(labels, dtype=torch.bool)
        return self.embedding_table(torch.where(drop, self.num_classes, labels))
