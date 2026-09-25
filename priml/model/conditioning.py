"""Diffusion timestep and class conditioning modules."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost, elementwise_cost, matmul_cost, traffic
from priml.math.diffusion.conditioning import timestep_embedding


class TimestepEmbedder(nn.Module):
    """Project a sinusoidal timestep into the model's conditioning width."""

    class Config(Fig["TimestepEmbedder"]):
        channels: int = -1
        """Width of the projected conditioning vector."""

        frequency_channels: int = 256
        """Width of the fixed sinusoidal input embedding."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the two projections and SiLU for one batch."""
            del kwargs
            return (
                matmul_cost(
                    channels_in=self.frequency_channels,
                    channels_out=self.channels,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.channels,
                    channels_out=self.channels,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=5 * batch_size * self.channels,
                    adjoint=5 * batch_size * self.channels,
                    channels=self.channels,
                    rows=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.frequency_channels = config.frequency_channels
        self.mlp = nn.Sequential(
            nn.Linear(config.frequency_channels, config.channels),
            nn.SiLU(),
            nn.Linear(config.channels, config.channels),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed one time per batch example."""
        return self.mlp(timestep_embedding(t, self.frequency_channels).to(t.dtype))


class ClassEmbedder(nn.Module):
    """Class conditioning with one extra unconditional label."""

    class Config(Fig["ClassEmbedder"]):
        num_classes: int = -1
        """Number of conditional labels, excluding the null class."""

        channels: int = -1
        """Width of each class embedding."""

        dropout: float = 0.1
        """Probability of replacing a label with the null class."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost embedding lookup; all class rows are owned."""
            del kwargs
            table = (self.num_classes + 1) * self.channels
            return traffic(
                "primal",
                "selection",
                elements=batch_size * self.channels,
                dtype=dtype,
            ) + Cost(
                params=table,
                params_active=min(batch_size, self.num_classes + 1) * self.channels,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if not 0 <= config.dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.embedding_table = nn.Embedding(config.num_classes + 1, config.channels)

    def forward(self, labels: Tensor, *, force_drop: Tensor | None = None) -> Tensor:
        """Embed labels, optionally replacing them with the null class."""
        if force_drop is not None:
            drop = force_drop.bool()
        elif self.training and self.dropout:
            drop = torch.rand(labels.shape, device=labels.device) < self.dropout
        else:
            drop = torch.zeros_like(labels, dtype=torch.bool)
        return self.embedding_table(torch.where(drop, self.num_classes, labels))
