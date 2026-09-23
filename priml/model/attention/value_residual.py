"""Self attention that mixes each layer's values with a reference value stream."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.model.attention.rope import RoPE


class ValueResidualAttention(nn.Module):
    """QK normalized attention with 2D RoPE and a first-block value reference."""

    class Config(Fig["ValueResidualAttention"]):
        channels: int = -1
        """Input and output token width."""

        heads: int = -1
        """Attention heads."""

        qk_norm: bool = True
        """Apply RMSNorm to query and key heads."""

        value_residual: bool = True
        """Mix values with a supplied reference value stream."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels % config.heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = config.heads
        self.head_dim = config.channels // config.heads
        self.qkv = nn.Linear(config.channels, 3 * config.channels)
        self.q_norm = nn.RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.proj = nn.Linear(config.channels, config.channels)
        self.v1_lambda = (
            nn.Parameter(torch.tensor(0.5)) if config.value_residual else None
        )

    def forward(
        self,
        x: Tensor,
        rope_factors: tuple[Tensor, Tensor],
        v1: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Attend to image tokens and return the raw value stream for reuse."""
        batch, tokens, channels = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        raw_v = v
        if v1 is not None and self.v1_lambda is not None:
            v = self.v1_lambda * v1 + (1 - self.v1_lambda) * v
        cos, sin = rope_factors
        q, k = RoPE.rotate(
            self.q_norm(q).transpose(1, 2),
            self.k_norm(k).transpose(1, 2),
            cos,
            sin,
            interleave=True,
        )
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        output = functional.scaled_dot_product_attention(q, k, v)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, channels)), raw_v
