"""Self attention that mixes each layer's values with a reference value stream."""

from __future__ import annotations

from typing import Protocol, override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.attention.rope import RoPE, rotation_cost
from priml.model.norm import RMSNorm


class ValueBlend(Protocol):
    """Blends one layer's attention values with the first layer's."""

    def __call__(self, v: Tensor, first: Tensor, /) -> Tensor:
        """Apply to the input."""
        ...


def blend_values(v: Tensor, first: Tensor, weight: Tensor) -> Tensor:
    """Mix current and first-layer values in the reference's operation order."""
    return weight * first + (1.0 - weight) * v


class ValueResidual(nn.Module):
    """Blend a layer's values toward the first layer's, by a learned scalar.

    References:
      https://arxiv.org/abs/2410.17897
        Zhou et al. 2024, "Value Residual Learning For Alleviating Attention
        Concentration In Transformers."

    """

    class Config(Fig["ValueResidual"], kw_only=False):
        """Configuration for ValueResidual."""

        initial: float = 0.5
        """Starting mixing weight on the first layer's values."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            heads: int,
            channels_head: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one value blend."""
            del kwargs
            return elementwise_cost(
                primal=3,
                adjoint=4,
                channels=channels_head,
                rows=seq_len * batch_size * heads,
                dtype=dtype,
                inputs=2,
                params=1,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(config.initial))

    @override
    def forward(self, v: Tensor, first: Tensor, **kwargs: object) -> Tensor:
        """Blend values toward the first layer's."""
        del kwargs
        return blend_values(v, first, self.weight)


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

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost projections, QK normalization, rotary, SDPA, and value blend."""
            del kwargs
            head_dim = self.channels // self.heads
            rows = seq_len * batch_size
            total = (
                matmul_cost(
                    channels_in=self.channels,
                    channels_out=3 * self.channels,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.channels,
                    channels_out=self.channels,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + attention_kernel_cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    num_heads=self.heads,
                    channels_head=head_dim,
                )
                + rotation_cost(
                    RoPE.Config(channels_head=(head_dim // 2, head_dim // 2)),
                    rows=rows,
                    dtype=dtype,
                    channels_head=head_dim,
                    heads=2 * self.heads,
                )
            )
            if self.qk_norm:
                total += cost(
                    RMSNorm.Config(channels_in=head_dim, elementwise_affine=True),
                    seq_len=seq_len,
                    batch_size=batch_size * self.heads,
                    dtype=dtype,
                ).tile(2, copies=2)
            if self.value_residual:
                elements = rows * self.channels
                total += elementwise_cost(
                    primal=3 * elements,
                    adjoint=4 * elements,
                    channels=head_dim,
                    rows=rows * self.heads,
                    params=1,
                    inputs=2,
                    dtype=dtype,
                )
            return total

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
            v = blend_values(v, v1, self.v1_lambda)
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
