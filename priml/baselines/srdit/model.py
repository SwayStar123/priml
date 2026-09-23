"""REG + SPRINT SiT used by SpeedrunDiT.

The small modules expose the architectural decisions independently: axial RoPE,
value residual attention, adaptive conditioning, sparse routing, and layerwise
MLP width. The default configuration is the published SiT-B/1 run.
"""

from __future__ import annotations

from typing import NamedTuple

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.srdit.conditioning import ClassEmbedder, TimestepEmbedder
from priml.baselines.srdit.rope import ImageRoPE
from priml.baselines.srdit.routing import SparseDenseFusion, select_tokens


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply per-example adaLN shift and scale to every token."""
    return x * (1 + scale[:, None]) + shift[:, None]


def sinusoidal_positions(grid_size: int, channels: int) -> Tensor:
    """Build a fixed 2D sine/cosine table with a zero CLS position."""
    if channels % 4:
        raise ValueError("position channels must be divisible by four")
    axis_channels = channels // 2
    omega = 1 / (
        10_000 ** (torch.arange(axis_channels // 2).float() / (axis_channels // 2))
    )
    positions = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(positions, positions, indexing="ij")

    def embed(p: Tensor) -> Tensor:
        angles = p.reshape(-1, 1) * omega[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    spatial = torch.cat((embed(grid_x), embed(grid_y)), dim=-1)
    return torch.cat((torch.zeros(1, channels), spatial), dim=0).unsqueeze(0)


class ValueResidualAttention(nn.Module):
    """QK normalized attention with 2D RoPE and a first-block value reference."""

    def __init__(
        self, channels: int, heads: int, *, qk_norm: bool, value_residual: bool
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(channels, channels)
        self.v1_lambda = nn.Parameter(torch.tensor(0.5)) if value_residual else None

    def forward(
        self, x: Tensor, rope: ImageRoPE, token_ids: Tensor, v1: Tensor | None
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
        q = rope(self.q_norm(q), token_ids)
        k = rope(self.k_norm(k), token_ids)
        output = functional.scaled_dot_product_attention(q, k, v)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, channels)), raw_v


class SiTBlock(nn.Module):
    """RMSNorm + adaLN-Zero block with GELU feedforward."""

    def __init__(
        self,
        channels: int,
        heads: int,
        mlp_ratio: float,
        *,
        qk_norm: bool,
        value_residual: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        self.attn = ValueResidualAttention(
            channels, heads, qk_norm=qk_norm, value_residual=value_residual
        )
        self.norm2 = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 6 * channels)
        )

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        rope: ImageRoPE,
        token_ids: Tensor,
        v1: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply attention and feedforward updates under adaLN conditioning."""
        s1, a1, g1, s2, a2, g2 = self.adaLN_modulation(condition).chunk(6, dim=-1)
        attention, raw_v = self.attn(
            modulate(self.norm1(x), s1, a1), rope, token_ids, v1
        )
        x = x + g1[:, None] * attention
        x = x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))
        return x, raw_v


class FinalLayer(nn.Module):
    """Project the CLS and image tokens into their velocity targets."""

    def __init__(
        self, channels: int, patch_size: int, out_channels: int, cls_channels: int
    ) -> None:
        super().__init__()
        self.norm_final = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 2 * channels)
        )
        self.linear = nn.Linear(channels, patch_size**2 * out_channels)
        self.linear_cls = nn.Linear(channels, cls_channels)

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        """Return patch and CLS velocities."""
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])


class Projection(NamedTuple):
    """Student tokens and optional indices retained by SPRINT."""

    tokens: Tensor
    ids_keep: Tensor | None


class ModelOutput(NamedTuple):
    """Latent velocity, CLS velocity, and REG alignment projections."""

    velocity: Tensor
    cls_velocity: Tensor
    projections: tuple[Projection, ...]


class SpeedrunDiT(nn.Module):
    """Class conditional latent SiT with REG and SPRINT."""

    class Config(Fig["SpeedrunDiT"]):
        input_size: int = 16
        """Spatial side of the INVAE latent grid."""
        in_channels: int = 32
        """INVAE latent channels."""
        patch_size: int = 1
        """Latent cells in each patch side."""
        hidden_size: int = 768
        """Transformer token width."""
        depth: int = 12
        """Total number of transformer blocks."""
        num_heads: int = 12
        """Attention heads per block."""
        num_classes: int = 1000
        """ImageNet classes, excluding the classifier-free token."""
        cls_channels: int = 768
        """DINO CLS feature width."""
        projector_hidden: int = 2048
        """Width of the REG projection MLP."""
        projection_depths: tuple[int, ...] = (2, 4, 6)
        """Blocks whose tokens are aligned to DINO features."""
        mlp_ratio_min: float = 2.0
        """Feedforward expansion in the first block."""
        mlp_ratio_max: float = 6.0
        """Feedforward expansion in the last block."""
        drop_ratio: float = 0.75
        """Fraction of patch tokens removed from the sparse middle blocks."""
        path_drop_prob: float = 0.05
        """Chance of removing the sparse branch during training."""
        class_dropout_prob: float = 0.1
        """Chance of using the classifier-free class embedding."""
        qk_norm: bool = True
        """Normalize query and key heads with RMSNorm."""
        encoder_blocks: int = 2
        """Dense blocks before SPRINT token selection."""
        decoder_blocks: int = 2
        """Dense blocks after sparse and dense streams are fused."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.depth < config.encoder_blocks + config.decoder_blocks:
            raise ValueError("depth shorter than SPRINT dense prefix and suffix")
        if tuple(sorted(set(config.projection_depths))) != config.projection_depths:
            raise ValueError("projection_depths must be strictly increasing")
        if any(d < 1 or d > config.depth for d in config.projection_depths):
            raise ValueError("projection depth outside the model")
        if config.input_size % config.patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.config = config
        self.grid_size = config.input_size // config.patch_size
        self.x_embedder = nn.Conv2d(
            config.in_channels, config.hidden_size, config.patch_size, config.patch_size
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.y_embedder = ClassEmbedder(
            config.num_classes, config.hidden_size, config.class_dropout_prob
        )
        self.cls_projector = nn.Linear(config.cls_channels, config.hidden_size)
        self.wg_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.register_buffer(
            "pos_embed",
            sinusoidal_positions(self.grid_size, config.hidden_size),
            persistent=True,
        )
        self.rope = ImageRoPE(config.hidden_size // config.num_heads, self.grid_size)
        ratios = [
            config.mlp_ratio_min
            + (config.mlp_ratio_max - config.mlp_ratio_min)
            * i
            / max(1, config.depth - 1)
            for i in range(config.depth)
        ]
        self.blocks = nn.ModuleList(
            SiTBlock(
                config.hidden_size,
                config.num_heads,
                ratio,
                qk_norm=config.qk_norm,
                value_residual=i > 0,
            )
            for i, ratio in enumerate(ratios)
        )
        self.fusion = SparseDenseFusion(config.hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.projector_hidden),
            nn.SiLU(),
            nn.Linear(config.projector_hidden, config.projector_hidden),
            nn.SiLU(),
            nn.Linear(config.projector_hidden, config.cls_channels),
        )
        self.final_layer = FinalLayer(
            config.hidden_size,
            config.patch_size,
            config.in_channels,
            config.cls_channels,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the reference SiT and adaLN-Zero initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.x_embedder.weight.flatten(1))
        assert self.x_embedder.bias is not None
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        for module in (self.t_embedder.mlp[0], self.t_embedder.mlp[2]):
            nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            bias = block.adaLN_modulation[-1].bias
            assert bias is not None
            nn.init.zeros_(bias)
        for module in (
            self.final_layer.adaLN_modulation[-1],
            self.final_layer.linear,
            self.final_layer.linear_cls,
        ):
            nn.init.zeros_(module.weight)
            assert module.bias is not None
            nn.init.zeros_(module.bias)

    def _project(
        self, x: Tensor, layer: int, ids: Tensor | None, projections: list[Projection]
    ) -> None:
        if layer in self.config.projection_depths:
            projections.append(Projection(self.projector(x), ids))

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        cls_token: Tensor,
        *,
        force_drop_labels: Tensor | None = None,
        drop_sparse_path: bool = False,
        route_tokens: bool | None = None,
    ) -> ModelOutput:
        """Predict latent and CLS velocities plus intermediate DINO projections."""
        cfg = self.config
        batch, channels, height, width = x.shape
        if (channels, height, width) != (
            cfg.in_channels,
            cfg.input_size,
            cfg.input_size,
        ):
            raise ValueError("latent shape does not match model configuration")
        if cls_token.shape != (batch, cfg.cls_channels):
            raise ValueError("cls_token must have shape [batch, cls_channels]")
        spatial = self.x_embedder(x).flatten(2).transpose(1, 2)
        cls = self.wg_norm(self.cls_projector(cls_token))[:, None]
        x = torch.cat((cls, spatial), dim=1) + self.pos_embed.to(spatial.dtype)
        ids = torch.arange(self.grid_size**2, device=x.device)[None].expand(batch, -1)
        full_ids = torch.cat((torch.full((batch, 1), -1, device=x.device), ids), dim=1)
        condition = self.t_embedder(t) + self.y_embedder(
            y, force_drop=force_drop_labels
        )
        projections: list[Projection] = []
        first_v: Tensor | None = None
        for i in range(cfg.encoder_blocks):
            x, raw_v = self.blocks[i](x, condition, self.rope, full_ids, first_v)
            if first_v is None:
                first_v = raw_v
            self._project(x, i + 1, None, projections)
        dense = x
        if route_tokens is None:
            route_tokens = self.training
        sparse, kept = (
            select_tokens(dense, cfg.drop_ratio) if route_tokens else (dense, None)
        )
        sparse_ids = full_ids.gather(1, kept) if kept is not None else full_ids
        sparse_v = (
            first_v.gather(
                2,
                kept[:, None, :, None].expand(
                    -1, first_v.shape[1], -1, first_v.shape[3]
                ),
            )
            if kept is not None and first_v is not None
            else first_v
        )
        middle_end = cfg.depth - cfg.decoder_blocks
        for i in range(cfg.encoder_blocks, middle_end):
            sparse, _ = self.blocks[i](
                sparse, condition, self.rope, sparse_ids, sparse_v
            )
            self._project(sparse, i + 1, kept, projections)
        if self.training and cfg.path_drop_prob:
            coin = torch.rand((), device=x.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(coin, 0)
            drop_sparse_path = drop_sparse_path or bool(coin < cfg.path_drop_prob)
        x = self.fusion(dense, sparse, kept, drop_path=drop_sparse_path)
        for i in range(middle_end, cfg.depth):
            x, _ = self.blocks[i](x, condition, self.rope, full_ids, first_v)
            self._project(x, i + 1, None, projections)
        patches, cls_velocity = self.final_layer(x, condition)
        p = cfg.patch_size
        velocity = (
            patches.reshape(batch, self.grid_size, self.grid_size, p, p, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, height, width)
        )
        return ModelOutput(velocity, cls_velocity, tuple(projections))
