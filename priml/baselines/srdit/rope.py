"""Two-dimensional RoPE for dense and routed image tokens."""

from __future__ import annotations

from torch import Tensor, nn

import torch


def _rotate_pairs(x: Tensor) -> Tensor:
    paired = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)


class ImageRoPE(nn.Module):
    """EVA-style axial RoPE; token index -1 denotes an unrotated CLS token."""

    def __init__(self, head_dim: int, grid_size: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by four for 2D RoPE")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size
        axis_dim = head_dim // 2
        frequencies = theta ** (-torch.arange(0, axis_dim, 2).float() / axis_dim)
        positions = torch.arange(grid_size, dtype=torch.float32)
        angles = torch.repeat_interleave(positions[:, None] * frequencies, 2, dim=-1)
        row = angles[:, None, :].expand(grid_size, grid_size, axis_dim)
        col = angles[None, :, :].expand(grid_size, grid_size, axis_dim)
        two_d = torch.cat((row, col), dim=-1).reshape(grid_size**2, head_dim)
        self.register_buffer("cos", two_d.cos(), persistent=False)
        self.register_buffer("sin", two_d.sin(), persistent=False)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        """Rotate ``[B,H,T,D]`` at original flattened grid positions."""
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0).expand(x.shape[0], -1)
        if token_ids.shape != (x.shape[0], x.shape[2]):
            raise ValueError("token_ids must have shape [batch, tokens]")
        if (token_ids < -1).any() or (token_ids >= self.grid_size**2).any():
            raise ValueError("token_ids outside the image grid")
        ids = token_ids.clamp_min(0)
        cos = self.cos[ids].unsqueeze(1).to(x.dtype)
        sin = self.sin[ids].unsqueeze(1).to(x.dtype)
        rotated = x * cos + _rotate_pairs(x) * sin
        return torch.where((token_ids >= 0)[:, None, :, None], rotated, x)
