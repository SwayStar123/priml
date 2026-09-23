"""Fixed position tables for spatial transformer tokens."""

from __future__ import annotations

from torch import Tensor

import torch


def image_token_positions(grid_size: int, device: torch.device) -> Tensor:
    """Return axial positions for CLS followed by row-major image tokens."""
    ids = torch.arange(grid_size**2, device=device)
    spatial = torch.stack((ids // grid_size, ids % grid_size), dim=-1)
    return torch.cat((torch.zeros(1, 2, device=device, dtype=ids.dtype), spatial))[None]


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
