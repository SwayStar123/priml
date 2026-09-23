"""Frozen DINOv2 representation targets for REG alignment."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch


def _load_encoder(variant: str) -> nn.Module:
    """Populate the shared Hub cache before the other ranks load DINOv2."""
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return torch.hub.load("facebookresearch/dinov2", variant)

    encoder = None
    error = None
    if distributed.get_rank() == 0:
        try:
            encoder = torch.hub.load("facebookresearch/dinov2", variant)
        except Exception as exc:  # noqa: BLE001 - propagate Hub failures to every rank
            error = exc
    status = [str(error) if error is not None else None]
    distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        if error is not None:
            raise error
        raise RuntimeError(f"rank 0 could not load DINOv2: {status[0]}")
    if encoder is None:
        encoder = torch.hub.load("facebookresearch/dinov2", variant)
    return encoder


class DinoV2Teacher(nn.Module):
    """Expose selected DINOv2 layers as CLS plus spatial patch tokens."""

    class Config(Fig["DinoV2Teacher"]):
        variant: str = "dinov2_vitb14"
        """Torch Hub DINOv2 backbone name."""
        layer_indices: tuple[int, ...] = (12, 12, 12)
        """Reference block indices for the three REG targets."""
        image_size: int = 256
        """Side length of the preprocessed teacher image."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        self.config = config
        self.encoder = _load_encoder(config.variant)
        self.encoder.eval().requires_grad_(False)
        patch_grid = config.image_size // 16
        # Match the reference's 224/448-pixel input and resized DINO position table.
        position = self.encoder.pos_embed.detach()
        source_grid = int((position.shape[1] - 1) ** 0.5)
        patch = (
            position[:, 1:].reshape(1, source_grid, source_grid, -1).permute(0, 3, 1, 2)
        )
        patch = functional.interpolate(
            patch,
            size=(patch_grid, patch_grid),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        resized = torch.cat((position[:, :1], patch.flatten(2).transpose(1, 2)), dim=1)
        self.encoder.pos_embed = nn.Parameter(resized, requires_grad=False)

    @torch.no_grad()
    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Return CLS plus patch tokens at the requested DINO block indices."""
        size = 224 * (self.config.image_size // 256)
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("teacher image size does not match configuration")
        image = image.float() / 255
        mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        image = functional.interpolate(
            (image - mean) / std, size=(size, size), mode="bicubic"
        )
        blocks = len(self.encoder.blocks)
        requested = tuple(max(0, min(i, blocks - 1)) for i in self.config.layer_indices)
        unique = sorted(set(requested))
        layers = self.encoder.get_intermediate_layers(
            image, n=unique, reshape=False, return_class_token=True
        )
        by_layer = {
            index: torch.cat((cls[:, None], patches), dim=1)
            for index, (patches, cls) in zip(unique, layers, strict=True)
        }
        return tuple(by_layer[i] for i in requested)
