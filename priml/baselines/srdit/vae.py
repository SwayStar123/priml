"""The baseline's INVAE input and latent scaling convention."""

from __future__ import annotations

from torch import Tensor, nn

import torch


LATENT_SCALE = 0.3099


@torch.no_grad()
def encode_image(vae: nn.Module, image: Tensor) -> Tensor:
    """Sample unscaled INVAE latents from uint8 NCHW images."""
    return vae.encode(image.float() / 127.5 - 1).sample()


@torch.no_grad()
def decode_latents(vae: nn.Module, latents: Tensor) -> Tensor:
    """Decode model latents to float RGB in [0, 1]."""
    return ((vae.decode(latents / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
