"""INVAE checkpoint loading and the baseline's latent scaling convention."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from torch import Tensor, nn

import torch

from priml.model.third_party.invae import VAE_F16D32


LATENT_SCALE = 0.3099


def load_invae(
    checkpoint: Path | str | None = None, *, device: torch.device | str = "cpu"
) -> nn.Module:
    """Load REPA-E/e2e-invae; a local checkpoint avoids network access."""
    if checkpoint is None:
        try:
            hub = import_module("huggingface_hub")
        except ImportError as error:
            raise ImportError(
                "Install priml[hub] or pass a local INVAE checkpoint path"
            ) from error
        checkpoint = hub.hf_hub_download(
            repo_id="REPA-E/e2e-invae", filename="e2e-invae-400k.pt"
        )
    assert checkpoint is not None
    vae = VAE_F16D32()
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    vae.load_state_dict(state)
    return vae.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_image(vae: nn.Module, image: Tensor) -> Tensor:
    """Sample unscaled INVAE latents from uint8 NCHW images."""
    return vae.encode(image.float() / 127.5 - 1).sample()


@torch.no_grad()
def decode_latents(vae: nn.Module, latents: Tensor) -> Tensor:
    """Decode model latents to float RGB in [0, 1]."""
    return ((vae.decode(latents / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
