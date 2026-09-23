"""INVAE loading and baseline latent conversion contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from torch import nn

import torch

from priml.model.third_party import invae


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_load_invae_from_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared loader restores weights and freezes the encoder."""
    source = nn.Linear(2, 2)
    checkpoint = tmp_path / "invae.pt"
    torch.save(source.state_dict(), checkpoint)
    monkeypatch.setattr(invae, "VAE_F16D32", lambda: nn.Linear(2, 2))

    loaded = invae.load_invae(checkpoint)

    assert isinstance(loaded, nn.Linear)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert torch.equal(loaded.weight, source.weight)
    assert loaded.bias is not None
    assert source.bias is not None
    assert torch.equal(loaded.bias, source.bias)


def test_invae_image_and_latent_conversions() -> None:
    """Keep uint8 normalization and the model latent scale together."""

    class FakeVAE:
        def encode(self, image: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(sample=lambda: image)

        def decode(self, latents: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(sample=latents)

    vae = cast(invae.AutoencoderKL, FakeVAE())
    image = torch.tensor([0, 255], dtype=torch.uint8)
    assert torch.equal(invae.encode_image(vae, image), torch.tensor([-1.0, 1.0]))
    latents = torch.tensor([-1.0, 0.0, 1.0]) * invae.LATENT_SCALE
    assert torch.allclose(
        invae.decode_latents(vae, latents), torch.tensor([0.0, 0.5, 1.0])
    )
