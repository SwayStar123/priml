"""Contrastive flow matching loss on unrelated targets within a batch."""

from __future__ import annotations

from torch import Tensor


def contrastive_flow_loss(
    prediction: Tensor, target: Tensor, *, time: Tensor | None = None
) -> Tensor:
    """Negative batch MSE against targets rolled by one sample."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.shape[0] < 2:
        raise ValueError("contrastive flow matching requires at least two samples")
    difference = (prediction - target.roll(1, dims=0)).square()
    if time is not None:
        difference = difference * time.reshape(-1, *([1] * (difference.ndim - 1)))
    return -difference.mean()
