"""Latent and REG CLS sampling with the reference Euler-Maruyama SDE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch import Tensor

import torch

from priml.baselines.srdit.objective import interpolant
from priml.math.diffusion.time_shift import time_shift


if TYPE_CHECKING:
    from priml.baselines.srdit.model import ModelOutput, SpeedrunDiT


def _predict(
    model: SpeedrunDiT,
    x: Tensor,
    cls: Tensor,
    t: Tensor,
    labels: Tensor,
    *,
    latent_dtype: torch.dtype,
    cls_dtype: torch.dtype,
    drop_path: bool,
) -> ModelOutput:
    return model(
        x.to(latent_dtype),
        t.to(latent_dtype),
        labels,
        cls.to(cls_dtype),
        drop_sparse_path=drop_path,
        route_tokens=False,
    )


def score_from_velocity(
    velocity: Tensor, noisy: Tensor, t: Tensor, path: Literal["linear", "cosine"]
) -> Tensor:
    """Convert interpolant velocity to score, matching the SiT SDE sampler."""
    alpha, sigma, d_alpha, d_sigma = interpolant(t, path)
    shape = (slice(None),) + (None,) * (noisy.ndim - 1)
    ratio = alpha[shape] / d_alpha[shape]
    variance = sigma[shape].square() - ratio * d_sigma[shape] * sigma[shape]
    return (ratio * velocity - noisy) / variance


@torch.no_grad()
def sample_latents(
    model: SpeedrunDiT,
    latents: Tensor,
    cls_latents: Tensor,
    labels: Tensor,
    *,
    num_steps: int = 250,
    cfg_scale: float = 1.0,
    cls_cfg_scale: float = 1.0,
    path: Literal["linear", "cosine"] = "linear",
    shift_time: bool = True,
    shift_base: int = 4096,
    path_drop_guidance: bool = False,
    guidance_low: float = 0.0,
    guidance_high: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample raw INVAE latents; decode after dividing by the 0.3099 scale.

    Path-drop guidance is intended for qualitative images only. The paper's
    reported quantitative metrics use ordinary sampling without it.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least two")
    was_training = model.training
    model.eval()
    try:
        x, cls = latents.double(), cls_latents.double()
        t_steps = torch.cat(
            (
                torch.linspace(
                    1, 0.04, num_steps, device=x.device, dtype=torch.float64
                ),
                torch.zeros(1, device=x.device, dtype=torch.float64),
            )
        )
        if shift_time:
            t_steps = time_shift(t_steps, latents[0].numel(), shift_base)
        for index in range(num_steps):
            t_cur, t_next = t_steps[index], t_steps[index + 1]
            t = t_cur.expand(x.shape[0])
            use_cfg = cfg_scale > 1 and guidance_low <= t_cur <= guidance_high

            cond = _predict(
                model,
                x,
                cls,
                t,
                labels,
                latent_dtype=latents.dtype,
                cls_dtype=cls_latents.dtype,
                drop_path=False,
            )
            score_x = score_from_velocity(cond.velocity.double(), x, t, path)
            score_cls = score_from_velocity(cond.cls_velocity.double(), cls, t, path)
            diffusion = 2 * t_cur
            drift_x = cond.velocity.double() - 0.5 * diffusion * score_x
            drift_cls = cond.cls_velocity.double() - 0.5 * diffusion * score_cls
            if use_cfg:
                null = torch.full_like(labels, model.config.num_classes)
                uncond = _predict(
                    model,
                    x,
                    cls,
                    t,
                    null,
                    latent_dtype=latents.dtype,
                    cls_dtype=cls_latents.dtype,
                    drop_path=path_drop_guidance,
                )
                score_u = score_from_velocity(uncond.velocity.double(), x, t, path)
                score_cls_u = score_from_velocity(
                    uncond.cls_velocity.double(), cls, t, path
                )
                drift_u = uncond.velocity.double() - 0.5 * diffusion * score_u
                drift_cls_u = (
                    uncond.cls_velocity.double() - 0.5 * diffusion * score_cls_u
                )
                drift_x = drift_u + cfg_scale * (drift_x - drift_u)
                if cls_cfg_scale > 0:
                    drift_cls = drift_cls_u + cls_cfg_scale * (drift_cls - drift_cls_u)
            dt = t_next - t_cur
            x = x + drift_x * dt
            cls = cls + drift_cls * dt
            if index < num_steps - 1:
                x = x + (diffusion * -dt).sqrt() * torch.randn_like(x)
                cls = cls + (diffusion * -dt).sqrt() * torch.randn_like(cls)
        return x.to(latents.dtype), cls.to(cls_latents.dtype)
    finally:
        model.train(was_training)
