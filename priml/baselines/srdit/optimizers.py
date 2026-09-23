"""The reference Muon arithmetic and its AdamW parameter partition."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, overload, override

from configgle import Fig, PartialConfig
from torch import Tensor
from torch.optim import Optimizer

import torch

from priml.optimizers.composite import CompositeOptimizer, complement, excluding
from priml.optimizers.muon import Muon


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class SrditMuon(Optimizer):
    """Muon update using the branch's exact bf16 Newton-Schulz operation order."""

    class Config(Fig["Callable[..., SrditMuon]"]):
        lr: float = 1e-3
        """Matrix learning rate."""

        momentum: float = 0.95
        """Exponential momentum coefficient."""

        weight_decay: float = 0.0
        """Decoupled weight decay on matrix parameters."""

        nesterov: bool = True
        """Use the Nesterov lookahead gradient."""

        ns_steps: int = 5
        """Number of quintic Newton-Schulz iterations."""

        @override
        def make(self) -> Callable[..., SrditMuon]:
            config = self.copy_tree().finalize()
            return partial(
                SrditMuon,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
                ns_steps=config.ns_steps,
            )

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        nesterov: bool,
        ns_steps: int,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
            },
        )

    @staticmethod
    def _zeropower(gradient: Tensor, steps: int) -> Tensor:
        """Match the reference's bf16 casts and polynomial evaluation order."""
        a, b, c = 3.4445, -4.7750, 2.0315
        update = gradient.bfloat16()
        transpose = gradient.size(-2) > gradient.size(-1)
        if transpose:
            update = update.mT
        update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            gram = update @ update.mT
            polynomial = b * gram + c * gram @ gram
            update = a * update + polynomial @ update
        return update.mT if transpose else update

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @torch.no_grad()
    @override
    def step(
        self, closure: Callable[[], Tensor | float] | None = None
    ) -> Tensor | float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            decay = group["weight_decay"]
            momentum = group["momentum"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.ndim < 2:
                    raise ValueError("SrditMuon needs matrix-shaped parameters")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - momentum)
                gradient = (
                    gradient.lerp_(buffer, momentum) if group["nesterov"] else buffer
                )
                if gradient.ndim == 4:
                    gradient = gradient.view(len(gradient), -1)
                update = self._zeropower(gradient, group["ns_steps"])
                if update.ndim == 2:
                    update = update.view_as(parameter)
                parameter.mul_(1 - lr * decay)
                parameter.add_(
                    update,
                    alpha=-lr * max(1, parameter.size(-2) / parameter.size(-1)) ** 0.5,
                )
        return loss


def srdit_optimizer() -> CompositeOptimizer.Config:
    """Put hidden matrices on reference Muon and all other weights on AdamW."""
    on_muon = excluding(
        Muon.eligible_tensor,
        "x_embedder",
        "t_embedder",
        "y_embedder",
        "final_layer",
        "cls_projector",
        "adaLN_modulation",
    )
    adamw = PartialConfig(torch.optim.AdamW)
    adamw.lr = 1e-4
    adamw.betas = (0.9, 0.999)
    adamw.weight_decay = 0.0
    adamw.eps = 1e-15
    config = CompositeOptimizer.Config()
    config.optimizers = [adamw, SrditMuon.Config()]
    config.select = [complement(on_muon), on_muon]
    return config
