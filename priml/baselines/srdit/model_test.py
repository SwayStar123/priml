"""Shape, routing, and objective checks for the SpeedrunDiT backbone."""

from __future__ import annotations

import torch

from priml.baselines.srdit.model import SpeedrunDiT
from priml.baselines.srdit.objective import SpeedrunObjective
from priml.baselines.srdit.optimizers import srdit_optimizer
from priml.baselines.srdit.sampling import sample_latents
from priml.math.diffusion.time_shift import time_shift
from priml.model.attention.image_rope import ImageRoPE
from priml.optimizers.muon import Muon


def tiny_model() -> SpeedrunDiT:
    config = SpeedrunDiT.Config(
        input_size=4,
        in_channels=2,
        patch_size=1,
        hidden_size=32,
        depth=6,
        num_heads=4,
        cls_channels=8,
        projector_hidden=16,
        projection_depths=(2, 3, 6),
        drop_ratio=0.5,
        path_drop_prob=0.0,
        class_dropout_prob=0.0,
    )
    return config.make()


def test_rope_leaves_cls_untouched_and_uses_original_positions() -> None:
    rope = ImageRoPE.Config(head_dim=8, grid_size=4).make()
    x = torch.randn(2, 4, 3, 8)
    ids = torch.tensor([[-1, 5, 2], [-1, 6, 1]])
    rotated = rope(x, ids)
    assert torch.equal(rotated[:, :, 0], x[:, :, 0])
    assert not torch.equal(rotated[:, :, 1], x[:, :, 1])
    assert torch.allclose(
        rotated[0, :, 1], rope(x[0:1, :, 1:2], ids[0:1, 1:2])[0, :, 0]
    )


def test_training_routing_alignment_and_backward() -> None:
    model = tiny_model().train()
    image = torch.randn(2, 2, 4, 4)
    labels = torch.tensor([1, 2])
    teacher = tuple(torch.randn(2, 17, 8) for _ in range(3))
    objective = SpeedrunObjective(shift_time=False)
    terms = objective(model, image, labels, teacher, time=torch.full((2,), 0.5))
    assert terms.loss.shape == (2,)
    assert terms.output.velocity.shape == image.shape
    assert terms.output.cls_velocity.shape == (2, 8)
    assert [p.tokens.shape[1] for p in terms.output.projections] == [17, 8, 17]
    kept = terms.output.projections[1].ids_keep
    assert kept is not None
    assert kept.shape == (2, 8)
    terms.loss.mean().backward()
    assert model.final_layer.linear.weight.grad is not None
    assert model.projector[0].weight.grad is not None


def test_optimizer_partitions_hidden_matrices_from_heads() -> None:
    model = tiny_model()
    optimizer = srdit_optimizer().make()(model)
    assert len(optimizer.optimizers) == 2
    adam, muon = optimizer.optimizers
    assert isinstance(muon, Muon)
    assert any(
        p is model.final_layer.linear.weight for p in adam.param_groups[0]["params"]
    )
    assert any(
        p is model.blocks[0].attn.qkv.weight for p in muon.param_groups[0]["params"]
    )
    assert any(p is model.projector[0].weight for p in muon.param_groups[0]["params"])


def test_shift_and_sampler_return_expected_latent_shapes() -> None:
    assert torch.allclose(
        time_shift(torch.tensor([0.0, 1.0]), 8192), torch.tensor([0.0, 1.0])
    )
    model = tiny_model().eval()
    latents = torch.randn(2, 2, 4, 4)
    cls = torch.randn(2, 8)
    sampled, sampled_cls = sample_latents(
        model,
        latents,
        cls,
        torch.tensor([1, 2]),
        num_steps=3,
        shift_time=False,
    )
    assert sampled.shape == latents.shape
    assert sampled_cls.shape == cls.shape
    assert torch.isfinite(sampled).all()
