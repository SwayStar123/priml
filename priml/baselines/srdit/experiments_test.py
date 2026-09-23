"""Published and smoke SpeedrunDiT configurations."""

from __future__ import annotations

from configgle.testing import assert_pprint_golden

from priml.baselines.srdit.experiments import exp000, exp_smoke


def test_exp000_config_golden() -> None:
    """Keep the complete published recipe reviewable as text."""
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_smoke_keeps_the_training_recipe() -> None:
    """The smoke run only reduces cost and the number of updates."""
    base, smoke = exp000(), exp_smoke()
    assert smoke.max_steps == smoke.step.train_budget_steps == 5
    assert smoke.max_steps < base.max_steps
    assert smoke.step.model.hidden_size < base.step.model.hidden_size
    assert smoke.step.model.depth < base.step.model.depth
    assert smoke.step.model.in_channels == base.step.model.in_channels
    assert smoke.step.model.patch_size == base.step.model.patch_size
    assert smoke.step.model.drop_ratio == base.step.model.drop_ratio
    assert smoke.step.model.qk_norm == base.step.model.qk_norm
    assert smoke.step.latent_scale == base.step.latent_scale
    assert smoke.step.projection_coeff == base.step.projection_coeff
    assert smoke.step.cfm_coeff == base.step.cfm_coeff
