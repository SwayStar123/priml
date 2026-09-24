"""SR-DiT: a latent flow-matching transformer for ImageNet, ported to Priml.

``exp000`` REPRODUCES the published SpeedrunDiT rather than stating a naive
recipe of our own. That is the unusual thing about this baseline: the control
is somebody else's model, pinned to a commit, and the claim under test is
numerical identity rather than a score. What makes the port interesting is
that four mechanisms the reference treats as fixed are values in slots here --
2D rotary positions, SPRINT sparse-dense routing, value residual learning, and
a class token diffused alongside the latent -- so removing one is a fork
rather than an edit.

The bit-for-bit goldens under ``testdata/`` cover ``exp000``. The updated
``exp001`` recipe lives in ``priml.baselines.srdit`` and uses shared Muon and
RoPE, so it does not claim numerical identity with its later REG source.
"""
