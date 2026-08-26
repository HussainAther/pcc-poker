# Prospective Score-Control two-sided intervention

This post-v0.8 synthetic extension tests one architectural hypothesis from the frozen matched-state decomposition: Control may require contextual allocation on both sides of the decision, not only context-modulated aggression.

The existing aggressive term remains unchanged:

`3.35 * (opponent_fold_probability - 1/3)`

One new zero-centered passive/check-call term is added:

`1.24 * (1/3 - opponent_fold_probability)`

The coefficient `1.24` was fixed before evaluation as `3.35 * (0.1701541715 / 0.4604725914)`, using only the preceding matched-state passive-mass and value-advantage summary.

The original three-stage Control recovery seeds, measurements, correlation threshold (`0.20`), discriminant-margin threshold (`0.05`), and Adaptive family are unchanged. No human data are accessed and the v0.8 human-facing panel is not modified.
