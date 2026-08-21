# Score-Control value-sensitive intervention decomposition

## Purpose

This post-v0.8 synthetic diagnostic asks why the prospective Score-Control
contextual-response intervention recovered **information uptake** and **context
alignment** but did not recover the final **value-sensitive intervention**
stage.

No policy is modified. No human data are accessed. The frozen v0.8 human-facing
panel remains Pressure-only.

## Frozen diagnostic hypothesis

After the contextual-gain intervention, Score-Control receives a real
Control-linked context signal, but Control-heavy trajectories increasingly make
choices with lower label-free counterfactual action efficiency. Multiplying
positive context alignment by the value guardrail therefore attenuates the
Control signal.

The diagnostic separates, per trajectory group:

- aligned-minus-yoked context gain;
- positive context gain;
- counterfactual action efficiency;
- regret;
- positive context gain x efficiency;
- positive-context decisions whose efficiency is below `0.80`.

Synthetic PCC weights are used only after trajectory aggregation for diagnostic
correlations.

## Prespecified checks

Support requires all four:

1. Score mean action efficiency correlates with Control at `<= -0.20`.
2. Score mean regret correlates with Control at `>= +0.20`.
3. Weighting positive context gain by efficiency reduces the Score Control
   correlation by at least `0.05`.
4. The rate of positive-context / low-efficiency decisions correlates with
   Control at `>= +0.20`.

These checks diagnose a bottleneck; they do **not** resolve Poker Control or
justify an automatic second intervention.
