# Prospective Pressure Mechanism Decomposition

## Question

The frozen v0.7 mechanism experiment established that correctly aligned public-context information improves engineered Control specifically against Pressure. What feature of the engineered Pressure component creates that exploitable contextual structure?

This experiment follows that causal thread without changing the frozen v0.3 Control implementation.

## Candidate mechanisms

The Adaptive Pressure component currently contains two explicit sources of selectivity:

1. `fold_leverage`: aggression is strengthened when the learned opponent model predicts folding;
2. `strength_selectivity`: equity-dependent terms determine when Pressure escalates, calls, checks, or releases.

The experiment compares the unchanged `full` Pressure component with two one-term ablations:

- `no_fold_leverage`: set the learned fold-leverage contribution to zero;
- `no_strength_selectivity`: set the equity-strength contribution to zero inside the Pressure component.

Control and Chaos code are not modified. The base `families.py` source must retain the frozen v0.3 checksum or the experiment refuses to run.

These are mechanism ablations, not matched behavioral controls: removing a Pressure term is allowed to change Pressure's action distribution. The estimand is therefore mechanistic necessity within the engineered policy, not an observationally matched causal effect.

## Alignment intervention

For each Pressure variant, a balanced probe calibrates a public-context opponent model on separate hands. Frozen Control is then evaluated with either:

1. `aligned`: the calibrated model in its observed contexts; or
2. `context_yoked`: action labels reassigned across rounds within the same legal-action stratum, preserving observation support while destroying action-to-round alignment.

The primary quantity for each variant is

`M(variant) = payoff(Control_aligned, variant) - payoff(Control_yoked, variant)`.

The decomposition contrast is

`D(variant) = M(full) - M(variant)`.

## Frozen default design

- 16 independent replicates;
- 250 calibration hands per seat order;
- 500 evaluation hands per seat order and condition;
- disjoint seed families `111001` and `121001`, stride `2000`;
- purity `.80`;
- temperature `.35`;
- common random numbers within each aligned/yoked comparison;
- both seat orders;
- replicate-level contrasts are inferential units.

The defaults deliberately use seed families not used by the v0.6 or v0.7 confirmation experiments.

## Prespecified confirmation rule

The decomposition is confirmed only if all of the following hold:

1. the unchanged full-Pressure alignment effect has a 95% interval above zero on the fresh seeds;
2. at least one ablation has a positive `full minus ablation` 95% interval; and
3. that ablation removes at least 50% of the full Pressure alignment effect by the replicate-level mean contrast.

The 50% threshold is an effect-attenuation criterion, not a claim that the mechanism is uniquely necessary. If both ablations qualify, the result supports distributed mechanism dependence rather than a single privileged term. If neither qualifies, the decomposition fails and the null is retained.

## Interpretation boundary

A positive result identifies which programmed Pressure term is necessary for a substantial portion of the previously confirmed Control alignment advantage. It does not show that the term is sufficient, uniquely causal, optimal, or present in humans. Because the ablations may change marginal action frequencies, this experiment complements rather than replaces the matched-context intervention in `CONTROL_PRESSURE_MECHANISM_PROTOCOL.md`.
