# Frozen Counterfactual Control Validation

## Question

Does a correctly matched opponent model improve held-out payoff specifically
for Control-heavy policies, beyond the model dependence of Pressure and Chaos?

The v0.5 temporal experiment established that opponent history predicts future
actions, but its trajectory score was more associated with Pressure than
Control. This experiment therefore intervenes on model identity directly.

## Intervention

A balanced probe observes each target opponent in both seat orders during a
calibration phase. The frozen Adaptive v0.3 focal policy is then evaluated
under three conditions:

1. `aligned`: the model calibrated against the evaluation opponent's mode;
2. `swapped`: a model calibrated against a different opponent mode;
3. `prior`: an empty, Laplace-smoothed model with no observations.

Only the focal policy's opponent model changes. Policy code, PCC weights,
temperature, opponent, deck seed, policy random seeds, hand count, and seat
balancing remain fixed within each comparison. The injected model is read-only
during evaluation so the intervention cannot disappear through relearning.

## Primary estimand

The outcome is held-out mean chip payoff per hand, averaged across both seat
orders. The primary model-dependence contrast is:

```text
payoff(aligned model) - payoff(swapped model)
```

The secondary contrast replaces the swapped model with the prior model.
Replicates—not individual hands—are the inferential unit. Results are averaged
over all three target modes before normal 95% intervals are computed across
replicates.

## Frozen default design

- 16 independent replicates;
- 250 calibration hands per seat order and target mode;
- 500 evaluation hands per seat order, condition, focal mode, and target mode;
- calibration seed family `71001`;
- disjoint evaluation seed family `81001`;
- seed stride `1000`;
- Adaptive policy purity `.80` and temperature `.35`;
- donor mapping: Pressure receives Control's model, Control receives Chaos's
  model, and Chaos receives Pressure's model.

## Prespecified confirmation rule

Counterfactual Control is confirmed only if all four conditions pass:

1. Control's aligned-minus-swapped 95% interval is entirely above zero;
2. Control's aligned-minus-prior 95% interval is entirely above zero;
3. the replicate-level Control contrast minus the larger of the Pressure and
   Chaos contrasts has a 95% interval entirely above zero; and
4. Control has the largest aligned-minus-swapped mean for at least two of the
   three target modes.

The policies, mappings, thresholds, seeds, and outcomes will not be changed
after the default evaluation merely to obtain confirmation. A null result is a
valid boundary finding.

## Interpretation boundary

This experiment can establish causal model dependence in engineered synthetic
policies. It cannot establish PCC states in humans. Human poker histories
remain outside the project until the appropriate institutional determination.

## Frozen result

The default run completed without modifying the v0.3 policy source. It did not
confirm counterfactual Control.

Across 16 replicate-level averages, Control's aligned-minus-swapped payoff was
`.0372` chips per hand with a normal 95% interval of
`[-.0039, .0783]`. Its aligned-minus-prior payoff was `-.2115`, with an
interval of `[-.2395, -.1834]`. The Control-specific contrast relative to the
larger Pressure or Chaos effect was `-.0190`, with an interval of
`[-.0829, .0448]`. All four prespecified checks failed.

The effect was conditional on the target. Alignment strongly helped a
Control-heavy focal policy against Pressure (`.4639`), but hurt against
Control (`-.0816`) and Chaos (`-.2707`). A target-mode model learned through
a balanced probe therefore did not transport as a generally useful model for a
different focal policy. This is retained as a boundary result, not repaired by
changing the donor mapping, seeds, thresholds, or policies.
