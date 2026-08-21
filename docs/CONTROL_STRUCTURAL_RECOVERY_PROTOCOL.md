# Poker Control structural-recovery protocol

## Status

This is a **post-v0.8 synthetic extension**. It does not rewrite the v0.8.0 synthetic evidence freeze, does not modify the frozen human-facing Pressure-only panel, and does not access HandHQ or any other human data.

## Question

Can Poker Control be recovered across two synthetic implementation families as the same three-stage structure rather than as one universal scalar observable?

The prospective structural hypothesis is:

```text
information uptake -> context alignment -> value-sensitive intervention
```

## Frozen measurement definitions

All measurements are label-free at decision time. Synthetic PCC weights are consulted only after aggregation by policy family, mixture, and focal seat.

1. **Information uptake**: mean `log p(chosen | aligned public history) - log p(chosen | static public context)`.
2. **Context alignment**: mean `log p(chosen | aligned public history) - log p(chosen | context-yoked public history)`.
3. **Value-sensitive intervention**: positive context-alignment gain multiplied by the existing label-free counterfactual `control_efficiency` score, then averaged over the trajectory.

The context-yoked model preserves static-context action margins and global action margins while destroying temporal alignment.

## Families

The test is run independently in the `score` and `adaptive` synthetic policy implementations on fresh evaluation seeds.

## Prespecified stage criterion

For each stage and each family:

- correlation with assigned Control weight must be at least `0.20`; and
- the Control correlation must exceed the larger Pressure/Chaos correlation by at least `0.05`.

A stage replicates only if both families pass. Full structural recovery requires all three stages to replicate and the matched-yoke margin checks to pass.

A one-family success is retained as partial mechanism evidence. The metric or thresholds are not retuned after observing the result.

## Interpretation boundary

Passing would establish synthetic cross-family structural recovery only. It would not establish a human Control state and would not modify the preregistered human-data protocol without a separately documented prospective amendment after ORIA guidance.
