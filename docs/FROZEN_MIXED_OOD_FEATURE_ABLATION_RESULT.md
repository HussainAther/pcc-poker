# Frozen Mixed OOD Feature-Ablation Result

## Status

Completed prospective synthetic decomposition of the previously frozen mixed OOD recovery advantage.

The experiment reused the frozen ordinary-mixture training data and prespecified OOD evaluation data. No OOD examples were used for fitting, and no private-card, equity, outcome, hidden-weight, component-score, policy-probability, or seed fields were used as predictors.

## Overall result

| Model | Observable information | MAE | Incremental change |
|---|---|---:|---:|
| M0 | action frequencies | 0.14162 | baseline |
| M1 | + public betting context | 0.11455 | -19.11% |
| M2 | + public state/value intensity | 0.11035 | -3.67% |
| M3 | + sequential history | 0.09339 | -15.36% |

Every added layer improved overall OOD MAE relative to the immediately preceding model.

The largest individual gain came from public betting context (M0 -> M1), followed by sequential action history (M2 -> M3). Public state/value-intensity summaries produced a smaller but positive incremental gain.

M3 exactly reproduces the observable information content of the prior contextual-history model, with feature ordering changed only for nested decomposition. Its MAE therefore matches the frozen mixed OOD contextual result.

## Regional result

| Region | M0 | M1 | M2 | M3 | Best |
|---|---:|---:|---:|---:|---|
| Pressure-heavy | 0.14107 | 0.08490 | 0.08003 | 0.07514 | M3 |
| Control-heavy | 0.16382 | 0.13293 | 0.12634 | 0.06603 | M3 |
| Chaos-heavy | 0.23157 | 0.17965 | 0.17312 | 0.15711 | M3 |
| Balanced | 0.04830 | 0.07277 | 0.07243 | 0.07549 | M0 |
| Boundary | 0.12334 | 0.10252 | 0.09981 | 0.09319 | M3 |

The full contextual model is best in four of five prespecified OOD regions.

The balanced region remains the informative negative case: every contextual extension is worse than marginal action frequencies. This failure was retained without tuning or reweighting.

The Control-heavy region shows the largest late gain from sequential history: M2 MAE falls from 0.12634 to 0.06603 under M3. This indicates that trajectory ordering contains especially strong information for identifying Control-heavy mixtures in this engineered family.

## Interpretation

Within this synthetic PCC policy family, the OOD recovery advantage decomposes into at least two substantial observable sources:

1. **situational betting context** — what actions occur under different public betting conditions; and
2. **sequential dynamics** — how actions transition within hands.

Coarse public state/value-intensity summaries add a smaller independent improvement.

The result also sharpens the balanced-region interpretation. Context is not uniformly useful across the simplex: near the center, richer features add variance without enough directional signal to compensate, while marginal action frequencies remain the lower-error estimator under this fixed design.

That balanced-region explanation remains mechanistic interpretation, not a universal theorem.

## What this supports

The experiment supports the narrower synthetic claim that the previously observed mixed OOD contextual advantage is not attributable to a single aggregate statistic. Public betting context and sequential action history each contribute materially to continuous PCC mixture-weight identifiability under distribution shift.

## What this does not establish

This experiment does not establish that:

- human poker behavior follows PCC;
- private strength or psychological state has been inferred;
- the same decomposition transfers to other policy implementations;
- context is always superior to action frequencies;
- or the feature contributions are causal psychological mechanisms.

No frozen v0.8 human-facing measurement contract is changed.

## Frozen conclusion

The feature-ablation hypothesis is supported.

```text
M0 action frequencies:          0.14162
M1 + betting context:           0.11455
M2 + public state intensity:    0.11035
M3 + sequential history:        0.09339
```

Incremental relative MAE improvements:

```text
M0 -> M1: 19.11%
M1 -> M2:  3.67%
M2 -> M3: 15.36%
```

The balanced-region negative result remains frozen:

```text
M0 MAE: 0.04830
M3 MAE: 0.07549
```

No post-result coefficient, region, feature, or threshold tuning was performed.

## Next experiment

The strongest next question is whether the sequential-history gain is genuinely temporal or merely a high-dimensional encoding of local action composition.

A prospective history-destruction test should compare M3 with yoked controls that preserve per-hand action counts and public-state margins while permuting within-hand action order. The key estimand is the loss of OOD recovery accuracy after sequence order is destroyed, especially in the Control-heavy region where the M2 -> M3 gain is largest.
