# Prospective Mixed OOD Feature-Ablation Protocol

## Question

The frozen mixed OOD result showed that a contextual-history recovery model estimated continuous synthetic Pressure-Control-Chaos mixture weights more accurately than action frequencies overall and in four of five prespecified OOD simplex regions. Which observable information layer produces that advantage?

This is a synthetic identifiability decomposition. It does not test whether humans instantiate PCC.

## Frozen inputs

Reuse the already-generated distributions without changing their mixture weights or evaluation regions:

- training: `outputs/mixed-recovery-data.jsonl` (ordinary Dirichlet mixtures)
- OOD evaluation: `outputs/mixed-ood-data.jsonl`
- regions: Pressure-heavy, Control-heavy, Chaos-heavy, balanced, boundary

No OOD example may be used for model fitting.

## Nested observable models

The feature sets are strictly nested:

- **M0 — action frequencies:** marginal check/bet/fold/call/raise rates.
- **M1 — public betting context:** M0 plus action rates stratified by round and whether the player is facing a bet.
- **M2 — public state/value intensity:** M1 plus mean pot, amount to call, legal-action count, round index, decisions per hand, and seat.
- **M3 — sequential history:** M2 plus within-hand action-transition frequencies.

M3 contains the same observable information as the original contextual-history model, only with a different feature ordering.

## Leakage prohibition

The following stored synthetic fields are forbidden predictors even though some exist in the raw rows:

- private cards / `private_rank`
- `showdown_equity`
- terminal payoff / future outcome
- hidden PCC target weights
- component scores
- policy action probabilities
- simulation seeds

This means the recommended "strength/equity" layer from the prior OOD result is operationalized conservatively as **public state/value intensity**, rather than by introducing private-card-derived equity leakage.

## Estimator and target

For each M0-M3 model, use the same standardized ridge estimator and simplex projection used by the frozen mixed-recovery analyses. Predict the three continuous synthetic PCC mixture weights.

Primary metric: overall OOD mean absolute error (MAE).

Secondary metrics: RMSE, per-mode MAE, dominant-mode accuracy, and MAE within each prespecified OOD region.

## Prespecified comparisons

Report, without retuning:

1. M0 -> M1 MAE change;
2. M1 -> M2 MAE change;
3. M2 -> M3 MAE change;
4. the best overall model by MAE;
5. the best model separately in each OOD region.

The balanced region is an informative negative case and must not be removed, reweighted, or tuned away.

## Confirmation boundary

The decomposition supports a contextual contribution if at least one M1-M3 model improves overall OOD MAE relative to M0. A layer is described as incrementally helpful only if adding that layer lowers MAE relative to the immediately preceding model.

No result from this experiment modifies the frozen v0.8 human-facing measurement contract.
