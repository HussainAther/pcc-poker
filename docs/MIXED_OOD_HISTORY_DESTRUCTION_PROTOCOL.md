# Mixed OOD history-destruction protocol

## Status

Prospective post-v0.8 synthetic control. This protocol is frozen before inspecting the history-destruction result.

## Question

The frozen nested feature ablation found that adding within-hand action-transition history (M3) improved continuous PCC mixture-weight recovery over public state/context features alone (M2), with the largest late gain in the Control-heavy OOD region.

This experiment asks whether that M2 -> M3 gain depends on **temporal ordering**, rather than merely on the extra dimensionality or local action composition encoded by transition features.

## Frozen data

Use exactly the existing mixed-recovery training set and prespecified mixed OOD evaluation set:

- training: `outputs/mixed-recovery-data.jsonl`
- OOD evaluation: `outputs/mixed-ood-data.jsonl`

No new policy simulation, seed selection, region selection, coefficient tuning, or feature selection is allowed for this test.

## Models

- **M2:** action frequencies + public betting context + public state/intensity summaries.
- **M3-original:** M2 + observed within-hand action-transition frequencies.
- **M3-shuffled:** M2 + transition frequencies computed after independently permuting the focal action labels within each hand.

The M2 vector is identical in original and shuffled conditions.

## Yoked destruction control

For each mixture example and each permutation replicate:

1. keep all rows, hand IDs, decision slots, round indices, pot values, amounts to call, legal-action sets, seat, and all other public-state fields fixed;
2. keep the original M0-M2 feature vector fixed;
3. within each hand, permute the focal action sequence only inside the transition-feature extractor;
4. preserve exactly the per-hand action multiset and therefore the example-level action frequencies; and
5. replace only the transition-frequency block of M3 with frequencies from the permuted order.

The shuffled sequence is a feature-level negative control and is not replayed through the poker engine. It therefore need not represent a legally realizable counterfactual hand.

Use 25 deterministic permutation replicates with seeds `88001..88025`.

## Estimation

For every permutation replicate, fit the same standardized ridge estimator and simplex projection used in the frozen mixed-recovery analyses:

- train M3-shuffled on the permuted training features;
- evaluate it on the independently permuted OOD features generated with the same replicate seed;
- keep target continuous Pressure/Control/Chaos weights unchanged.

M2 and M3-original are computed once from the unmodified records.

## Primary estimands

Let:

- `G_original = MAE(M2) - MAE(M3-original)`
- `G_shuffled = MAE(M2) - mean[MAE(M3-shuffled)]`
- `attenuation = (G_original - G_shuffled) / G_original`

when `G_original > 0`.

Report the same quantities overall and for every prespecified OOD region.

## Prespecified focal region

The **Control-heavy** region is the primary regional test because the frozen ablation showed the largest M2 -> M3 gain there. All other OOD regions must still be reported, including the frozen balanced negative case.

## Frozen interpretation checks

Temporal-order evidence is supported if:

1. the original M3 improves over M2 overall;
2. at least 50% of the original overall M2 -> M3 gain is lost after within-hand order destruction; and
3. at least 50% of the original Control-heavy M2 -> M3 gain is lost after order destruction.

These are descriptive synthetic-mechanism criteria, not human construct-validation thresholds.

## Integrity checks

The implementation must verify for every shuffled example that:

- per-hand action counts are exactly preserved;
- M0/M1/M2 features are unchanged;
- only the M3 transition block changes;
- no forbidden private/latent fields are introduced as predictors.

## Interpretation boundary

A positive result would show that **ordering information itself** contributes to synthetic OOD mixture identifiability in this engineered policy family. It would not establish a human Control detector, a psychological mechanism, or cross-family invariance.
