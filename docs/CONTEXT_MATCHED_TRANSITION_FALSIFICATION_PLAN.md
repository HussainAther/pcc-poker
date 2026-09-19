# Context-matched transition falsification plan

## Status

Prospective post-v0.8 synthetic validation. This plan freezes the next test motivated by the exploratory transition decomposition in `CONTROL_MIXED_PAIR_STABILITY_RESULT.md`. It does not modify the v0.8 synthetic evidence freeze or the human-analysis preregistration.

## Fixed training set

Use `outputs/mixed-recovery-data.jsonl` as the unchanged ordinary-Dirichlet training distribution.

## Fresh evaluation seeds

Use five fresh Control-heavy OOD evaluation seeds:

- 1101
- 1102
- 1103
- 1104
- 1105

Each seed generates 20 independently sampled Control-heavy mixtures, both focal seats, with 100 hands per seat and focal/reference temperature 0.35.

## Primary hypotheses

1. **Facing-bet bet-call.** In transitions assigned to the public context of the second action, unordered `bet-call` adjacency contributes positively to Control-heavy recovery when that second action occurs while facing a bet, pooling both rounds.
2. **Round-1 open bet-check.** Unordered `bet-check` adjacency contributes positively to Control-heavy recovery in round index 1 while not facing a bet.

`call-raise` is not a primary endpoint because its exploratory effect weakened after context matching.

## Estimator and ablation

For each hypothesis, fit the existing standardized ridge/simplex recovery estimator using M2 plus the context-specific unordered mixed-pair block. Compare it with the same model after setting only the prespecified pair coordinate to zero in both training and evaluation examples. No feature re-normalization is performed after zeroing.

The primary effect is:

`delta_mae = ablated_mae - full_mae`

Positive values mean that removing the prespecified transition worsens recovery.

## Frozen replication criterion

A primary hypothesis replicates only if both conditions hold:

- mean `delta_mae` across fresh seeds is greater than zero; and
- at least 80% of fresh seeds have `delta_mae > 0` (4 of 5 for the frozen seed set).

No threshold will be changed after inspecting the fresh-seed results.

## Seat-asymmetry diagnostic

For each hypothesis and seed, compute the absolute difference between the seat-0 and seat-1 ablation effects. Compare that gap with the corresponding all-context mixed-pair ablation gap. The seat-asymmetry diagnostic passes when the mean matched-context gap across the two primary hypotheses is smaller than the mean all-context gap.

This diagnostic is secondary: it does not rescue a failed primary replication.

## Falsification

The contextual interpretation is weakened if either primary effect has a non-positive mean, fails the 4-of-5 sign-replication rule, or reverses consistently on fresh seeds. Failure of seat-gap attenuation argues against the specific claim that the previously observed seat asymmetry is primarily context-mediated.

## Command

```bash
python -m pcc_poker.context_matched_transition_validation \
  --seeds 1101 1102 1103 1104 1105 \
  --output validation/context-matched-transition-validation.json
```

## Scope warning

This is synthetic construct validation only. A positive result does not establish that human poker behavior follows PCC.
