# Paired matched-opportunity falsification result

## Status

Prospective synthetic validation executed from the frozen `PAIRED_MATCHED_OPPORTUNITY_FALSIFICATION_PLAN.md` on fresh seeds 1401-1405 without retuning. The existing ordinary mixed-recovery record-level training set was unchanged. Paired-minimum matching was applied to both training and evaluation representations.

## Matching integrity

The paired-minimum design solved the feasibility problem from the preceding fixed-budget experiment.

For every latent yoke and each round, the retained transition budget was set to the minimum eligible count across the two seats. The exact selected-count match fraction was `1.0` on all five fresh seeds. Thus both seats contributed exactly the same number of retained opportunities per round for every evaluated yoke.

The large underlying opportunity imbalance nevertheless replicated descriptively. Across seeds, seat 0 had roughly 64-67 eligible round-0 transitions per example while seat 1 had roughly 10-12. Round-1 counts were much closer, roughly 50-52 for seat 0 and 41-44 for seat 1. The paired rule therefore retained the seat-1-limited round-0 budget in both seats, while round-1 budgets were usually close to the available counts in both seats.

## Primary recovery result

The ordinary facing-bet `bet-call` coordinate remained positively useful in both seats on all five fresh seeds. The same was true after exact paired opportunity matching.

| Representation | Seat 0 mean delta MAE | Seat 1 mean delta MAE | Mean absolute seat gap | Median absolute seat gap |
|---|---:|---:|---:|---:|
| Ordinary | 0.005614 | 0.007142 | 0.003011 | 0.003018 |
| Paired matched | 0.005982 | 0.005087 | 0.003657 | 0.004247 |

The mean absolute seat gap did not attenuate under exact matching. It increased from `0.003011` to `0.003657` MAE. The median absolute gap likewise increased from `0.003018` to `0.004247` MAE.

## Seed-level gaps

| Seed | Ordinary gap | Paired-matched gap | Gap attenuation |
|---|---:|---:|---:|
| 1401 | 0.004129 | 0.001536 | 0.628 |
| 1402 | 0.004675 | 0.004247 | 0.091 |
| 1403 | 0.000690 | 0.004336 | -5.285 |
| 1404 | 0.002544 | 0.001121 | 0.559 |
| 1405 | 0.003018 | 0.007046 | -1.334 |

Gap attenuation was positive on 3/5 seeds, with median attenuation `0.091`. The mean attenuation ratio was `-1.068`, but—as in the prior experiment—this ratio is unstable when the ordinary denominator is very small. Seed 1403 had an ordinary gap of only `0.000690` MAE, so absolute gaps remain the more interpretable primary quantities.

## Opportunity diagnostics

Mean paired round-0 budgets ranged from `10.2` to `11.95` transitions per example across seeds, exactly matching the seat-1 availability in each yoke by construction. Mean paired round-1 budgets ranged from `41.45` to `43.8`.

The conditional `bet-call` shares after matching did not collapse to a common value across seats. This is consistent with the fact that count matching equalizes the number of opportunities, not the conditional action composition or the trajectory context generating those opportunities.

## Interpretation

The opportunity-count hypothesis receives strong descriptive support but does not provide a sufficient explanation for the seat-specific recovery asymmetry.

Exact per-yoke, per-round matching removed the opportunity-count imbalance from the retained transition representation, yet the absolute seat gap did not consistently shrink and was larger on average. Therefore the prior seat effect cannot be attributed solely to unequal numbers of eligible facing-bet mixed transitions.

The next mechanistic target is the residual conditional-response/trajectory explanation: whether the two seats differ in the conditional composition, state context, or ordering structure of matched opportunities even when their retained counts are identical.

A suitable next prospective experiment should keep the current 1401-1405 results frozen and use a new seed block. It should test matched opportunities at a finer state/context level rather than retuning the paired-count rule.

## Scientific boundary

This remains synthetic construct validation in the engineered poker environment. It does not establish a human Control observable or a human psychological construct.
