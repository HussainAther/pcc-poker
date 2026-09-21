# Seat-opportunity decomposition result

## Status

Prospective synthetic validation executed from the frozen `SEAT_OPPORTUNITY_DECOMPOSITION_PLAN.md` on fresh seeds 1301-1305 without retuning. The existing ordinary mixed-recovery training set was unchanged.

## Main finding

The new data reveal a large seat difference in the *availability* of facing-bet mixed-transition opportunities.

Across the five seeds, seat 0 averaged roughly 115-118 eligible transitions per example while seat 1 averaged only roughly 54-56. The asymmetry was concentrated in round 0: seat 0 averaged about 64-65 eligible round-0 transitions, versus only about 11-12 for seat 1. Round-1 counts were much closer (roughly 50-53 versus 42-45).

This is direct evidence that the two seats do not observe the target transition under equally informative sampling conditions in the synthetic environment.

## Ablation effect before opportunity standardization

The ordinary facing-bet `bet-call` feature remained positively useful in both seats on all five fresh seeds.

| Quantity | Seat 0 | Seat 1 |
|---|---:|---:|
| Mean delta MAE | 0.004995 | 0.006132 |
| Positive seeds | 5/5 | 5/5 |

Mean absolute seat-effect gap: `0.004154` MAE.

## Frozen fixed-budget standardization

The preregistered fixed budgets were 14 round-0 and 41 round-1 eligible transitions per example. Seat 0 almost always met these budgets, but seat 1 frequently did not because its round-0 opportunity count was structurally lower.

Seat-1 budget-shortfall fractions by seed were:

| Seed | Seat 1 shortfall fraction |
|---|---:|
| 1301 | 0.75 |
| 1302 | 0.80 |
| 1303 | 0.90 |
| 1304 | 0.80 |
| 1305 | 0.80 |

The frozen standardization therefore did **not** achieve full opportunity matching across seats. Per the plan, the budget was not changed after observing this result.

## Standardized ablation result

Under the frozen fixed-budget representation:

| Quantity | Seat 0 | Seat 1 |
|---|---:|---:|
| Mean delta MAE | 0.003736 | 0.001203 |
| Positive seeds | 5/5 | 4/5 |

Mean absolute seat-effect gap: `0.003924` MAE.

Gap attenuation was positive on 3/5 seeds. The median seed-level attenuation was `0.382` (38.2%), but the mean ratio was unstable and negative because seed 1304 had a very small ordinary denominator (`0.000575` MAE) before standardization. Absolute gaps are therefore more interpretable than the mean ratio here.

Per-seed absolute gaps:

| Seed | Ordinary gap | Standardized gap | Attenuation |
|---|---:|---:|---:|
| 1301 | 0.006893 | 0.000483 | 0.930 |
| 1302 | 0.004149 | 0.008965 | -1.161 |
| 1303 | 0.002818 | 0.001742 | 0.382 |
| 1304 | 0.000575 | 0.005436 | -8.450 |
| 1305 | 0.006335 | 0.002995 | 0.527 |

## Interpretation

The opportunity hypothesis receives clear descriptive support: seat 0 has about twice as many eligible facing-bet mixed-transition observations as seat 1, with an especially large round-0 imbalance.

The stronger causal question -- whether *equalizing* opportunity removes the seat-specific ablation gap -- remains unresolved by this run because the preregistered fixed budgets were infeasible for most seat-1 examples. The partial attenuation on 3/5 seeds is suggestive but cannot be treated as a clean matched-opportunity result.

Accordingly, the result narrows the next experiment: use a genuinely feasible, lower fixed budget chosen and frozen before another fresh seed block, or a paired per-yoke minimum-budget design that guarantees equal opportunity counts by construction. The current 1301-1305 results should not be reused to tune and retest that follow-up.

## Scientific boundary

This is synthetic construct validation in the engineered poker environment. It does not establish a human Control observable or a human psychological construct.
