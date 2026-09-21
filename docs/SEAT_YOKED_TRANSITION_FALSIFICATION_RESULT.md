# Seat-yoked transition falsification result

## Status

Prospective post-context-matched synthetic validation following `SEAT_YOKED_TRANSITION_FALSIFICATION_PLAN.md` without retuning. The ordinary mixed-recovery training set remained fixed, and the prespecified fresh seeds 1201-1205 were used once.

## Primary result

The facing-bet unordered `bet-call` transition retained a positive recovery contribution in both focal seats under the yoked latent-mixture design.

| Seat | Mean delta MAE | Positive seeds |
|---|---:|---:|
| 0 | 0.005455 | 5/5 |
| 1 | 0.011864 | 5/5 |

Positive `delta_mae = ablated_mae - full_mae` means removing the prespecified transition worsened PCC mixture recovery. The effect therefore replicated directionally in both seats on every fresh seed.

## Seat-dependence diagnostic

Across the five prespecified seeds:

- mean signed seat difference (`seat 0 - seat 1`) = `-0.006409` MAE;
- mean absolute seat gap = `0.006409` MAE;
- same-direction fraction = `1.0`.

Per-seed seat gaps were:

| Seed | Seat 0 delta MAE | Seat 1 delta MAE | Absolute gap |
|---|---:|---:|---:|
| 1201 | 0.006542 | 0.009194 | 0.002652 |
| 1202 | 0.006467 | 0.011130 | 0.004663 |
| 1203 | 0.005517 | 0.010387 | 0.004870 |
| 1204 | 0.006526 | 0.013184 | 0.006658 |
| 1205 | 0.002224 | 0.015427 | 0.013202 |

Seat 1 showed the larger ablation effect on all five seeds.

## Interpretation

The replicated facing-bet `bet-call` signal is not explained simply by different latent PCC mixture draws across seats. The exact same target mixture was evaluated in both focal seats, yet the seat-specific effect gap persisted and had the same direction on every seed.

The transition can therefore be described as a robust synthetic Control-heavy recovery feature across both seats, but **not** as seat invariant. The remaining asymmetry must arise from some other part of the seat-conditioned interaction or observation process, such as action opportunity structure, public-state visitation, opponent response dynamics, or the feature-to-target mapping induced by seat-specific trajectories.

This result narrows the next mechanistic question from "is the asymmetry caused by unequal mixture draws?" to "which seat-conditioned trajectory property amplifies the same transition signal?"

## Scientific boundary

This is synthetic construct validation in the engineered poker environment. It does not establish a human Control observable or any human psychological construct.
