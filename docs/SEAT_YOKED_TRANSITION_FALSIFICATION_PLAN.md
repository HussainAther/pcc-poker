# Seat-yoked transition falsification plan

## Status

Prospective post-context-matched synthetic validation.

The previous fresh-seed experiment replicated the
facing-bet `bet-call` transition signal but failed the
secondary prediction that context matching would attenuate
seat-specific ablation-effect asymmetry.

This experiment tests whether that asymmetry persists when
the exact same latent PCC mixture is evaluated in both focal
seats.

## Fixed training set

Use:

`outputs/mixed-recovery-data.jsonl`

without modification.

## Fresh evaluation seeds

Use:

- 1201
- 1202
- 1203
- 1204
- 1205

These seeds have not been used in the preceding prospective
validation.

Each seed contains:

- 20 Control-heavy mixtures
- both focal seats
- 100 hands per focal seat
- focal temperature 0.35
- reference temperature 0.35

## Yoked design

For each mixture index, sample one Control-heavy PCC mixture.

Evaluate that exact mixture twice:

1. focal policy in seat 0;
2. focal policy in seat 1.

The target PCC weights must therefore be identical for the
two seat observations belonging to each yoke.

Simulation randomness remains independent across seats.

## Prespecified transition

Only the previously replicated transition is tested:

- unordered `bet-call`
- second action occurs while facing a bet
- both rounds pooled

No additional transition will be promoted after inspection
of these results.

## Primary estimand

For each focal seat:

`delta_mae = ablated_mae - full_mae`

where only the facing-bet `bet-call` coordinate is zeroed.

Positive values mean removal of the transition worsens PCC
mixture recovery.

## Primary question

Does the replicated transition retain a positive recovery
effect in both seats under a yoked latent mixture design?

## Seat-dependence diagnostic

For every seed calculate:

`seat_difference = delta_mae_seat0 - delta_mae_seat1`

and

`absolute_seat_gap = abs(seat_difference)`

Report:

- mean effect in each seat;
- positive-seed fraction in each seat;
- mean signed seat difference;
- mean absolute seat gap.

## Interpretation

If the transition remains positive in both seats but the
seat gap persists, the previous asymmetry cannot be
attributed simply to different latent PCC mixture draws
between seats.

If one seat loses the effect under yoking, the transition
should not be treated as a seat-invariant Control
observable.

If both seats lose the effect, the earlier result is
weakened and should not be rescued by post-hoc feature
search.

## Scope warning

This experiment uses synthetic poker behavior only.

It does not establish that human poker players exhibit PCC.
