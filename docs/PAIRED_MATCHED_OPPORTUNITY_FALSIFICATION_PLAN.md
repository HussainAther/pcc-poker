# Paired matched-opportunity falsification plan

## Status

Prospective post-seat-opportunity synthetic validation. This plan is frozen before inspecting any results from the fresh evaluation seeds listed below.

## Motivation

The seat-opportunity decomposition established a large seat asymmetry in the availability of facing-bet mixed-transition opportunities, concentrated in round 0. Its preregistered fixed budgets (14 round-0, 41 round-1 transitions) were infeasible for most seat-1 examples, so the stronger causal question remained unresolved: does equalizing opportunity attenuate the seat-specific `bet-call` recovery gap?

A second global fixed budget would either remain vulnerable to shortfalls or would discard more information than necessary. The present design instead matches opportunity within each latent yoke by construction.

## Prespecified hypotheses

1. **Matched-opportunity attenuation hypothesis.** If unequal opportunity count is an important driver of the seat-specific ablation gap, the absolute seat gap should shrink when both seats in each latent yoke contribute exactly the same number of eligible transitions in each round.
2. **Residual conditional-response hypothesis.** If a substantial seat gap persists under exact opportunity matching, the residual asymmetry is more consistent with conditional response composition, trajectory structure, or another seat-conditioned mechanism than with opportunity count alone.

## Frozen design

- Training distribution: existing ordinary mixed-recovery dataset, unchanged at the record level.
- Training representation: paired-minimum opportunity matching is applied to the two seat examples sharing each training `mixture_id`.
- Evaluation region: Control-heavy OOD only.
- Fresh evaluation seeds: `1401, 1402, 1403, 1404, 1405`.
- Mixtures per seed: `20`.
- Hands per focal seat per mixture: `100`.
- Latent yoking: the same sampled PCC mixture is evaluated in both focal seats.
- Target transition: unordered `bet-call` adjacency whose second action occurs while facing a bet.
- No human data are accessed.

### Eligible opportunity

An eligible opportunity is an adjacent pair of focal-policy decisions within the same hand such that:

- the second decision is facing a bet (`to_call > 0`), and
- the two actions form one of the mixed unordered transition categories used by the prior validations.

Opportunities are partitioned by the round index of the second decision.

### Paired-minimum matching rule

For each paired seat yoke and each round `r`:

`K_r = min(n_seat0,r, n_seat1,r)`

where `n_seatX,r` is the number of eligible transitions available to that seat in that round.

Each seat's eligible transitions are deterministically ranked by the existing SHA-256 stable trajectory key, which excludes the transition action-pair label. Exactly the first `K_r` transitions are retained for each seat. Thus selected opportunity counts are identical across seats by construction for every yoke and round.

No minimum positive budget is imposed. If `K_r = 0`, both seats contribute zero transitions for that round. The observed `K_r` values are diagnostics only and are not used to retune the procedure.

### Representations

Two recovery representations are compared using the same target and ridge procedure:

1. **Ordinary:** all eligible facing-bet mixed transitions.
2. **Paired matched:** only the deterministically retained paired-minimum transitions.

For both representations, the `bet-call` coordinate is ablated by setting only that coordinate to zero while all other features remain unchanged.

## Primary quantities

For each seed and representation:

- seat 0 `delta_mae = ablated_mae - full_mae`;
- seat 1 `delta_mae`;
- signed seat difference;
- absolute seat-effect gap.

The main quantity is the absolute paired-matched seat gap, compared descriptively with the ordinary absolute seat gap.

A secondary summary is:

`gap_attenuation = 1 - matched_absolute_gap / ordinary_absolute_gap`

When the ordinary gap is numerically zero, attenuation is reported as undefined rather than stabilized or retuned.

## Secondary diagnostics

For each seat and seed, report:

- mean eligible opportunity count overall and by round;
- mean paired budget `K_r` by round;
- mean selected transition count by round;
- exact-match verification rate for selected counts across paired seats;
- ordinary and matched conditional `bet-call` shares.

## Interpretation boundary

Positive attenuation is evidence that opportunity-count imbalance explains part of the synthetic seat effect. Persistence or amplification of the gap under exact matching argues against opportunity count as a sufficient explanation, but does not uniquely identify the residual mechanism.

This remains synthetic construct validation in the engineered poker environment. It does not establish a human Control observable or human psychological construct.
