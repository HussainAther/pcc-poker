# Seat-opportunity decomposition plan

## Status

Prospective post-seat-yoked synthetic validation. This plan is frozen before inspecting any results from the fresh evaluation seeds listed below.

## Motivation

The seat-yoked validation showed that the facing-bet unordered `bet-call` transition contributes positively to Control-heavy mixture recovery in both focal seats, but the ablation effect is consistently larger in seat 1. Because the exact same latent PCC mixture was evaluated in both seats, unequal mixture draws cannot explain that asymmetry.

The next question is whether seat 1 receives a larger or more precise sample of informative transition opportunities, rather than the same opportunity carrying a different conditional relationship to the latent mixture.

## Prespecified hypotheses

1. **Opportunity hypothesis.** The two seats differ in the number and round composition of facing-bet mixed-transition opportunities.
2. **Opportunity-standardization hypothesis.** If unequal opportunity/round visitation is an important driver of the seat-effect gap, then forcing the `bet-call` transition feature to use the same fixed number of eligible transitions per round should attenuate the absolute seat gap relative to the ordinary all-opportunity feature.
3. **Conditional-response alternative.** If the seat gap remains similar after opportunity standardization, the remaining asymmetry is more consistent with differences in conditional response composition or other seat-conditioned trajectory/model effects than with unequal opportunity counts alone.

## Frozen design

- Training distribution: existing ordinary mixed-recovery dataset, unchanged.
- Evaluation region: Control-heavy OOD only.
- Fresh seeds: `1301, 1302, 1303, 1304, 1305`.
- Mixtures per seed: `20`.
- Hands per focal seat per mixture: `100`.
- Latent yoking: the same sampled PCC mixture is evaluated in both focal seats.
- Target transition: unordered `bet-call` adjacency whose second action occurs while facing a bet.
- No human data are accessed.

### Opportunity definition

An eligible opportunity is an adjacent pair of focal-policy decisions within the same hand such that:

- the second decision is facing a bet (`to_call > 0`), and
- the two actions form one of the mixed unordered transition categories used by the prior validation.

Opportunity counts are recorded separately for round 0 and round 1.

### Opportunity-standardized transition feature

The ordinary transition feature uses every eligible mixed transition and normalizes pair counts by the total eligible count.

The standardized feature uses a fixed, training-derived transition budget per round. Before fresh evaluation is inspected, the budget is frozen from the 5th percentile of eligible opportunity counts in the existing training examples:

- round 0: `14` transitions,
- round 1: `41` transitions.

Within each round, eligible transitions are deterministically ranked by a SHA-256 hash of stable trajectory identifiers and the first `K` are retained. This avoids tuning the retained sample to outcomes or action labels. Counts are then normalized over the pooled retained transitions from both rounds.

If an evaluation example has fewer than the frozen budget in either round, it is marked as a budget shortfall. The budget is not changed in response.

## Primary quantities

For each seed and each representation (ordinary and opportunity-standardized):

- seat 0 `delta_mae = ablated_mae - full_mae`;
- seat 1 `delta_mae`;
- signed seat difference;
- absolute seat-effect gap.

The primary decomposition statistic is:

`gap_attenuation = 1 - standardized_absolute_gap / ordinary_absolute_gap`.

Positive attenuation means standardizing transition opportunity reduced the seat gap. No binary confirmation threshold is imposed; the magnitude and sign are reported directly across all five seeds.

## Secondary diagnostics

For each seat and seed, report:

- mean eligible opportunity count per example, overall and by round;
- budget-shortfall fraction;
- ordinary conditional `bet-call` share among facing-bet mixed transitions;
- standardized conditional `bet-call` share after fixed-budget sampling.

These diagnostics describe mechanism and are not used to retune the feature or budget.

## Interpretation boundary

This experiment can distinguish a coarse opportunity/round-visitation explanation from a residual conditional-response/model explanation inside the synthetic environment. It does not establish a human Control observable, and failure to attenuate the gap does not identify a unique alternative mechanism.
