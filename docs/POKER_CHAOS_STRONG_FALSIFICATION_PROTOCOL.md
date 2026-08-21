# Poker Chaos strong falsification protocol

## Purpose

This post-v0.8 synthetic extension tests the cross-game proposition that **Chaos is not randomness**. It does not modify the v0.8.0 synthetic evidence freeze, the Pressure-only human-facing contract, or the ORIA human-data gate.

The test asks whether each existing Poker Chaos implementation combines three properties:

1. **behavioral unpredictability** relative to a deterministic value baseline;
2. **strategic adequacy** relative to uniform random legal play; and
3. **resistance to exploitation** by an opponent selected without observing Chaos outcomes.

No human data are accessed.

## Baselines and candidates

- **Predictable value baseline:** chooses the highest exact Leduc information-set action value under the already-existing uniform-continuation value model. Ties are resolved deterministically.
- **Uniform-random baseline:** samples uniformly from legal actions. This deliberately represents randomness without strategic adequacy.
- **Score Chaos:** the existing Score-family `(0.1, 0.1, 0.8)` policy at temperature `0.35`.
- **Independent Chaos:** the existing Independent-family `(0.1, 0.1, 0.8)` policy at temperature `0.35`.

The Chaos policies are not changed for this experiment.

## Exploiter calibration

A fixed grid of five Adaptive-family opponents is evaluated **only against the predictable value baseline**. The opponent producing the lowest seat-balanced predictable-baseline payoff is selected and frozen. Random and Chaos outcomes are not inspected during selection.

Calibration uses 500 hands per seat order beginning at seed `181001`.

## Held-out evaluation

Evaluation uses six fresh replicates, 400 hands per seat order per comparison, with disjoint neutral and exploiter seed streams beginning at `191001` and `201001` and stride `40`.

For each focal policy we record:

- mean normalized policy entropy;
- mean payoff against the fixed neutral Adaptive mixture; and
- mean payoff against the frozen exploiter.

## Prespecified family-level checks

Each Chaos family must pass all five:

1. normalized policy entropy is at least `0.20` above predictable play;
2. payoff versus neutral is at least `0.30` above uniform random;
3. payoff versus neutral is no more than `0.10` below predictable play;
4. payoff versus the frozen exploiter is at least `0.10` above predictable play; and
5. payoff versus the frozen exploiter is at least `0.30` above uniform random.

Before these checks can confirm the experiment, the selected exploiter must reduce the predictable baseline to mean payoff `<= -0.20` during calibration.

## Confirmation rule

`poker_chaos_strong_falsification_confirmed = true` only if the exploiter calibration requirement passes and **both** Score and Independent Chaos pass all five family-level checks.

This supports only the exact synthetic structural claim tested here. It does not by itself alter the frozen v0.8 human-facing Chaos status.
