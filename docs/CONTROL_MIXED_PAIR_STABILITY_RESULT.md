# Exploratory Control-heavy mixed-pair stability result

This post-freeze exploratory probe decomposes the mixed-OOD M3 history advantage without changing the frozen v0.8 synthetic evidence.

## Main finding

For the Control-heavy OOD region, M2 MAE is 0.126342, the mixed unordered-pair model reaches 0.077702, the full unordered-pair model reaches 0.070864, and M3 reaches 0.066033. The mixed-pair signal is therefore a substantial part of the Control-heavy recovery advantage.

Under fixed-normalization single-feature zero-out, the largest degradations are bet-call (+0.008760 MAE), bet-check (+0.005769), and call-raise (+0.004191). Zeroing those three cumulatively degrades Control-heavy MAE by +0.019462 relative to the mixed-pair baseline.

## Seat-wise stability

The cumulative top-three zero-out degrades recovery in both focal seats:

- seat 0: 0.060776 -> 0.072444 (+0.011668)
- seat 1: 0.094628 -> 0.121884 (+0.027256)

The signal is therefore directionally stable across seats but materially stronger in seat 1. It should not be described as seat-invariant.

## Interpretation

The current exploratory interpretation is that Control-heavy recovery is largely associated with local unordered action structure, with important contribution from bet-call, bet-check, and call-raise adjacency. The seat asymmetry suggests position/context sensitivity and motivates a matched-seat or interaction analysis before any stronger mechanistic claim.

This remains exploratory and does not alter the frozen confirmatory claims.
