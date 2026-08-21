# Score-family Control mechanism decomposition

## Status

Post-v0.8 synthetic diagnostic only. No human data are accessed and no frozen human-facing definition is changed.

## Question

Why did the Score family fail all three Control structural-recovery stages while the Adaptive family passed?

## Frozen diagnostic

Hold representative Leduc public decision states fixed and perturb only the learned opponent fold probability from 0.10 to 0.90. Compare the resulting Control action distributions without modifying either policy.

Prespecified checks:

- Score mean absolute aggression shift <= 0.20.
- Adaptive mean absolute aggression shift >= 0.40.
- Adaptive mean total-variation sensitivity >= 3x Score sensitivity.

## Result

All checks passed. Score mean TV shift was about 0.164; Adaptive mean TV shift was about 0.998, a sensitivity ratio of about 6.09.

The Score Control objective is therefore primarily a card/value-state objective with comparatively weak learned-response gain. Adaptive Control explicitly amplifies learned opponent response and consequently expresses the information -> context -> value-sensitive intervention structure much more strongly.

## Interpretation

This localizes the previous structural-recovery split. It does **not** confirm Poker Control across families. The next justified step is a prospective Score-family Control intervention that increases contextual response sensitivity while preserving the value/risk guardrail, followed by the original structural-recovery gate unchanged.
