# Prospective Score-Control contextual-response intervention

## Purpose

The frozen post-v0.8 Control structural-recovery experiment recovered all three
Control stages in the Adaptive family but none in the Score family. A subsequent
mechanism decomposition found that Adaptive Control's action distribution was
about 6.09 times more sensitive than Score Control to the same learned opponent
fold-probability perturbation.

This experiment makes **one prospective policy change** before rerunning the
existing three-stage gate.

## Intervention

For Score-family Control only, add to each aggressive (`bet`/`raise`) Control
score:

`3.35 * (opponent_fold_probability - 1/3)`

The gain is fixed as `0.55 * 6.09 ~= 3.35`, using the original Score Control
fold-leverage coefficient and the preceding frozen sensitivity ratio.

The term is zero at the neutral smoothed prior `1/3`, so the original Score
Control decision score is unchanged at that reference point.

## Frozen invariants

The intervention does **not** change:

- Score card/showdown-value terms;
- Score flexibility terms;
- Score commitment-risk penalties;
- Pressure or Chaos components;
- the Adaptive family;
- structural-recovery measurements;
- calibration/evaluation seeds;
- the minimum Control correlation (`0.20`);
- the minimum discriminant margin (`0.05`);
- the v0.8 human-facing Pressure-only panel or ORIA gate.

## Acceptance rule

The existing structural hypothesis remains:

`information uptake -> context alignment -> value-sensitive intervention`

Poker Control is structurally resolved by this extension only if **all three
stages pass in both Score and Adaptive families** under the already-frozen gate.
A partial result is retained as partial and is not promoted to the v0.8 human
measurement contract.
