# Frozen Poker Control Architecture Result

> Post-v0.8 synthetic architecture falsification; no human data are used.

## Result

- Additive standardized LOAO MAE: **0.3185**
- `Control x context` standardized LOAO MAE: **0.3315**
- Relative improvement: **-4.09%**
- Primary result: **FAIL**

## Interaction controls

- `Pressure x context`: **7.93%** relative MAE improvement
- `Control x context`: **-4.09%** relative MAE improvement
- `Chaos x context`: **3.92%** relative MAE improvement

- Targets improved by `Control x context`: **1/4** (action_entropy)
- Control interaction disproportionately strongest: **False**

## Interpretation

The Blotto v1.1 architecture does **not** transfer unchanged to Leduc poker under this frozen design. The Control interaction worsens held-out prediction, while the Pressure interaction improves it most strongly. Poker therefore becomes evaluable in the cross-game architecture study as a **negative game-native result**, not as missing evidence.

This does not invalidate Poker Control as a construct. It rejects the narrower claim that Control is the uniquely or disproportionately context-modulatory component in this Leduc implementation and measurement regime.

## Evidentiary boundary

This evaluates architectural organization in engineered synthetic PCC-mixture agents; unlike Blotto v1.1 it is not an emergence-without-latent-PCC experiment.
