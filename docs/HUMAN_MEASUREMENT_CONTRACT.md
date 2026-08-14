# Frozen human measurement contract

## Status

This contract is frozen for PCC Poker `v0.8.0` before confirmatory HandHQ analysis. It defines which PCC quantities may be treated as confirmatory versus exploratory.

## Confirmatory axis: Pressure

Only the two components that survived the preregistered two-family invariance gate are eligible for confirmatory human analysis:

1. **Predicted fold probability** — the held-out public action model's probability that the responding opponent folds after facing a wager.
2. **Pressure exposure** — the fixed equal-weight mean of public response compression, predicted fold probability, and commitment ratio, as defined by the synthetic Pressure-surprise protocol.

For human Hold'em, the response model may use the sanitized public state representation appropriate to the game, but the component definitions and equal weighting may not be changed after confirmatory evaluation is inspected. Model fitting is confined to calibration data.

The two components are analyzed separately. A new optimized composite is not a confirmatory endpoint.

## Control

No family-invariant Control observable is selected. The matched public-history likelihood contrast and related contextual-history quantities may be reported only as **exploratory** diagnostics. A positive human result cannot retroactively promote them to confirmatory status in this release.

## Chaos

No family-invariant Chaos observable is selected. Raw behavioral surprisal, value-weighted/effective surprisal, and pressure-adjusted variants may be reported only as **exploratory** diagnostics unless a separately preregistered Hold'em value model and a new synthetic/human validation release are completed first.

## Prohibited inputs

Confirmatory human measurements must not use:

- source player handles or source table names;
- source hand IDs as predictive features;
- exact timestamps, dates, or timezone metadata;
- future actions or terminal outcomes when constructing a pre-decision state;
- private cards unavailable to the acting player;
- synthetic PCC labels or hidden mixture weights; or
- named-player identity as a predictive feature.

Study-specific pseudonymous IDs may be used only for grouping, leakage prevention, and cluster-aware inference.

## Leakage boundary

Every decision feature must be computable from information available immediately before the focal decision, except the Pressure response quantities, which are defined for the focal action and the opponent response state and are evaluated on held-out responses. Calibration/evaluation grouping must prevent the same persistent player identity from crossing the model-fitting/evaluation boundary when persistent obfuscated identifiers are retained under the approved protocol.
