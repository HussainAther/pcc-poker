# Live PCC Regime Detector — Exploratory Protocol

The interactive player includes a rolling online evidence layer for exploratory
Pressure-, Chaos-, and Control-like behavior. It is a visualization/debugging
instrument, not a validated latent-state classifier.

## No-hindsight constraint

For a human decision at time t, the detector uses only information available to
the human at that decision plus the human's prior actions. It does not use the
opponent's private card, the terminal payoff, or any future action.

## Decision-level evidence

- **Pressure-like evidence** emphasizes wager constraint faced by the player and
  whether the response is a constrained fold.
- **Chaos-like evidence** combines online action surprise and recent action
  diversity, gated by an own-information adequacy proxy so arbitrary randomness
  does not automatically count as Chaos.
- **Control-like evidence** combines initiative (bet/raise) and immediate wager
  leverage, moderated by the player's own-information equity.

All scores are bounded to [0, 1]. These formulas are transparent exploratory
operationalizations and are intentionally not tuned against hidden PCC labels.

## Rolling regime

The live panel averages decision-level evidence over a fixed recent window
(default 12 human decisions). A regime label is emitted only when the leading
score is sufficiently large and separated from the runner-up; otherwise the
state is `MIXED`.

Labels are always suffixed `-LIKE` to avoid claiming ground-truth latent modes.

## Telemetry

Human rows written by interactive play include:

- `live_regime_evidence`
- `live_regime_snapshot`

The session summary includes the final rolling `live_regime` snapshot.

## Scientific status

This detector is suitable for personal gameplay inspection, demonstrations, and
hypothesis generation. It is not confirmatory human-subject evidence and does
not revise the frozen v0.8 measurement claims.
