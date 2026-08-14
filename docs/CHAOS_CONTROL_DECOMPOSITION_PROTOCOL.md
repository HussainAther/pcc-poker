# Chaos–Control entanglement decomposition protocol

## Question

The frozen effective-Chaos validation found positive Chaos correlations in both
policy families but failed discriminant validity because the independent
family's effective surprisal correlated slightly more strongly with Control.
This follow-up asks **why**, without changing either generator or the value
floor.

## Frozen decomposition

A public action model is calibrated on separate mixtures and seeds in two forms:

1. a **static** model conditioned on the current public betting context;
2. a **temporal** model conditioned on the same context plus prior public action
   history (capped action counts and the two most recent public actions).

For an observed action,

- static surprisal is `-log p_static(a)`;
- temporal surprisal is `-log p_temporal(a)`;
- **history-explained surprisal** is `max(0, static - temporal)`;
- **history-residual surprisal** is temporal surprisal.

All quantities are normalized by the legal-action entropy scale and multiplied
by the *unchanged* `UniformContinuationValueModel` adequacy floor.

The interpretation is intentionally narrow. History-explained surprisal is a
candidate for prediction-sensitive mixing; history-residual surprisal is a
candidate for value-preserving stochasticity that remains after public-history
conditioning. Neither is relabeled as Control or Chaos merely because a
correlation is favorable.

## Leakage boundary

Calibration and measurement use public betting state and prior public actions.
They exclude PCC weights, policy labels, component scores, generator action
probabilities, outcomes, future events, and actual opponent private cards.
Hidden mixture weights are consulted only after seat/mixture aggregation for
construct-validity correlations.

## Fresh seeds and prespecified checks

Default calibration seeds are `2001` (score) and `2009` (independent). Default
evaluation seeds are `2201` and `2209`.

The decomposition is supported only if, in the independent family:

1. history-residual effective surprisal correlates at least `.20` with Chaos;
2. its Chaos correlation exceeds both off-axis correlations by at least `.03`;
3. history-explained effective surprisal is Control-discriminant by at least
   `.03`; and
4. the residual Chaos discriminant margin improves on the original static
   effective-surprisal margin by at least `.03`.

Failure is retained without changing the contexts, thresholds, seeds,
generators, or value floor.

## Frozen result

The default fresh-seed run is retained as a **partial/null decomposition**.

In the independent family, history-residual effective surprisal remains
Chaos-discriminant (`r_Chaos = .554`, `r_Control = .499`, margin `.054`) and
passes the minimum Chaos correlation and discriminant checks.  However,
history conditioning does not improve the Chaos margin: the residual margin is
`.007` lower than the static effective-surprisal margin.  The history-explained
piece is only weakly Control-discriminant (margin `.0236`), below the frozen
`.03` criterion.

In the score family, the pattern is different.  History-explained effective
surprisal is strongly Control-specific (`r_Control = .596` versus
`r_Chaos = -.152`), while the history-residual quantity remains more correlated
with Control than Chaos.  Public-history conditioning therefore separates a
Control-linked component in that implementation family but does not provide a
cross-family explanation for the earlier Chaos/Control entanglement.

Two of four prespecified independent-family checks pass, so
`chaos_control_entanglement_decomposition_supported` is `false`.  No contexts,
thresholds, seeds, generators, or value-floor parameters were changed after the
run.
