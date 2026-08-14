# Contextual Control Observable Protocol

## Frozen candidate

The candidate Control observable is the mixture-level mean log-likelihood advantage of a public-history action model with correctly aligned history over a matched context-yoked history model.

The yoke shuffles observed actions only within the same static public decision context. Therefore current-state/action margins are preserved exactly while the association between public history and action is destroyed.

## Preregistered success rule

The candidate is supported only if, in **both score and adaptive policy families** on fresh evaluation seeds:

- correlation with Control weight is at least 0.20;
- the Control correlation exceeds both Pressure and Chaos correlations by at least 0.05; and
- the absolute cross-family Control-correlation gap is at most 0.20.

PCC weights are not inputs to the observable. They are consulted only after mixture-level aggregation for construct-validation correlations. Failure leaves observational Control unresolved; the metric or thresholds are not tuned after evaluation.
