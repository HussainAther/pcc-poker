# Pressure suppression of effective surprise

## Question

Does a label-free public-state measure of Pressure exposure explain the strong negative association between assigned Pressure and effective surprisal?

## Frozen measurement

For each chosen action, the action model is calibrated on disjoint synthetic hands. If the action leaves the opponent facing a wager, Pressure exposure is the mean of:

1. response-distribution compression;
2. predicted fold probability; and
3. commitment relative to the resulting pot.

Otherwise exposure is zero. Effective surprisal uses the unchanged independent value floor.

At the mixture/seat level, effective surprisal is linearly residualized against Pressure exposure within each policy family. This regression uses **no hidden PCC weights**. Assigned weights are consulted only after residualization.

## Prespecified support criteria

In **both** score and independent policy families:

- Pressure exposure correlates at least `.20` with assigned Pressure;
- pressure adjustment reduces the magnitude of the effective-surprisal/Pressure correlation by at least `.20`; and
- the Chaos discriminant margin improves by at least `.03`.

Failure of any criterion is retained. Seeds, thresholds, generators, action model, and value floor are not changed after the run.
