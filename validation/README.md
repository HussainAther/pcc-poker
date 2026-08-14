# Frozen smoke-run outputs

These files record the first reproducible run of the prototype (August 13,
2026). They are included to make the ZIP inspectable without generating the
larger JSONL trajectory file.

- `recovery.json`: nearest-centroid recovery from 3,000 seat-balanced hands.
- `sweep.json`: pairwise policy results from 2,000 hands per seat ordering.
- `mixed-recovery.json`: continuous-weight recovery from 60 generated mixtures,
  including action-frequency and shuffled-target baselines.
- `mixed-grid-summary.json`: aggregate results from five seeds crossed with
  three focal-policy temperatures.
- `family-transfer-grid-summary.json`: bidirectional transfer across the
  original score family and an independently coded probability-mixture family.
- `behavioral-measures.json`: disjoint-calibration validation of label-free
  behavioral measurements. It retains the failed Control construct check.
- `control-confirmation.json`: fresh-seed confirmation of opponent-adaptive
  Control, including its discriminant check and inconclusive cross-family result.
- `adaptive-family.json`: v0.3 construct validation for the playable family.
  Pressure and Chaos exceed the descriptive threshold; observational Control
  recovery is retained as inconclusive.
- `adaptive-sweep.json`: v0.3 single-run seat-balanced diagnostic showing all
  three proposed directions.
- `balanced-cycle.json`: frozen 12-replicate confirmation of the engineered
  cycle, including replicate-level intervals and an edge-strength ratio.
- `robustness-grid.json`: complete frozen out-of-sample robustness surface.
  The global result is retained as failed (`36/48`, or `75%`, versus the frozen
  `80%` criterion), with stratified summaries and all twelve failure cells.
- `robustness-grid.csv`: one heatmap-ready row per parameter condition.
- `temporal-control.json`: frozen cross-fitted test of whether aligned prior
  opponent history improves action prediction and identifies synthetic Control.
  Temporal prediction succeeds, but the Control-specific construct test is
  retained as failed because the score is more associated with Pressure.

They are engineering validation, not empirical evidence about humans and not a
confirmatory PCC test. Regenerate them with the commands in the root README.
- `chaos-control-decomposition.json` — frozen fresh-seed decomposition of effective surprisal into public-history-explained and history-residual value-preserving components; partial/null result retained.

## Pressure suppression of effective surprise

`pressure-surprise-decomposition.json` is the frozen fresh-seed test of whether
a public-state Pressure exposure explains suppression of effective surprisal.
The overall confirmation is false: the score family shows a large Chaos-margin
improvement after Pressure adjustment, while the independent family's Chaos
margin decreases. The failed cross-family criterion is retained.
- `pressure-surprise-decomposition.json`: fresh-seed family-split null showing that public Pressure exposure suppresses effective surprise but does not universally explain Chaos/Control overlap.
- `family-invariant-panel.json`: fresh-seed cross-family selection gate. Only public Pressure exposure and predicted fold probability pass; Control and Chaos coverage remain empty.

## Contextual Control observable

`contextual-control-observable.json` is a frozen fresh-seed test of a public-only Control candidate: the mean log-likelihood advantage of correctly aligned public history over a matched context-yoked history model. The yoke preserves static-context and global action margins exactly.

Result: **not confirmed as family invariant**. Control correlation is positive and discriminant in both tested families (Adaptive: ~0.812; Score: ~0.210), but the cross-family gap (~0.602) exceeds the preregistered 0.20 maximum. Control therefore remains observationally unresolved for the conservative cross-family panel. The metric and thresholds were not retuned after evaluation.

## Reproducibility manifest

`reproducibility-manifest.json` fingerprints the current PCC source/protocol
surface and the frozen synthetic validation artifacts with SHA-256 hashes. It
also records environment metadata and, when requested, the pytest result. It is
an engineering provenance artifact only; it does not analyze human data.
