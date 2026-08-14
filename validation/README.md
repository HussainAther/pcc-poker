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

They are engineering validation, not empirical evidence about humans and not a
confirmatory PCC test. Regenerate them with the commands in the root README.
