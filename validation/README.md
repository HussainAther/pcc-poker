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
- `adaptive-family.json`: construct validation for the playable Control-v2
  family, with an explicit circularity warning.
- `adaptive-sweep.json`: initial seat-balanced game-balance diagnostic; it does
  not show the complete proposed cycle.

They are engineering validation, not empirical evidence about humans and not a
confirmatory PCC test. Regenerate them with the commands in the root README.
