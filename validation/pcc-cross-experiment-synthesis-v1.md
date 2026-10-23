# PCC-Poker: Cross-Experiment Evidence Synthesis

Scope: computational policy-family transfer in the current Leduc simulator; not external validation.

## Primary findings

| Study | Prespecified result | Outcome |
|---|---|---|
| Patch 29/30 | Positive improvement both directions, all endpoints | Not met |
| Patch 31/32 | Directional asymmetry on value_vs_loose | 11.292832; CI [7.457350261936524, 16.06514383132766] |
| Patch 33 | PCC clipping reduces A-to-B error on value_vs_loose | 33.826887; CI [20.049220671762857, 51.879123962287956] (hash checks only) |
| Patch 34/35 | Matched-clipping incremental PCC improvement | -0.299244; CI [-0.7357179191232488, 0.1622282967663685]; criterion not met |

## Research interpretations

- **Universal bidirectional predictive improvement of PCC augmentation:** not supported by prospective Patch 29 criterion
- **Directional cross-family transfer asymmetry:** supported across eight simulated cohorts in Patch 31; motivated by earlier outcomes
- **Input-range extrapolation contributes to A-to-B failure:** supported by input clipping intervention in Patch 33; not causal identification of PCC constructs
- **Incremental PCC advantage under matched clipping:** not demonstrated on primary endpoint in Patch 34

## Important limitations

- Hypotheses were developed sequentially after earlier results; not independently preregistered.
- The audits share portions of the original experimental evaluators.
- Only eight in-silico cohorts for repeated-cohort experiments; bootstrap intervals are descriptive.
- Different studies use distinct fresh seeds and designs; estimates must not be pooled.
- Secondary endpoints lack multiplicity-controlled confirmatory inference.
- Failure to demonstrate incremental PCC benefit does not prove equivalence or impossibility.
- Results do not validate PCC constructs or extend automatically to human or organizational behavior.

All numerical statements are generated from local frozen evidence; see JSON report for provenance.
