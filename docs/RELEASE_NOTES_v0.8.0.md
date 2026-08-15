# PCC Poker v0.8.0 release notes

## Synthetic evidence freeze / pre-human protocol

v0.8.0 is the scientific boundary before confirmatory human HandHQ analysis. The release preserves the accumulated synthetic evidence, the negative and partial results as well as confirmations, and the preregistered human-analysis contract.

### Frozen scientific position

- **Pressure:** eligible as the only confirmatory human-facing PCC axis in this release.
- **Control:** causal/mechanistic synthetic evidence exists, but a family-invariant observational detector has not been established; human use remains exploratory.
- **Chaos:** behavioral signal exists, but cross-family discriminant invariance has not been established; human use remains exploratory.

The confirmatory Pressure panel is limited to the two components selected by the frozen family-invariance gate: `pressure_exposure` and `predicted_fold_probability`.

### Human-data gate

This release does not authorize confirmatory HandHQ analysis. Human-data work remains gated on the applicable Georgia Tech ORIA/IRB determination or approval. The future source scope and preprocessing rules are defined in `docs/HUMAN_ANALYSIS_PREREGISTRATION.md`, `docs/HUMAN_MEASUREMENT_CONTRACT.md`, and `docs/HUMAN_DATA_INGESTION_PROTOCOL.md`.

### Verification

Before tagging or publishing the release, run:

```bash
make preflight
```

The preflight workflow runs the test suite, immutable freeze verification, non-frozen audit/report regeneration under `build/audit/`, release metadata checks, and `git diff --check`.

For the scientific freeze alone:

```bash
python -m pcc_poker verify-freeze
```

### Distribution checklist

After `make preflight` succeeds on the intended release commit:

1. push the commit and confirm GitHub Actions pass;
2. tag that exact commit `v0.8.0`;
3. create the GitHub Release from that tag using these notes;
4. archive the exact release on Zenodo and record the DOI in the repository.

These distribution actions should not regenerate or modify the frozen scientific artifacts.
