# Synthetic evidence freeze — v0.8.0

## Purpose

Version `0.8.0` is the pre-human-analysis evidence freeze for PCC Poker. It closes the synthetic construct-development phase before any confirmatory analysis of the HandHQ human poker records.

The freeze is intentionally asymmetric. The synthetic evidence supports a conservative cross-family **Pressure** panel, but does not yet support family-invariant observational measures for **Control** or **Chaos**. Those missing axes remain missing.

## Frozen scientific status

`validation/RESEARCH_STATUS.md` and `validation/research-status.json` are the canonical claim inventory at this release. They preserve confirmed, partial, failed, and unresolved results, including null findings.

The conservative human-facing panel is fixed by `validation/family-invariant-panel.json`:

- Pressure: `pressure_exposure`, `predicted_fold_probability`.
- Control: no confirmatory observable selected.
- Chaos: no confirmatory observable selected.

The Control-over-Pressure contextual mechanism is synthetically confirmed, but this does not license a family-invariant human Control detector. Effective surprisal contains Chaos-related signal, but its discriminant invariance is unresolved.

## Freeze boundary

After this release, human results must not be used to:

- alter synthetic acceptance thresholds;
- change the mathematical definition of a frozen component;
- select friendlier synthetic seeds;
- reclassify a frozen null as a confirmation;
- add a Control or Chaos confirmatory endpoint post hoc; or
- tune the human pipeline against the confirmatory evaluation partition.

Any scientifically necessary change after the freeze requires a new version plus a dated amendment written **before** examining the affected confirmatory endpoint.

## Human-data gate

This release does not authorize human-data analysis. Confirmatory HandHQ analysis begins only after the applicable Georgia Tech ORIA/IRB determination or approval. Engineering may continue with invented PHH fixtures and mock records.

## Reproduction

Run:

```bash
python -m pcc_poker research-status
python -m pcc_poker reproduce --run-tests
python -m pcc_poker synthetic-freeze
```

The final command writes `validation/synthetic-freeze-manifest.json`, which hashes the frozen evidence/protocol bundle and records the seed inventory and human-analysis gate.
