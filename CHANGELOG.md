## Unreleased

- Add a synthetic-only `oria-ingestion-preflight` command for the future HandHQ pipeline.
- Reject any input outside sentinel-marked `tests/fixtures/` before reading contents while the human-data gate is closed.
- Audit schema fields, identifier scrubbing, prohibited-field leakage, private-card exclusion, outcome isolation, and audit-output placement.
- Add `make oria-preflight` and include it in `make preflight`.

# Changelog

All notable changes to PCC Poker are documented here.

## v0.8.0 - Synthetic evidence freeze

### Scientific freeze

- Froze the pre-human synthetic evidence bundle and preregistered the future HandHQ analysis boundary.
- Restricted confirmatory human-facing measurement to the Pressure axis using the two components that survived the cross-family invariance gate.
- Kept Control and Chaos explicitly exploratory/unresolved for human measurement.
- Kept confirmatory human-data analysis closed pending the applicable Georgia Tech ORIA/IRB determination or approval.

### Reproducibility and reporting

- Added reproducibility fingerprints for frozen validation artifacts.
- Added a generated research-status inventory covering confirmed, partial, failed, and unresolved claims.
- Added immutable SHA-256 verification for the frozen evidence/protocol bundle.

### Release hardening

- Added `python -m pcc_poker verify-freeze` and CI enforcement.
- Added `python -m pcc_poker release-check` for read-only version, documentation, freeze, and whitespace checks.
- Added a `Makefile` with safe developer targets and a one-command `make preflight` workflow.
- Routed developer-generated reproducibility and research-status outputs to `build/audit/` so routine checks do not rewrite the frozen v0.8.0 evidence bundle.

No release-hardening change authorizes human-data analysis or changes a frozen scientific threshold, component definition, or claim status.
