## Unreleased

- Add a post-v0.8 synthetic Control structural-recovery experiment using the frozen three-stage hypothesis `information uptake -> context alignment -> value-sensitive intervention`.
- Require positive, discriminant recovery independently in the Score and Adaptive implementation families; do not promote one-family success into the frozen human-facing panel.
- Freeze the first structural-recovery result as partial: Adaptive passes all three stages, while Score passes none; leave the v0.8 human-facing Control axis unresolved.
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


## Post-v0.8 synthetic extension: Score Control mechanism decomposition

- Added a fixed-state opponent-response perturbation diagnostic for Score vs Adaptive Control.
- Confirmed a ~6.1x action-distribution sensitivity gap, localizing the Score-family structural-recovery failure to weak contextual response gain.
- No policy, frozen v0.8 artifact, or human-data gate was changed.

## Post-v0.8 Score-Control contextual intervention

- Added one prospective zero-centered contextual-response gain to a post-freeze
  Score-Control extension; frozen v0.8 policy files remain unchanged.
- Re-ran the existing three-stage Control structural-recovery gate without
  changing thresholds, seeds, or measurements.
- Score recovered information uptake and context alignment but still failed
  value-sensitive intervention; retained the result as partial with no retuning.

## Post-v0.8 Score-Control value-sensitive decomposition

- Added a read-only decomposition of the remaining Score-family Control failure after the contextual-response intervention.
- Separate positive context gain, counterfactual efficiency, regret, and their value-weighted product without changing any policy.
- Diagnose whether the value guardrail attenuates Control-linked contextual gains in Control-heavy trajectories.
- Keep the frozen v0.8 human-facing panel and ORIA gate unchanged.

## Post-v0.8 Score-Control value-aware intervention

- Add one prospective value-aware intervention after the frozen Score-Control value bottleneck decomposition.
- Preserve the contextual response gain at `3.35` and gate its effect on aggressive actions using the already-frozen `0.80` counterfactual-efficiency threshold.
- Re-run the existing three-stage Control recovery gate without changing its seeds, yoking, measurements, or acceptance thresholds.
- Retain the negative/partial result: Score keeps context alignment, narrowly loses information uptake, and does not recover value-sensitive intervention.
- Perform no post-result retuning; keep the v0.8 human-facing Control axis unresolved and the ORIA gate closed.


## Post-v0.8 synthetic extension — strong Poker Chaos falsification

- Added a frozen predictable-vs-random-vs-Chaos falsification across Score and Independent policy families.
- Added an Adaptive exploiter calibrated only against predictable play and frozen before Chaos evaluation.
- Both existing Chaos families pass unpredictability, value-preservation, and exploitability-resistance checks.
- Uniform random play is maximally entropic but substantially worse strategically, directly supporting `Chaos != randomness`.
- No v0.8 frozen artifact, human-facing measurement contract, or human-data gate is modified.
