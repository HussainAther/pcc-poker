# v0.8.0 synthetic-freeze release checklist

- [x] Research-status table generated from frozen validation JSONs.
- [x] Confirmed, partial, failed, and unresolved synthetic claims retained.
- [x] Cross-family human measurement contract frozen.
- [x] Confirmatory human axis restricted to Pressure.
- [x] Control and Chaos explicitly exploratory/unresolved.
- [x] Human data source/exclusion boundary documented.
- [x] ORIA/IRB gate documented before confirmatory human analysis.
- [x] Human analysis preregistration written before evaluation.
- [x] Seed inventory and validation hashes captured in freeze manifest.
- [x] Synthetic/human feedback firewall documented.
- [x] Full automated test suite required to pass.
- [ ] Create Git tag `v0.8.0` in the canonical Git repository.
- [ ] Push tag/release to GitHub.
- [ ] Archive the exact GitHub release in Zenodo and record the DOI.

The final three distribution steps require an authenticated canonical GitHub/Zenodo release workflow; this repository snapshot contains everything needed before those external publication actions.

## Freeze enforcement hardening

- [x] Add read-only `verify-freeze` command for the v0.8.0 manifest.
- [x] Fail verification when a frozen artifact or protocol is missing or hash-mismatched.
- [x] Re-check the pressure-only confirmatory axis and closed human-data gate.
- [x] Add tamper-detection unit coverage.
- [x] Run freeze verification in GitHub Actions on pushes and pull requests.

This hardening layer does **not** regenerate synthetic evidence, change frozen scientific definitions, or authorize human-data analysis.
## Release preflight

Before pushing the intended v0.8.0 release commit:

- [x] Provide safe developer targets in `Makefile`.
- [x] Route regenerated audit/status outputs to ignored `build/audit/` paths rather than the frozen validation bundle.
- [x] Add `release-check` for version, required-file, freeze, and whitespace checks.
- [x] Add `CHANGELOG.md` and `docs/RELEASE_NOTES_v0.8.0.md`.
- [x] Add a GitHub Actions `release preflight` job.
- [ ] Run `make preflight` from the canonical Git worktree immediately before tagging.
- [ ] Confirm all GitHub Actions checks pass on that exact commit.

The preflight is read-only with respect to frozen scientific evidence.
