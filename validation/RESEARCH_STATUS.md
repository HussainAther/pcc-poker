# PCC Poker research status

> Frozen synthetic evidence only. This table does not make claims about human poker behavior.

**Summary:** 5 confirmed · 6 partial · 2 failed · 2 unresolved

| Claim | Status | Evidence | Scope | Source |
|---|---|---|---|---|
| Frozen engineered PCC policies exhibit the prespecified balanced Pressure→Chaos→Control→Pressure cycle. | CONFIRMED | balanced_cycle_confirmed=True; edge_strength_ratio=1.553 | synthetic frozen policy family | `validation/balanced-cycle.json` |
| The frozen PCC cycle is robust across the prespecified temperature, purity, and match-length grid. | PARTIAL | cycle_fraction=0.750; no_mode_dominates_grid=True | synthetic robustness surface | `validation/robustness-grid.json` |
| Contextually aligned Control specifically improves performance against Pressure under matched counterfactual controls. | CONFIRMED | passed 6/6 prespecified checks | synthetic causal-mechanism intervention | `validation/control-pressure-mechanism.json` |
| Generic model-alignment counterfactuals identify a Control-specific dependence across targets. | FAILED | failed: control_aligned_beats_swapped, control_aligned_beats_prior, control_model_dependence_is_discriminant, control_largest_for_at_least_two_targets | synthetic counterfactual intervention | `validation/counterfactual-control.json` |
| Public temporal-history prediction gain is a discriminant observational measure of Control. | PARTIAL | passed 2/4 prespecified checks; failed: control_correlation_at_least_0_20, control_correlation_discriminant | synthetic observational measurement | `validation/temporal-control.json` |
| Matched public-history likelihood contrast is a family-invariant observational measure of Control. | PARTIAL | cross_family_control_gap=0.602; passed 2/3 prespecified checks; failed: cross_family_control_gap_at_most_0_20 | two synthetic policy families | `validation/contextual-control-observable.json` |
| Value-floor-weighted effective surprisal is a cross-family discriminant observational measure of Chaos. | PARTIAL | passed 3/4 prespecified checks; failed: all_families_discriminant | two synthetic policy families | `validation/effective-chaos-validation.json` |
| Conditioning on public history universally separates Control-linked surprise from Chaos-linked residual surprise. | PARTIAL | passed 2/4 prespecified checks; failed: independent_history_explained_is_control_discriminant_by_0_03, independent_residual_improves_chaos_margin_by_0_03 | two synthetic policy families | `validation/chaos-control-decomposition.json` |
| Public Pressure exposure universally explains the negative Pressure/effective-surprisal association and improves Chaos discrimination. | PARTIAL | passed 5/6 prespecified checks; failed: independent_chaos_margin_improves_by_0_03 | two synthetic policy families | `validation/pressure-surprise-decomposition.json` |
| At least one conservative label-free Pressure observable is stable across both policy families. | CONFIRMED | selected: pressure_exposure, predicted_fold_probability | two synthetic policy families | `validation/family-invariant-panel.json` |
| A conservative label-free Control observable is currently supported across both policy families. | UNRESOLVED | selected: none | two synthetic policy families | `validation/family-invariant-panel.json` |
| A conservative label-free Chaos observable is currently supported across both policy families. | UNRESOLVED | selected: none | two synthetic policy families | `validation/family-invariant-panel.json` |
| Supervised PCC coordinate recovery transfers invariantly between the Score and Independent policy families. | FAILED | Current supervised PCC coordinates are not implementation-invariant. | cross-family supervised transfer | `validation/family-transfer-grid-summary.json` |
| Continuous synthetic PCC mixture weights are identifiable above the prespecified baselines within the development family. | CONFIRMED | relative_mae_improvement_over_action_frequency=0.127; passed 2/2 prespecified checks | synthetic mixture recovery; not human construct validity | `validation/mixed-recovery.json` |
| The frozen synthetic validation bundle is complete and reproducibility-audit ready. | CONFIRMED | reproducibility_ready=True; present=6/6 | engineering/reproducibility | `validation/reproducibility-manifest.json` |
