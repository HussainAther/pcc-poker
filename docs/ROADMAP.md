# Roadmap

## Phase 1 — Leduc ground truth

- Validate rules and exact equity.
- Sweep mixture weights, temperatures, and opponent-memory lengths.
- Replicate continuous mixture recovery across seeds, temperatures, and sample
  sizes.
- Compare action-frequency and sequence baselines.
- Estimate pairwise payoff matrices without cyclic bonuses.
- [Completed candidate pass] Replace direct cross-family weight regression with
  shared behavioral measurements. Pressure and effective surprisal transferred;
  exact-value Control efficiency did not and remains a recorded negative result.
- [Completed null] Test whether prior opponent history improves held-out action
  prediction specifically for Control. History helped, but the score tracked
  Pressure more strongly, so the frozen specificity criterion failed.
- [Completed null] Intervene prospectively on opponent-model identity. A model
  aligned to target mode helped Control against Pressure but hurt against
  Control and Chaos; all frozen confirmation checks failed. Test pair-specific
  interaction models prospectively rather than treating an opponent model as a
  portable opponent trait.
- [Completed confirmation] Destroy round timing or context alignment while
  preserving the calibrated model's observation and action-count margins. On
  fresh seeds, aligned contextual knowledge improves Control against Pressure
  across all nine robustness cells and more strongly than against Chaos.
- [Next prospective test] Decompose the engineered Pressure component while
  leaving frozen Control unchanged. Remove learned fold leverage or
  equity-strength selectivity one at a time and test whether either ablation
  removes at least half of the confirmed contextual-alignment advantage.

## Phase 2 — stronger game-theoretic baselines

- Add OpenSpiel compatibility.
- Add CFR/approximate-equilibrium policies.
- Compute counterfactual regret and exploitability.
- Compare PCC quantities with established poker constructs.
- Test whether unlabeled behavioral measurements recover common structure
  across CFR, best-response, entropy-regularized, and existing policy families.

## Phase 3 — human poker histories

- Select a legally reusable, anonymous heads-up corpus.
- Freeze privacy, sampling, and exclusion rules.
- Infer modes per decision, never per named player.
- Evaluate held-out hands, sessions, stakes, and formats.

## Phase 4 — transfer

- Port the measurement contract to Melee telemetry.
- Test whether the same latent structure survives continuous action spaces.

### Contextual Control observable — completed, not family invariant

A frozen matched/yoked public-history likelihood contrast was evaluated on fresh Score and Adaptive seeds. It passed within-family Control positivity and discriminant checks in both families, but failed the preregistered cross-family invariance bound. Keep this as mechanistic evidence and a diagnostic candidate; do not promote it into the conservative family-invariant human-facing panel.

## v0.8.0 — synthetic evidence freeze / pre-human boundary

The synthetic construct-development phase is frozen for the next human-analysis stage. The canonical claim table is `validation/RESEARCH_STATUS.md`. Human confirmatory analysis is restricted to the cross-family-invariant Pressure panel (`pressure_exposure`, `predicted_fold_probability`); Control and Chaos remain exploratory/unresolved. The pre-human analysis protocol is frozen in `HUMAN_ANALYSIS_PREREGISTRATION.md` and cannot be changed in response to confirmatory evaluation outcomes without a documented versioned amendment.

External release actions still pending outside this snapshot: create/push Git tag `v0.8.0`, publish the GitHub release, and archive that exact release in Zenodo.
