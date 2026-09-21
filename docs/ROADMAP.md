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
- [Completed confirmation] Decompose the engineered Pressure component while
  leaving frozen Control unchanged. Strength-selectivity removal eliminates the
  positive contextual-alignment advantage; fold-leverage removal does not meet
  the prespecified 50% attenuation threshold.

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


## Post-v0.8 mixed recovery frontier

- [Completed] Continuous mixed-weight OOD recovery: contextual-history features beat action frequencies overall and in four of five prespecified simplex regions; balanced mixtures remain the negative case.
- [Completed] Nested observable-feature ablation: betting context and sequential history account for most of the OOD advantage; public state/value-intensity summaries add a smaller gain; balanced remains action-frequency-favored.
- [Completed partial/null] Destroy within-hand sequence order while preserving M0-M2 and per-hand action counts. Overall M3 gain attenuated 38.8% and Control-heavy 28.2%, failing the frozen 50% criteria; Chaos-heavy attenuated 90.4%. The strong temporal-order interpretation is not supported.
- [Completed exploratory] Decompose the transition block into persistence, same-action pairs, mixed unordered adjacent pairs, full unordered adjacent-pair composition, and fully directed transitions. Mixed unordered pairs recover most of the Control-heavy M2 -> M3 advantage; bet-call and bet-check are the strongest contributors, with clear context dependence.
- [Completed prospective test] `CONTEXT_MATCHED_TRANSITION_FALSIFICATION_PLAN.md` replicated the facing-bet bet-call signal on fresh seeds but did not eliminate the seat-effect gap.
- [Completed prospective follow-up] `SEAT_YOKED_TRANSITION_FALSIFICATION_PLAN.md` evaluated identical latent PCC mixtures in both focal seats on fresh seeds 1201-1205. The facing-bet bet-call ablation effect stayed positive in both seats on 5/5 seeds, while seat 1 remained consistently stronger. Different latent mixture draws therefore do not explain the seat asymmetry.
- [Completed prospective decomposition] `SEAT_OPPORTUNITY_DECOMPOSITION_PLAN.md` / `RESULT.md` found a large structural opportunity imbalance: seat 0 receives about twice as many facing-bet mixed-transition opportunities as seat 1, concentrated in round 0. The frozen 14/41 fixed-budget standardization was infeasible for most seat-1 examples, so it did not cleanly resolve whether equal opportunity removes the ablation gap.
- [Next mechanistic frontier] Run a newly preregistered, genuinely feasible matched-opportunity follow-up on untouched seeds; do not retune on seeds 1301-1305.
