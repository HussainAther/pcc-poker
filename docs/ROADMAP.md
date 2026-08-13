# Roadmap

## Phase 1 — Leduc ground truth

- Validate rules and exact equity.
- Sweep mixture weights, temperatures, and opponent-memory lengths.
- Test mode recovery from observable histories.
- Compare action-frequency and sequence baselines.
- Estimate pairwise payoff matrices without cyclic bonuses.

## Phase 2 — stronger game-theoretic baselines

- Add OpenSpiel compatibility.
- Add CFR/approximate-equilibrium policies.
- Compute counterfactual regret and exploitability.
- Compare PCC quantities with established poker constructs.

## Phase 3 — human poker histories

- Select a legally reusable, anonymous heads-up corpus.
- Freeze privacy, sampling, and exclusion rules.
- Infer modes per decision, never per named player.
- Evaluate held-out hands, sessions, stakes, and formats.

## Phase 4 — transfer

- Port the measurement contract to Melee telemetry.
- Test whether the same latent structure survives continuous action spaces.
