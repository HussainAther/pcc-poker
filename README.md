# PCC Poker

An auditable empirical bridge between **Pressure–Control–Chaos (PCC)** theory
and imperfect-information games.

This repository begins with heads-up Leduc poker because every decision has an
explicit information state, legal action set, counterfactual alternatives, and
terminal payoff. It asks whether hidden action-selection objectives can be
recovered from observable play and whether their interactions are conditionally
non-transitive.

## Scientific contract

PCC modes are not poker actions or permanent player types.

- **Pressure** scores immediate coercion: initiative, commitment, pot leverage,
  and the probability of forcing a fold.
- **Control** scores prediction-weighted action value across an opponent range
  and response model, with a risk penalty.
- **Chaos** scores conditional action surprise subject to an expected-value
  performance floor. Ineffective randomness is not automatically strategic.

Every policy may check, call, bet, raise, or fold. A raise is not intrinsically
Pressure, and a check is not intrinsically Control.

## What is implemented

- deterministic, tested two-player Leduc engine;
- exact showdown equity from each player's information state;
- empirical opponent response tracking;
- continuous PCC mixture weights at every decision;
- separate component scores and combined action probabilities;
- JSONL trajectory logs with hidden weights available as simulation truth;
- AI-versus-AI sweeps and payoff matrices;
- observable-history mode recovery with grouped train/test evaluation;
- continuous-mixture recovery with held-out mixture and seed groups;
- action-frequency and shuffled-target baselines;
- bidirectional transfer tests across independently coded policy mechanisms;
- label-free, information-set behavioral measurements with hidden-card leakage tests;
- disjoint calibration/evaluation validation across both policy families;
- falsification-oriented reports and reproducible seeds.

No cyclic advantage is hardcoded. The initial simulations may or may not
produce the proposed PCC cycle.

## Play the game

Play six hands against the prediction-oriented Control v3 AI:

```bash
python -m pcc_poker play --opponent control --hands 6
```

You can replace `control` with `pressure` or `chaos`. The terminal game uses
the tested Leduc engine, alternates your seat, enforces legal actions, reveals
showdowns, and keeps a zero-sum session score. Add
`--output outputs/my-session.jsonl` for an anonymous local debugging log. See
`docs/PLAYING.md` for rules, opponent definitions, and the research boundary.

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m pcc_poker dataset --hands-per-seat 500 --output outputs/recovery-data.jsonl
python -m pcc_poker analyze outputs/recovery-data.jsonl --output outputs/recovery.json
python -m unittest discover -s tests -v
```

Run a balanced pairwise sweep:

```bash
python -m pcc_poker sweep --hands-per-matchup 2000 --output outputs/sweep.json
```

Every ordered payoff comparison averages the focal policy's result across both
seats; the cycle check therefore does not mistake first-seat advantage for a
mode advantage.

`dataset` places each pure PCC policy in both seats against the same balanced
reference. Recovery uses only public betting state and actions, filters out the
reference agent, and holds out whole hands rather than individual decisions.

Run the stronger continuous-mixture experiment:

```bash
python -m pcc_poker mixed-dataset --mixtures 60 --hands-per-seat 100 \
  --alpha 0.7 --seed 41 --output outputs/mixed-recovery-data.jsonl
python -m pcc_poker mixed-analyze outputs/mixed-recovery-data.jsonl \
  --output outputs/mixed-recovery.json
python -m pcc_poker mixed-grid --seeds 41 42 43 44 45 \
  --temperatures 0.25 0.35 0.5 --output outputs/mixed-grid.json
```

This split holds out complete mixture vectors. Both seats and all simulation
seeds associated with a mixture remain in the same partition. Hidden weights,
private cards, component scores, and action probabilities are excluded from the
features. See `docs/MIXED_MODE_PROTOCOL.md` for the frozen design and limits.
The grid command repeats the full grouped evaluation across independent seeds
and policy temperatures rather than treating a single split as decisive.

Run the policy-family anti-circularity test:

```bash
python -m pcc_poker family-transfer-grid \
  --score-seeds 61 62 63 64 65 \
  --independent-seeds 71 72 73 74 75 \
  --output outputs/family-transfer-grid.json
```

The independent family never calls the original PCC component-score function.
Instead, it constructs separate coercive, one-step value, and novelty-filtered
action distributions and mixes probabilities. This is a stricter domain-shift
test, although both families still receive researcher-assigned mixture weights.

Run the label-free behavioral measurement test:

```bash
python -m pcc_poker behavioral-validation \
  --output outputs/behavioral-measures.json
```

The action model is frozen on separate calibration hands. Candidate Pressure,
Control, and effective-surprisal quantities are then calculated without policy
weights, component scores, actual opponent cards, future cards, or outcomes.
Assigned synthetic weights are consulted only afterward for construct-validity
correlations. The first run is deliberately reported as partial: Pressure and
effective Chaos transfer across both families, while this value-regret definition
of Control does not. See `docs/BEHAVIORAL_MEASURES.md`.

Run the prospective opponent-adaptation Control confirmation:

```bash
python -m pcc_poker control-confirmation \
  --output outputs/control-confirmation.json
```

This uses new seeds and a larger evaluation set. The resulting Control signal
is discriminant but only partly replicates: moderate in the independent family
and weak in the score family. The repository records that result as
inconclusive rather than changing the metric after confirmation.

Validate and balance-check the playable Adaptive PCC family:

```bash
python -m pcc_poker adaptive-validation \
  --output outputs/adaptive-family.json
python -m pcc_poker adaptive-sweep \
  --output outputs/adaptive-sweep.json
python -m pcc_poker balanced-cycle \
  --output outputs/balanced-cycle.json
```

Version 0.3 meets the frozen synthetic balance criterion: replicated,
seat-balanced payoff intervals support Control over Pressure, Chaos over
Control, and Pressure over Chaos, with an edge-strength ratio below `3.0`.
This was achieved through general policy mechanisms, not opponent labels or
cyclic payoff bonuses. The revised label-free Control correlation is below the
repository's descriptive threshold, so observational recovery remains an open
measurement problem. See `docs/BALANCE_PROTOCOL.md`.

Stress-test the frozen v0.3 policies across the preregistered robustness
surface:

```bash
python -m pcc_poker robustness-grid \
  --output outputs/robustness-grid.json \
  --csv-output outputs/robustness-grid.csv
```

The default grid crosses four temperatures, four mode purities, three match
lengths, and ten fresh replicates. It runs every matchup in both seat orders
and refuses to start if the Adaptive policy source differs from the frozen v0.3
checksum. See `docs/ROBUSTNESS_PROTOCOL.md`.

The frozen 5.04-million-hand run retained the complete cycle in 36/48
conditions (`75%`), narrowly missing the preregistered `80%` criterion. The
cycle was present in 15/16 of the 4,000-hand conditions but only 8/16 of the
250-hand conditions. No single mode dominated the surface. This is recorded as
a failed global robustness confirmation with informative horizon boundaries;
the policies were not retuned afterward.

Test whether prior opponent history makes Control observable without generator
labels:

```bash
python -m pcc_poker temporal-control-validation \
  --output outputs/temporal-control.json
```

This cross-fitted test compares a static information-state action predictor
against a temporal predictor that additionally receives prior opponent actions.
Training and evaluation use disjoint mixture groups and seeds. Hidden weights
are consulted only afterward to validate the resulting trajectory score. See
`docs/TEMPORAL_CONTROL_PROTOCOL.md`.

The frozen result shows genuine temporal predictability: log loss improves by
`7.13%`, AUC rises from `.664` to `.741`, and aligned history outperforms the
shuffled-history baseline. It does not confirm Control specificity. The
trajectory score correlates `.166` with Control weight but `.293` with Pressure,
so both prespecified construct checks fail and the null result is retained.

Intervene directly on opponent-model alignment under the unchanged policies:

```bash
python -m pcc_poker counterfactual-control \
  --output outputs/counterfactual-control.json
```

This prospective test calibrates opponent models separately, freezes them, and
compares aligned, swapped-opponent, and no-history conditions using common
random numbers and both seat orders. Its primary outcome is actual held-out
chip payoff rather than an aggression-prediction proxy. See
`docs/COUNTERFACTUAL_CONTROL_PROTOCOL.md`.

The frozen result is negative but informative. Control's aligned-minus-swapped
mean is `.037`, with an interval crossing zero, while aligned models perform
`.211` chips per hand worse than the no-history prior. Alignment helps Control
against Pressure but hurts against Control and Chaos. All four prespecified
checks remain false, showing that target identity alone is insufficient:
opponent models may need to represent the interaction between opponent and
focal policy rather than a portable opponent type.

Test that pair-specific mechanism while matching model margins:

```bash
python -m pcc_poker control-pressure-mechanism \
  --output outputs/control-pressure-mechanism.json
```

This fresh-seed intervention keeps the frozen Control policy unchanged and
preserves calibrated observation counts and global action frequencies. It
destroys only round timing or action-to-context alignment, with Chaos as a
discriminant target. See `docs/CONTROL_PRESSURE_MECHANISM_PROTOCOL.md`.

The frozen v0.7 result confirms all prespecified checks. Against Pressure,
correct contextual information improves Control payoff by `.284` chips per
hand over round-swapping and `.207` over support-preserving context-yoking;
both intervals are
entirely positive. The effects are significantly larger than their Chaos
counterparts, both contrasts are positive in all nine robustness cells, and
all matched-margin checks pass. This is causal evidence for the engineered
Control-over-Pressure mechanism, not evidence about human players.

Decompose which programmed Pressure term sustains that effect prospectively:

```bash
python -m pcc_poker pressure-decomposition \
  --output outputs/pressure-decomposition.json
```

This fresh-seed test leaves the frozen Control implementation unchanged and
removes either learned fold leverage or equity-strength selectivity from the
Pressure component. The prespecified estimand is attenuation of the previously
confirmed aligned-versus-context-yoked Control payoff advantage. See
`docs/PRESSURE_DECOMPOSITION_PROTOCOL.md`. No default result is reported until
the frozen run is executed.

## Fresh-seed effective Chaos validation

Validate the independent value-floor candidate without changing its definition:

```bash
python -m pcc_poker effective-chaos-validation \
  --output validation/effective-chaos-validation.json
```

This freezes the public surprisal model on disjoint calibration hands, evaluates
fresh score-family and independent-family mixture groups, reports the full
Pressure/Control/Chaos correlation matrix, compares against raw surprisal, and
includes a shuffled-weight diagnostic. See
`docs/EFFECTIVE_CHAOS_CONSTRUCT_VALIDATION.md`. No human data are used.

The frozen default run is **not confirmed**. Effective surprisal correlates
positively with assigned Chaos in both families (`.447` independent, `.500`
score), but the independent-family Control correlation (`.487`) remains larger
than its Chaos correlation, so the prespecified discriminant requirement fails.
The score family improves from a raw discriminant margin of `-.006` to `.030`,
while the independent family remains slightly negative (`-.039`). The null is
retained without retuning the value floor or thresholds.

## Repository map

```text
pcc_poker/
  engine.py       Leduc rules, state transitions, equity
  policies.py     PCC objectives and policy mixtures
  families.py     independently implemented policy family
  behavioral.py   label-free measurements and counterfactual oracle
  behavioral_experiment.py  disjoint calibration/evaluation harness
  play.py         interactive human-versus-AI game
  simulate.py     hands, matches, logging, pairwise sweeps
  analyze.py      observable mode recovery and payoff tests
  handhq.py       privacy-minimized PHHS/HandHQ ingestion
  handhq_features.py  pre-decision public game-state reconstruction
  effective_chaos_validation.py  fresh-seed value-floor construct test
  cli.py          command-line interface
docs/
  MEASUREMENT_CONTRACT.md
  FALSIFICATION_PLAN.md
  ETHICS_AND_DATA.md
  MIXED_MODE_PROTOCOL.md
  POLICY_FAMILY_TRANSFER.md
  BEHAVIORAL_MEASURES.md
  BALANCE_PROTOCOL.md
  ROBUSTNESS_PROTOCOL.md
  TEMPORAL_CONTROL_PROTOCOL.md
  COUNTERFACTUAL_CONTROL_PROTOCOL.md
  CONTROL_PRESSURE_MECHANISM_PROTOCOL.md
  PRESSURE_DECOMPOSITION_PROTOCOL.md
  HUMAN_DATA_INGESTION_PROTOCOL.md
  EFFECTIVE_CHAOS_VALUE_PROTOCOL.md
  EFFECTIVE_CHAOS_CONSTRUCT_VALIDATION.md
  PLAYING.md
  ROADMAP.md
tests/
```

## Interpretation boundary

Successful recovery in simulated Leduc would show that known PCC-like
objectives leave distinguishable behavioral signatures in an accepted
imperfect-information game. It would not establish that human poker players—or
Melee players—possess those intentions. Human hand histories and fighting-game
telemetry require separate validation.

## Status

Research prototype. The immediate goal is to test identifiability and boundary
conditions before scaling to Hold'em or human data.

### Mock-only human PCC observable layer

The repository now includes a prospective public-state measurement layer for
future HandHQ work. `pcc_poker/human_observables.py` defines a transparent
commitment/escalation Pressure candidate, a frozen history-alignment signal, and
behavioral surprisal from a separately calibrated public-state action model.
The latter two are deliberately not treated as validated Control or effective
Chaos constructs. Development tests use invented PHHS fixtures only; see
`docs/HUMAN_PCC_OBSERVABLES_PROTOCOL.md`.

### Independent effective-Chaos value floor

`pcc_poker/value_floor.py` adds a synthetic-only, non-PCC value reference for the
next Chaos construct-validation step. Exact Leduc information-set action values
are computed under a fixed uniform legal-action continuation model, then used
to discount behavioral surprisal when the chosen action has excessive regret.
The prespecified sanity tests require random-but-bad behavior to score low,
strong deterministic behavior to remain low-surprisal, and strong mixed but
value-adequate behavior to score higher. This does not analyze HandHQ data and
does not yet define a Hold'em value model. See
`docs/EFFECTIVE_CHAOS_VALUE_PROTOCOL.md`.

### Chaos–Control entanglement decomposition

A fresh-seed follow-up now decomposes effective surprisal into a component
explained by prior public action history and a value-preserving residual that
remains after history conditioning:

```bash
python -m pcc_poker chaos-control-decomposition \
  --output validation/chaos-control-decomposition.json
```

The frozen result is partial/null.  In the independent family the residual is
Chaos-discriminant (`r=.554` for Chaos versus `.499` for Control), but history
conditioning does not improve the Chaos margin and the history-explained piece
misses the prespecified Control-discrimination threshold.  In the score family,
history-explained surprise is strongly Control-specific.  The earlier
Chaos/Control overlap therefore cannot be explained by one universal
public-history mechanism.  See `docs/CHAOS_CONTROL_DECOMPOSITION_PROTOCOL.md`.
