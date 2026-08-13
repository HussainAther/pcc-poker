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
- falsification-oriented reports and reproducible seeds.

No cyclic advantage is hardcoded. The initial simulations may or may not
produce the proposed PCC cycle.

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

## Repository map

```text
pcc_poker/
  engine.py       Leduc rules, state transitions, equity
  policies.py     PCC objectives and policy mixtures
  simulate.py     hands, matches, logging, pairwise sweeps
  analyze.py      observable mode recovery and payoff tests
  cli.py          command-line interface
docs/
  MEASUREMENT_CONTRACT.md
  FALSIFICATION_PLAN.md
  ETHICS_AND_DATA.md
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
