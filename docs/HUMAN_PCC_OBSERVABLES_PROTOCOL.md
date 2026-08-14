# Human PCC Observable Protocol (mock-only development stage)

## Purpose

This protocol defines public-state candidate measurements for future human
HandHQ analyses without treating those measurements as latent PCC ground truth.
No human confirmatory data are analyzed during this development stage.

## Input boundary

The observable layer consumes only `PublicDecisionState` objects produced by the
sanitized ingestion and public-state reconstruction pipeline. It does not accept
raw PHH source records.

Excluded inputs include:

- source player handles or obfuscated source identifiers;
- table names, source hand identifiers, exact timestamps/dates, and timezone;
- private hole-card contents;
- outcomes/winnings;
- future actions; and
- hidden PCC policy weights or component scores.

## Candidate Pressure observable

For a chosen bet/raise, define the incremental commitment as the increase from
the actor's current street contribution to the PHH `cbr` target. Normalize that
increment by the larger of actor remaining stack and effective stack, capped at
1. Define an escalation indicator equal to 1 when the chosen target exceeds the
current live bet.

The descriptive Pressure index is

`0.5 * commitment_fraction + 0.5 * escalation_indicator`.

Non-bet/raise actions receive zero on this index. The equal weighting is fixed
for transparency and is not tuned against PCC labels or outcomes.

## Static public-state action model

A smoothed frequency model is fit only on a separate calibration set. Its
context includes street, active-player count, whether the actor faces a wager,
coarse to-call and pot/effective-stack buckets, current-street raise count, and
legal actions.

Evaluation decisions never update the frozen calibration counts. Unseen
contexts use deterministic global-count backoff.

## History-alignment candidate

A second frozen model augments the static context with a coarse summary of prior
public opponent actions: fold/check-call/bet-raise counts, most recent opponent
action, and a coarse amount bucket.

For the observed evaluation action:

`history_alignment = log P_temporal(action) - log P_static(action)`.

Positive values mean that prior opponent history makes the observed action more
predictable under the frozen calibration model. This is a Control-adjacent
measurement, not a validated Control construct. Prior PCC Poker temporal tests
showed that history predictability was not Control-specific, so specificity must
be re-established rather than assumed.

## Behavioral surprisal candidate

For the observed evaluation action:

`behavioral_surprisal = -log P_static(action)`.

A normalized descriptive version divides by `log(number of legal actions)` when
there is more than one legal action.

This quantity is not called effective surprisal or Chaos. The PCC definition of
Chaos requires surprise subject to an expected-value/performance floor. A
separately validated value model is required before that stronger construct can
be claimed.

## Calibration/evaluation separation

Any future human analysis must split complete hands, and where longitudinal
player grouping is retained under the approved data protocol, must avoid placing
linked observations across calibration and evaluation groups. Thresholds and
model definitions should be frozen before confirmatory evaluation.

## Current validation

Repository tests use invented PHHS fixtures only. They verify:

- raises create positive commitment/escalation Pressure;
- non-aggressive actions do not;
- behavioral surprisal equals negative log frozen static probability;
- the history model can capture a synthetic history-conditioned pattern;
- evaluation does not mutate calibration counts;
- private-card contents do not affect measurements; and
- the split API returns measurements only for evaluation decisions.

## Interpretation boundary

These measurements establish a reproducible feature contract. They do not show
that human poker players possess Pressure, Control, or Chaos intentions. Human
construct validity requires prospective tests, discriminant comparisons, and
retention of negative results.
