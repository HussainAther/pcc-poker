# Fresh-seed effective-Chaos construct validation

## Frozen question

Does the independent value-floor-weighted surprisal candidate track assigned synthetic Chaos more specifically than Pressure or Control across both independently implemented PCC policy families?

The definition in `value_floor.py` is frozen before these evaluation seeds are examined. This experiment uses no human data.

## Separation of calibration and evaluation

A smoothed public-action model is fitted on disjoint synthetic calibration hands from both policy families. Evaluation then uses fresh mixture groups, seeds, and hands. Hidden PCC weights are never inputs to the action model or value floor; they are consulted only after scoring as construct-validity targets.

The default seeds are:

- score calibration: `1401`
- independent calibration: `1409`
- score evaluation: `1601`
- independent evaluation: `1609`

## Metrics

For each focal decision:

1. raw surprisal is `-log p(a|public state)` divided by `log(number of legal actions)`;
2. independent performance adequacy is computed from exact Leduc information-set Q-values under uniform legal-action continuation;
3. effective surprisal is raw normalized surprisal multiplied by adequacy.

Scores are averaged within mixture and focal seat before correlations are calculated.

## Full discriminant matrix

For each policy family, both raw and effective surprisal are correlated with all three assigned mixture weights: Pressure, Control, and Chaos. The primary discriminant margin is

`r(metric, Chaos) - max(r(metric, Pressure), r(metric, Control))`.

## Prespecified confirmation checks

Confirmation requires all of the following:

1. effective-surprisal/Chaos correlation is at least `0.20` in both families;
2. the Chaos correlation exceeds both off-axis correlations by at least `0.05` in both families;
3. the effective-surprisal discriminant margin is no more than `0.02` worse than raw surprisal in either family; and
4. the value floor improves the discriminant margin by at least `0.02` in at least one family.

These thresholds are frozen before the default evaluation run. A failed check is retained; the value floor or thresholds are not retuned on the same seeds.

## Falsification baselines

Two baselines are reported:

- **constant floor:** raw normalized surprisal, equivalent to setting adequacy to one;
- **shuffled target:** Chaos weights are permuted across complete mixture groups before correlation with effective surprisal.

The shuffled result is diagnostic rather than an acceptance criterion because any one finite permutation can fluctuate. It is included to expose whether the observed association resembles an arbitrary group assignment.

## Interpretation boundary

A positive result validates this exact effective-surprisal operationalization only inside the synthetic Leduc setting and these engineered policy families. It does not establish human Chaos, intention, personality, or a Hold'em value model.

## Frozen default result

The default fresh-seed run does **not** confirm the construct across both
families.

- independent family: effective surprisal correlates `.447` with Chaos, `.487`
  with Control, and `-.926` with Pressure; discriminant margin `-.039`;
- score family: effective surprisal correlates `.500` with Chaos, `.470` with
  Control, and `-.955` with Pressure; discriminant margin `.030`;
- the value floor is non-inferior to raw surprisal in both families and improves
  the score-family margin by `.036`, but the required `0.05` discriminant margin
  is not reached in either family;
- shuffled-Chaos correlations are `.027` (independent) and `.124` (score).

Thus the candidate carries a reproducible Chaos-related signal but does not
separate Chaos from Control strongly enough under the frozen criterion. This is
retained as a boundary result. The next revision, if any, must be specified and
evaluated on new seeds rather than optimized on these results.
