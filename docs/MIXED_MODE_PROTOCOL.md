# Continuous-mixture recovery protocol

## Question

Can observable betting behavior recover continuous hidden PCC objective weights
for mixtures that were never present in training?

This is an identifiability test in synthetic agents. It is not a test of human
intentions and cannot establish that PCC exists in human poker.

## Frozen design

- Sample 60 vectors from a symmetric Dirichlet distribution with alpha 0.7.
- Play each mixture for 100 hands in each seat against the same balanced agent.
- Keep both seats and all seeds associated with a mixture in one partition.
- Assign complete mixtures to a deterministic 80/20 train/test split.
- Predict Pressure, Control, and Chaos weights constrained to the simplex.

## Permitted predictors

The action-frequency baseline sees only the marginal rates of check, bet, fold,
call, and raise. The contextual model additionally sees public betting context,
within-player action transitions, pot and call summaries, round, hand length,
and seat. Neither model sees private cards, hidden PCC component scores, action
probabilities, simulation seeds, or target weights as predictors.

## Prespecified comparisons

The contextual model must have lower continuous-weight MAE than:

1. ridge regression using marginal action frequencies alone; and
2. the mean result from 25 shuffled-target fits using the same contextual
   features.

Dominant-mode accuracy, RMSE, and per-mode MAE are reported even when they do
not favor the contextual model.

## Frozen smoke result

Using seed 41, the split contained 51 training mixtures and 9 unseen test
mixtures. Contextual MAE was 0.0906, action-frequency MAE was 0.1038, and mean
shuffled-target MAE was 0.2614. Both prespecified MAE checks passed. Dominant
mode accuracy was 0.833 for the contextual model and 0.889 for action
frequencies, so the contextual advantage is specific to continuous estimation.

The test set contains only 18 seat-level examples. Replication across a grid of
seeds, sample sizes, Dirichlet concentrations, temperatures, and policy
implementations is required before making a robust identifiability claim.

## Seed and temperature replication

A 15-condition grid crossed seeds 41–45 with focal-policy temperatures 0.25,
0.35, and 0.50 while holding the reference temperature at 0.35. Mean contextual
MAE was 0.0962, compared with 0.1146 for action frequencies and 0.2410 for
shuffled targets. Contextual MAE beat action frequencies in 14 of 15 conditions
and shuffled targets in all 15. It failed the action-frequency comparison at
temperature 0.50 with seed 42, so the richer model's advantage is robust in this
grid but not universal.

This still evaluates one hand-authored family of policies. Generalization to
different policy mechanisms remains untested.
