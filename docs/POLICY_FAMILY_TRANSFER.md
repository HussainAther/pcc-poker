# Policy-family transfer protocol

## Motivation

Recovery within agents built from one PCC score equation may be circular: the
detector could learn signatures peculiar to that equation rather than a stable
strategic structure. This experiment trains on one policy implementation and
tests on another without refitting.

## Families

The **score family** is the original implementation. It assigns each legal
action separate Pressure, Control, and Chaos scores and takes a weighted sum
before softmax action selection.

The **independent probability family** does not call that score function. It
constructs three independently normalized action distributions:

- a card-independent coercive mechanism favoring initiative and fold leverage;
- a one-step expected-chip-value mechanism;
- a context-conditioned novelty mechanism with an admissible value-loss bound.

The assigned mixture combines these action probabilities rather than component
scores. A unit test forces the original score function to fail and verifies that
the independent family continues to act.

Both implementations still use researcher-assigned PCC coordinates. This test
therefore reduces one form of implementation circularity but does not discover
PCC without assumptions.

## Split and features

- Training and transfer mixture vectors are generated from distinct seeds.
- Mixture identifiers never overlap between domains.
- Each mixture is represented from both seats.
- Predictors contain observable public betting histories only.
- Hidden weights, private cards, action probabilities, and component values are
  excluded.

Models are compared with a training-set mean predictor, marginal action
frequencies, and shuffled training targets.

## Frozen result

A five-seed bidirectional grid used 40 mixtures per family and 75 hands per
seat. For score-to-independent transfer, contextual MAE averaged 0.2186,
compared with 0.1961 for action frequencies, 0.2443 for the constant predictor,
and 0.2785 for shuffled targets. The contextual model beat shuffled targets in
5/5 runs and the constant predictor in 4/5, but it never beat action frequencies.

For independent-to-score transfer, contextual MAE averaged 0.2574, compared
with 0.2381 for action frequencies, 0.2158 for the constant predictor, and
0.2745 for shuffled targets. It beat action frequencies and the constant
predictor in only 1/5 runs each.

## Interpretation

The current supervised PCC coordinates are not implementation-invariant.
Pressure transfers more reliably than Control and Chaos in the initial
single-seed diagnostic, but the richer contextual representation does not
provide robust bidirectional transfer. This is an informative failure.

The next model should not directly regress researcher-assigned weights across
families. It should estimate preregistered behavioral quantities—coercive
response compression, prediction-conditioned regret, and effective conditional
surprise—and test whether a shared latent geometry emerges without using the
generator coordinates as labels.
