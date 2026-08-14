# Independent value-floor protocol for an effective Chaos candidate

## Question

Can behavioral surprisal be discounted by an independently defined decision-quality floor so that value-destroying randomness is not mislabeled as strategic Chaos?

This is a **synthetic construct-validation step only**. It does not authorize or perform analysis of the HandHQ/Zenodo human dataset.

## Independence contract

The value floor must not use PCC mixture weights, PCC component scores, policy labels, learned opponent identity, the chosen policy's action probabilities, future observed actions, or terminal outcomes from the evaluated hand.

For heads-up Leduc, `UniformContinuationValueModel` computes exact information-set action values under a fixed reference in which every future actor chooses uniformly among legal actions. The acting player's information state is respected by enumerating compatible hidden states; the simulator's actual opponent card is not used as privileged information.

This reference is intentionally simple. It is not claimed to be optimal poker and is not a model of human play. Its role is to make the performance criterion independent of the PCC generators.

## Frozen candidate

For action `a`, let

- `Q(a)` be its reference value;
- `Q* = max_a Q(a)`;
- `R(a) = max(0, Q* - Q(a))` be regret;
- `tau > 0` be a declared regret tolerance;
- `A(a) = max(0, 1 - R(a)/tau)` be performance adequacy; and
- `S(a)` be normalized behavioral surprisal from a separately frozen action model.

The candidate effective-surprisal score is

`E(a) = S(a) * A(a)`.

The default synthetic tolerance is `pot + bet_size`. Any later human-data tolerance must be specified prospectively and cannot be inherited automatically from Leduc.

## Prespecified sanity tests

Before testing PCC policy families, the metric must satisfy three qualitative cases:

1. **random-but-bad:** high surprisal plus regret at or above the tolerance gives effective surprisal zero;
2. **strong deterministic:** adequate value plus very low surprisal remains low effective surprisal;
3. **strong mixed:** a surprising action that remains within the value tolerance scores higher than the strong deterministic case.

The repository unit tests freeze these logical requirements without fitting thresholds to PCC labels.

## What success would mean

Passing the sanity tests establishes only that the composite has the intended mathematical behavior. The next construct-validity experiment must evaluate it on fresh synthetic PCC trajectories and report the full relationship to Pressure, Control, and Chaos weights rather than only the favorable Chaos correlation.

## Human-data boundary

No human PHH record is required by this module or its tests. A Hold'em value floor for future HandHQ work is a separate scientific problem and must be defined before human confirmatory scoring. Leduc Q-values must not be applied to Hold'em decisions.
