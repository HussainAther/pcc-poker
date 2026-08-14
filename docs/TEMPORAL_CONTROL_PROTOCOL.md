# Frozen Temporal Control Validation

## Question

Does prior opponent behavior improve prediction of a focal policy's next
aggressive action beyond its current information state, and is that incremental
predictive gain specifically associated with assigned synthetic Control weight?

This operationalizes Control as adaptation, not passivity or raw action value.

## Frozen data split

- policy family: unchanged Adaptive v0.3 mechanisms;
- training: 80 mixtures, seed `61001`;
- evaluation: 80 disjoint mixtures, seed `62001`;
- hands: 100 per mixture and focal seat;
- both seats remain grouped under the same mixture identity;
- no mixture identifier occurs in both partitions.

## Prediction target and models

The binary target is whether the action is aggressive (`bet` or `raise`).

The static logistic model receives the acting policy's own card, public card,
legal actions, current-hand history, pot, amount to call, round, and seat.

The temporal model receives the same variables plus only information available
before the current decision: prior opponent action rates by public context,
context entropy and sample size, the last eight opponent actions, and smoothed
fold, resistance, and open-aggression estimates.

Both models use the same fixed L2-regularized logistic fitting procedure. Hidden
PCC weights, generator component scores, generator action probabilities,
terminal outcomes, future cards, and actual opponent private cards are excluded.

## Primary score and falsification baseline

For every held-out trajectory, temporal Control is:

```text
mean[log p_temporal(observed action) - log p_static(observed action)]
```

Assigned synthetic weights are used only after this score is calculated. The
falsification baseline repeats evaluation 25 times after shuffling temporal
histories among decisions with the same round, wager-facing status, and legal
action set.

## Frozen acceptance rule

Temporal Control is confirmed only if all four conditions pass:

1. temporal-model held-out log loss is below static-model log loss;
2. temporal-model log loss is below the context-preserving shuffled-history
   mean;
3. trajectory score correlation with Control weight is at least `.20`; and
4. its correlation with Control exceeds its correlations with Pressure and
   Chaos weights.

The result will be retained without changing the policies, predictors,
threshold, split, or outcome after evaluation.

## Interpretation boundary

A successful result would establish synthetic identifiability of a programmed
adaptive mechanism. It would not prove that human players possess PCC states,
nor authorize analysis of human hand histories before the appropriate
institutional determination.

## Frozen result

The temporal-dependence checks passed, but the Control-specific confirmation
did not.

| Quantity | Static | Temporal |
| --- | ---: | ---: |
| Held-out log loss | 0.6115 | 0.5679 |
| Held-out AUC | 0.6638 | 0.7405 |

The temporal model reduced log loss by `7.13%`. Its log loss was also well below
the context-preserving shuffled-history mean of `0.7155`, showing that aligned
opponent history contains genuine predictive information.

The trajectory log-likelihood gain correlated with assigned weights as follows:

- Pressure: `.293`;
- Control: `.166`, approximate Fisher 95% interval `[.011, .313]`;
- Chaos: `-.475`.

Control therefore missed the frozen `.20` threshold and was not discriminant
from Pressure. `temporal_control_confirmed` is retained as `false`.

The correct interpretation is narrower but informative: opponent history helps
predict these synthetic policies, yet an aggression-prediction advantage is not
specific to Control. Pressure also uses learned fold leverage, so temporal
responsiveness and Control cannot be treated as synonyms. No policies,
predictors, thresholds, or outcomes were changed after this result.
