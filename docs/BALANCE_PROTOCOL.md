# Engineered PCC Cycle: Balance Protocol

## Scope

This protocol tests whether the three playable synthetic policies form the
proposed payoff cycle in heads-up limit Leduc:

```text
Control > Pressure, Chaos > Control, Pressure > Chaos
```

It is an engineering test of policies chosen by the researcher. It is not
evidence that human poker behavior has the same structure.

## Mechanisms

- Pressure uses selective force. It favors betting and raising, but conditions
  commitment on information-set equity and can release weak states.
- Control makes a sharper response to a learned, round-specific opponent fold
  rate while retaining card-value constraints.
- Chaos favors underused viable branches. Its performance tolerance expands
  with the pot, and actions outside that band remain possible at low weight.

The decision function receives cards available to the acting player, public
state, and prior observed actions. It does not receive the opponent's PCC label,
the current matchup identity, future cards, or a cyclic payoff adjustment.

## Development and confirmation split

Mechanism parameters were explored on development seeds. The final candidate
was frozen before running confirmation seeds beginning at `23001`, spaced by
`20`. The confirmation result was retained without post-confirmation retuning.

Each of 12 confirmation replicates contains all three matchups in both seat
orders, with 1,000 hands per order. A replicate therefore contains 6,000 hands;
the confirmation contains 72,000 hands total.

## Acceptance rule

The engineered cycle is called balanced only when:

1. the lower bound of the approximate 95% normal interval across replicate-level
   seat-balanced payoff edges is above zero for all three directions; and
2. the largest mean edge divided by the smallest is no greater than `3.0`.

The replicate, rather than the individual hand, is the uncertainty unit.

## Frozen result

| Edge | Mean payoff | Approximate 95% interval | Positive replicates |
| --- | ---: | ---: | ---: |
| Control over Pressure | 0.2058 | [0.1181, 0.2934] | 10/12 |
| Chaos over Control | 0.1662 | [0.0974, 0.2350] | 12/12 |
| Pressure over Chaos | 0.1325 | [0.0191, 0.2458] | 9/12 |

The edge-strength ratio is `1.553`. Both acceptance conditions pass.

## Measurement boundary

Balance does not establish construct validity. Under the separately frozen
behavioral measures, the revised family has correlations of Pressure `.748`,
Control `.157`, and Chaos `.224` with assigned synthetic weights. Control falls
below the descriptive `.20` threshold and remains inconclusive as an
observationally recovered quantity, even though its programmed adaptive
mechanism and response tests are present.

The normal intervals are approximate and 12 replicates are modest. Results may
also depend on Leduc rules, mixture purity, policy temperature, and learning
horizon. Replication under alternate settings is the next robustness test.
