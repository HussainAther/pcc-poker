# Prospective Score-Control value-aware intervention

## Status

Post-v0.8 synthetic extension only. This protocol does not modify the frozen
human-facing measurement contract and does not authorize human-data analysis.

## Motivation

The preceding contextual-gain intervention recovered Score-family information
uptake and context alignment but not value-sensitive intervention. The frozen
value decomposition then showed that Control-heavy Score trajectories carried
more positive context signal while also having lower counterfactual efficiency
and higher regret.

The next intervention therefore targets **value selection, not additional
context gain**.

## Single prospective change

Keep the existing contextual Control term unchanged:

```text
3.35 * (opponent_fold_probability - 1/3)
```

For `bet`/`raise`, apply that term only when a separately frozen synthetic
counterfactual oracle estimates the aggressive action's efficiency to be at
least **0.80**.

The 0.80 threshold is inherited unchanged from the prior frozen value
bottleneck decomposition. The oracle uses only the acting player's information
state plus a synthetic public continuation model; it does not use the hidden
opponent card, PCC labels, or human data.

## Frozen evaluation rule

Re-run the existing three-stage Control recovery gate unchanged:

```text
information uptake -> context alignment -> value-sensitive intervention
```

The same calibration/evaluation seeds, matched context-yoking, minimum Control
correlation (`0.20`), and minimum discriminant margin (`0.05`) are retained.
Control is resolved only if all three stages recover in both Score and Adaptive
families.

No second intervention or threshold change is permitted after observing the
result.
