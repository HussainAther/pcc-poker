# Playing PCC Poker

PCC Poker is a playable heads-up limit Leduc game built on the same tested
engine used by the synthetic experiments. It is a research alpha, not a
real-money poker product.

## Start a game

```bash
python -m pcc_poker play --opponent control --hands 6
```

Choose `pressure`, `control`, or `chaos`. Enter the displayed action name or its
number. Your seat alternates each hand. Cards are revealed after each hand and
the session score is zero-sum.

To retain an anonymous local trajectory for personal debugging:

```bash
python -m pcc_poker play --opponent control --hands 12 \
  --output outputs/my-session.jsonl
```

No network service or player identifier is used. Personal play is not included
in the human-data study and should not be analyzed as research evidence without
the appropriate institutional determination and a prospectively frozen plan.

## Opponents

- **Pressure** applies selective force: strong information states commit and
  escalate, while weak states can release instead of bluffing indiscriminately.
- **Control v3** maintains a round-specific opponent response model. It times
  aggression to learned fold vulnerability while retaining a card-value safety
  constraint.
- **Chaos** favors actions that have been unusual in comparable contexts, but
  retains several viable branches rather than always taking the estimated best
  action.

The modes are policy objectives, not fixed actions. Control may bet, check,
call, raise, or fold; its defining mechanic is conditional adaptation.

## Current balance

Version 0.3 passes the frozen engineering criterion for the complete cycle. In
12 fresh replicates of every matchup in both seat orders, mean payoff edges were
Control over Pressure `.206`, Chaos over Control `.166`, and Pressure over Chaos
`.132`. All replicate-level 95% normal intervals were above zero, and the
largest edge was `1.55` times the smallest (limit `3.0`). No policy receives its
opponent's mode label and no cyclic payoff bonus exists.

This is synthetic game balance, not evidence about humans. The frozen
label-free detector also places the revised Control signature at `r=.157`, below
the repository's descriptive `.20` threshold. Control is therefore an explicit,
tested game mechanic whose observational recovery remains inconclusive. See
`docs/BALANCE_PROTOCOL.md` for the full boundary.
