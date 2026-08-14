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

- **Pressure** applies initiative and wager commitment broadly.
- **Control v2** maintains a round-specific opponent response model. It times
  aggression to learned fold vulnerability while retaining a card-value safety
  constraint.
- **Chaos** favors actions that have been unusual in comparable contexts, but
  filters choices through a value tolerance.

The modes are policy objectives, not fixed actions. Control may bet, check,
call, raise, or fold; its defining mechanic is conditional adaptation.

## Current balance

The first frozen Adaptive-family validation recovered the intended synthetic
weights from behavior: Pressure `r=.796`, Control `r=.311`, and Chaos `r=.446`.
This validates implementation because the Control generator and detector both
encode opponent-response adaptation; it is not independent evidence of PCC.

The first seat-balanced payoff sweep did not produce the complete proposed
cycle. Control beat Pressure, Chaos beat Control, and Chaos also beat Pressure.
The alpha is therefore playable and behaviorally differentiated, but not yet
competitively balanced. Balance changes should use new seeds and preserve the
unmodified diagnostic as a baseline.
