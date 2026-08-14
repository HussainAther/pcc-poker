# Candidate behavioral PCC measurements

## Purpose

This layer asks a stricter question than supervised weight recovery: can PCC-like
quantities be computed from a poker decision without reading the policy's hidden
PCC weights or component scores? These are candidate operationalizations, not
ground-truth labels and not claims about a player's enduring style or intention.

## Inputs and leakage boundary

The public action model is fitted on a disjoint calibration simulation. It sees
only seat, betting round, whether a wager is faced, pot size, legal actions, and
the observed action. For counterfactual value, the oracle additionally sees the
acting player's own private rank, as a real player would. It discards the actual
opponent card and unrevealed deck order, enumerating all compatible information
states instead.

Target PCC weights, policy labels, action probabilities, component scores,
showdown equity, future events, and terminal payoff are not measurement inputs.
Assigned weights are used only afterward to test construct validity.

## Definitions

For a legal action `a`, the oracle computes expected chip value `Q(a)` by exact
Leduc recursion under the fixed public continuation model.

- **Control efficiency** is `exp(-(max Q - Q(a)) / scale)`. It equals one for a
  counterfactually best action and declines with regret. It tests one narrow
  meaning of Control: value-aligned calculation.
- **Pressure index** is nonzero only if the chosen action leaves the opponent
  facing a wager. It averages predicted response compression, predicted fold
  probability, and commitment relative to the resulting pot.
- **Effective surprisal** is the chosen action's negative log probability under
  the public action model, multiplied by Control efficiency. Rare but severely
  value-destroying play is therefore discounted rather than rewarded as Chaos.

Each decision retains the component values as well as the three summary
quantities so later work can replace the composites without rerunning hands.

## Validation design

`behavioral-validation` fits the action model on calibration hands and freezes
it before evaluating new mixtures, seeds, and hands from both policy families.
Results are aggregated by mixture and focal seat. The complete 3-by-3 matrix of
measurement-to-assigned-weight correlations is reported, rather than only the
three favorable diagonal entries.

The included first full run is a partial result: Pressure and effective
surprisal have positive matching-axis correlations in both families. Control
efficiency is nearly flat in the independent family and negatively related to
the score-family Control weight. That failure is retained. It suggests either
that the candidate misses prediction-oriented Control, that the original
Control generator is not an exact-value policy, or both.

## Prospective Control follow-up

The failed value-efficiency result motivated two prospective candidates. First,
private-information gain measured how much the player's own card improved
prediction of the chosen action. It was positive for assigned Control on fresh
seeds, but failed discriminant validity because it tracked assigned Chaos even
more strongly in the independent family. That candidate was rejected.

The second candidate directly operationalized opponent prediction. For each
focal seat and betting round, it estimated the opponent's fold rate using only
earlier visible responses. **Opponent-adaptation Control** is the within-seat
covariance between that running estimate and subsequent aggressive choices.
It therefore asks whether aggression is timed to a learned vulnerability rather
than applied indiscriminately.

After development on seeds 401/409, the definition was frozen and evaluated on
new seeds 601/607 with 60 mixtures and 100 hands per seat. It was discriminant
in both families—the Control correlation exceeded the Pressure and Chaos
correlations—but strength did not fully replicate:

- independent family: `r = .219`, approximate 95% CI `[.041, .383]`;
- score family: `r = .094`, approximate 95% CI `[-.086, .269]`.

Accordingly this is an **inconclusive boundary result**, not a successful
universal Control detector. It supports opponent adaptation as one observable
facet of Control while indicating that the original score-family Control
generator only weakly expresses that facet.

## Interpretation boundary

This experiment establishes neither naturally occurring PCC nor human intent.
It is a synthetic construct-validity check used to decide which measurements
are credible enough to take to anonymized human hand histories. No metric should
be redefined after seeing human outcomes without a fresh held-out confirmation.
