# Frozen PCC Robustness-Surface Protocol

## Purpose

The v0.3 policies produced the engineered cycle at their reference settings:

```text
Control > Pressure, Chaos > Control, Pressure > Chaos
```

This experiment asks whether that result persists when action stochasticity,
mode purity, and learning horizon change. The policies themselves are frozen.
The runner verifies the SHA-256 checksum of `pcc_poker/families.py` before any
simulation begins.

## Frozen grid

The complete Cartesian surface is:

- temperatures: `0.20`, `0.35`, `0.50`, `0.75`;
- dominant-mode purities: `0.70`, `0.80`, `0.90`, `1.00`;
- hands per matchup and seat order: `250`, `1,000`, `4,000`;
- fresh replicates per condition: `10`;
- first seed: `41001`, with a stride of `20` within each condition.

There are 48 conditions and 480 replicated sweeps. Each sweep contains all
three matchups in both seat orders, for 5,040,000 hands total.

Purity places the specified weight on the named mode and divides the remainder
equally between the other two axes. For example, Control at `.80` is
`(.10, .80, .10)`.

## Frozen acceptance rule

A condition preserves the cycle when all three replicate-mean payoff edges are
positive. The robustness criterion passes only when:

1. at least 80% of the 48 conditions preserve the complete cycle; and
2. no single mode beats both other modes in more than 20% of conditions.

The report also provides approximate 95% normal intervals across the ten
replicate-level, seat-balanced edges. Those intervals are descriptive and do
not change the condition or global acceptance decisions.

## Outputs

The JSON report retains the full design, aggregate decision, condition-level
edge summaries, and individual replicate seeds and edges. The companion CSV
contains one row per condition for heatmaps and independent analysis.

## Interpretation boundary

No policy receives an opponent mode label or matchup-specific payoff. A failed
condition is retained as a boundary result; the v0.3 policies must not be
retuned after inspecting this surface and then described as though the same
experiment confirmed them.

Passing would demonstrate robustness of these engineered Leduc policies only.
It would not show that PCC is universal, that humans instantiate the modes, or
that the label-free Control measurement problem has been solved.

## Frozen result

The global criterion did not pass. The complete cycle appeared in 36 of 48
conditions (`75%`), below the frozen `80%` requirement. The second criterion
did pass: Pressure, Control, and Chaos dominated both opponents in `14.6%`,
`6.3%`, and `4.2%` of conditions, respectively, so no mode dominated the grid.

Each canonical edge was individually widespread:

- Control over Pressure: 41/48 conditions (`85.4%`);
- Chaos over Control: 44/48 (`91.7%`);
- Pressure over Chaos: 45/48 (`93.8%`).

The boundary was strongly related to horizon. The cycle occurred in 8/16
conditions at 250 hands, 13/16 at 1,000 hands, and 15/16 at 4,000 hands. It was
also most stable at temperature `.35` (11/12) and purity `1.0` (11/12), and
least stable at temperature `.75` (7/12) and purity `.70` (7/12).

This supports a conditional result rather than a universal one: the engineered
cycle is highly persistent after longer learning horizons, but is not robust
over the complete preregistered surface. The policies were not retuned after
this result.
