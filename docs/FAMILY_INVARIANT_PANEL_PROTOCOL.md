# Family-invariant PCC measurement panel

## Question
Which label-free behavioral components survive a change in the engineered policy implementation strongly enough to be eligible for a future human-data measurement panel?

This is a **selection/falsification experiment**, not an attempt to force one observable onto each PCC axis. A missing axis remains missing.

## Frozen candidates
All quantities are computed before hidden mixture weights are consulted.

Pressure-targeted candidates: public Pressure exposure, response compression, predicted fold probability, and commitment ratio. Chaos-targeted candidates: normalized behavioral surprisal and the unchanged independent-value-floor effective surprisal.

Control is intentionally not supplied a new candidate in this experiment. Previous temporal and mechanism experiments show that Control measurement remains implementation-sensitive; inventing a new Control score after seeing those failures would defeat the purpose of an invariance test.

## Invariance rule
A candidate enters the conservative panel only when, in **both** independently coded policy families:

1. correlation with its intended synthetic weight is at least `0.20`;
2. that correlation exceeds each off-axis correlation by at least `0.05`; and
3. intended-axis correlations differ across families by no more than `0.20`.

The action model is calibrated on separate fresh seeds. Assigned PCC weights are used only after seat/mixture aggregation to evaluate construct validity.

## Interpretation
Passing this test means an observable is comparatively robust to these two engineered implementations. It does not establish a human latent state or justify causal interpretation. Human Hold'em also requires a separately justified value model before any effective-Chaos quantity can be used.
