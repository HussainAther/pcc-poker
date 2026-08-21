# Frozen Score-Control value-aware intervention result

## Result

**Partial; intervention rejected as a route to full Control resolution.**

The prospective intervention preserved the contextual-response gain at `3.35`
and required an aggressive action to have frozen counterfactual efficiency of at
least `0.80` before context could alter its Score-Control value.

The existing three-stage recovery gate was then rerun unchanged.

| Stage | Score Control correlation | Score discriminant margin | Score stage |
| --- | ---: | ---: | --- |
| Information uptake | 0.196 | 0.179 | failed |
| Context alignment | 0.256 | 0.264 | passed |
| Value-sensitive intervention | 0.101 | 0.142 | failed |

Adaptive remains part of the same cross-family gate; the overall structural
recovery is not confirmed.

Compared with the preceding contextual-only intervention, the value-aware gate
did not improve the target final stage. It also reduced the Score information-
uptake correlation from above threshold to just below threshold.

## Interpretation

The result rejects the simple hypothesis that **hard-gating contextual
aggression at counterfactual efficiency >= 0.80** is sufficient to make Score
Control value-sensitive. Context alignment remains measurable, but the final
context-by-value signal weakens further.

No threshold, seed, gain, policy component, or measurement was changed after
observing this result. This is a post-v0.8 synthetic result only. The frozen
human-facing Control axis remains unresolved and the human-data gate remains
closed.
