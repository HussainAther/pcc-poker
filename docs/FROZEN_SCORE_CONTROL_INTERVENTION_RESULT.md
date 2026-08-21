# Frozen Score-Control intervention result

## Status

**Partial. Poker Control is not yet structurally resolved across both families.**

The single prospective contextual-gain intervention caused the Score family to
recover the first two stages of the already-frozen Control gate, while the
final value-sensitive intervention stage remained below threshold.

| Stage | Score Control correlation | Score discriminant margin | Score status | Adaptive status |
|---|---:|---:|---|---|
| Information uptake | 0.358 | 0.449 | **pass** | **pass** |
| Context alignment | 0.315 | 0.311 | **pass** | **pass** |
| Value-sensitive intervention | 0.132 | 0.049 | fail | **pass** |

The unchanged thresholds are Control correlation >= `0.20` and discriminant
margin >= `0.05`.

## Interpretation

The intervention materially improved Score Control's use of learned opponent
response information without changing the measurement or the Adaptive family.
This supports the preceding architectural diagnosis: weak contextual response
sensitivity was responsible for the missing information-uptake and
context-alignment stages.

However, stronger response sensitivity alone did **not** make Score Control's
contextual changes sufficiently value-sensitive. The final stage remains the
specific unresolved link.

No second gain adjustment was attempted after observing this result.

## Scientific boundary

This is a post-v0.8 synthetic extension only. It does not alter the frozen
Pressure-only human-facing panel, does not authorize human-data analysis, and
does not change ORIA gating.
