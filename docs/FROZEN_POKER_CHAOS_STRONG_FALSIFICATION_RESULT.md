# Frozen Poker Chaos strong falsification result

The prespecified post-v0.8 synthetic Chaos falsification **passed across both policy families**.

## Exploiter calibration

The selected exploiter was `adaptive-pressure-cold`, with weights `(0.8, 0.1, 0.1)` and temperature `0.15`. It was selected using only predictable-baseline calibration outcomes and reduced the predictable baseline to mean payoff **-0.955**.

## Held-out summaries

| Policy | Mean normalized entropy | Payoff vs neutral | Payoff vs frozen exploiter |
|---|---:|---:|---:|
| Predictable value | 0.000 | -0.659 | -0.716 |
| Uniform random | **1.000** | **-1.447** | **-1.794** |
| Score Chaos | 0.908 | -0.300 | -0.439 |
| Independent Chaos | 0.539 | **+0.432** | **+0.737** |

Both Chaos families passed every prespecified held-out check.

## Interpretation

The result directly rejects the shortcut `Chaos = randomness` in this synthetic Poker laboratory. Uniform random legal play is maximally entropic but destroys substantially more value and performs substantially worse against the frozen exploiter. The two existing Chaos implementations are less random than the random baseline yet preserve more value and are harder for the separately calibrated exploiter to punish.

The supported structural statement is therefore closer to:

`effective Chaos = unpredictability + strategic adequacy + exploitability resistance`.

This is a post-freeze synthetic result. It does **not** modify the v0.8.0 freeze, authorize human-data analysis, or silently promote the frozen human-facing Chaos panel.
