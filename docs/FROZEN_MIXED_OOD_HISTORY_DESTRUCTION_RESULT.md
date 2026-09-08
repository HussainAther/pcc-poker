# Frozen mixed OOD history-destruction result

## Status

Post-v0.8 synthetic prospective control. The analysis follows `MIXED_OOD_HISTORY_DESTRUCTION_PROTOCOL.md` using the frozen mixed-recovery training set and prespecified mixed OOD evaluation set.

This result does not modify the v0.8 human-facing measurement contract.

## Primary result

Within-hand action order was destroyed only inside the M3 transition-feature extractor while M0-M2 were held fixed exactly. Across 25 deterministic yoked permutation replicates:

| Model / control | Overall MAE |
|---|---:|
| M2 public context/state | 0.11035 |
| M3 original sequence | 0.09339 |
| M3 shuffled sequence, mean | 0.09997 |

The original M2 -> M3 gain was `0.01695` MAE. After order destruction, the surviving mean gain was `0.01038`. Thus the estimated gain attenuation was **38.8%**.

The preregistered 50% overall attenuation criterion therefore **failed**.

## Regional result

| OOD region | M2 MAE | M3 original | M3 shuffled mean | Gain attenuation |
|---|---:|---:|---:|---:|
| Pressure Heavy | 0.08003 | 0.07514 | 0.07400 | -23.3% |
| Control Heavy | 0.12634 | 0.06603 | 0.08307 | 28.2% |
| Chaos Heavy | 0.17312 | 0.15711 | 0.17158 | 90.4% |
| Balanced | 0.07243 | 0.07549 | 0.07887 | n/a |
| Boundary | 0.09981 | 0.09319 | 0.09233 | -13.1% |

The prespecified Control-heavy focal region improved from M2 MAE `0.12634` to original M3 MAE `0.06603`. Shuffled M3 retained MAE `0.08307`, corresponding to only **28.2%** attenuation. The preregistered 50% Control-heavy criterion therefore **failed**.

Chaos-heavy behaved differently: destroying order removed approximately **90.4%** of the original M2 -> M3 gain. In this engineered family, the late Chaos-heavy recovery gain is therefore much more sequence-order dependent than the Control-heavy gain.

Pressure-heavy and Boundary do not support a temporal-order interpretation under this control: shuffled transitions performed slightly better than the original M3 transition block on average, producing negative attenuation estimates. Balanced remains the frozen negative region, with M3 worse than M2 in both original and shuffled conditions.

## Prespecified checks

```text
original M3 beats M2 overall:                 PASS
overall gain >=50% attenuated by destruction: FAIL
Control-heavy gain >=50% attenuated:          FAIL
```

## Interpretation

The strong version of the temporal-history hypothesis is **not supported**.

Sequential transition features do contain useful ordering information overall, because destruction worsened M3 from `0.09339` to a mean `0.09997`. However, most of the original M3 advantage survives the destruction control. The transition block therefore appears to encode a mixture of:

- genuine order-sensitive information;
- local action-composition structure recoverable even after within-hand permutation; and
- region-specific effects that are not uniform across PCC coordinates.

For Control-heavy mixtures specifically, the prior large M2 -> M3 gain should **not** be described as primarily temporal. The result instead points toward richer local composition or interaction structure as the larger source of that recovery advantage.

The Chaos-heavy result is the main positive mechanistic lead: its M3 advantage is strongly order dependent and merits a targeted follow-up that separates transition directionality from simpler run-length, alternation, or action-pair composition statistics.

## Scientific boundary

This remains synthetic identifiability evidence in one engineered policy family. It does not establish human PCC constructs, a human Control or Chaos detector, psychological state, or cross-family invariance.

## Next experiment

The next prospective decomposition should focus on **what survives the shuffle**. Compare M2 against transition controls that progressively retain only:

1. unordered adjacent action-pair composition;
2. directionless transition counts (A-B pooled with B-A);
3. run-length / repetition statistics; and
4. fully directed transitions.

This would identify whether the surviving Control-heavy signal comes from local pair composition, directionality, persistence, or a combination of these.
