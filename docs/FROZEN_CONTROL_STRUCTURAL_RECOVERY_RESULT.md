# Frozen post-v0.8 Poker Control structural-recovery result

## Result

The prospective three-stage Control structure was **not recovered across both synthetic implementation families**.

```text
information uptake -> context alignment -> value-sensitive intervention
```

The Adaptive family passed both prespecified checks for all three stages. The Score family failed both checks for all three stages. Therefore the result is retained as **partial synthetic mechanism evidence**, not cross-family Control resolution.

| Family | Information uptake | Context alignment | Value-sensitive intervention | All three |
| --- | --- | --- | --- | --- |
| Score | failed | failed | failed | no |
| Adaptive | passed | passed | passed | yes |

### Control correlations

| Family | Information uptake | Context alignment | Value-sensitive intervention |
| --- | ---: | ---: | ---: |
| Score | 0.003 | -0.023 | -0.087 |
| Adaptive | 0.654 | 0.543 | 0.381 |

### Discriminant margins

| Family | Information uptake | Context alignment | Value-sensitive intervention |
| --- | ---: | ---: | ---: |
| Score | -0.156 | -0.205 | -0.153 |
| Adaptive | 0.850 | 0.720 | 0.557 |

The matched context-yoke preserved both static-context action margins and global action margins.

## Interpretation

This result localizes the current Poker Control portability problem. The three-stage structure is strongly recoverable in the Adaptive implementation, but it is not expressed by the Score implementation under the frozen definitions and thresholds. The experiment therefore does **not** justify adding Control to the conservative human-facing panel.

The next synthetic question should investigate why Score-family Control is structurally inert under these tests rather than retuning the metric to force cross-family agreement.

## Boundary

No human data were accessed. The immutable v0.8.0 freeze and Pressure-only human measurement contract remain unchanged.
