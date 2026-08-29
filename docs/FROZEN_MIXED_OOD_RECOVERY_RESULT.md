# Frozen Mixed OOD Recovery Result

## Status

Completed prospective out-of-distribution evaluation of continuous PCC mixture-weight recovery.

This result evaluates whether a recovery model trained on ordinary synthetic Dirichlet mixtures generalizes to prespecified out-of-distribution regions of the Pressure-Control-Chaos simplex.

The evaluation uses only observable behavioral features.

## Training distribution

The recovery models were trained on:

```text
outputs/mixed-recovery-data.jsonl
```

Training examples:

* 120 mixture-seat examples
* ordinary synthetic Dirichlet mixtures
* continuous Pressure / Control / Chaos target weights

No OOD examples were used for training.

## OOD evaluation design

OOD dataset:

```text
outputs/mixed-ood-data.jsonl
```

Generation command:

```bash
python -m pcc_poker mixed-ood-dataset \
  --mixtures-per-region 20 \
  --hands-per-seat 100 \
  --temperature 0.35 \
  --seed 51 \
  --output outputs/mixed-ood-data.jsonl
```

Analysis command:

```bash
python -m pcc_poker mixed-ood-analyze \
  --training outputs/mixed-recovery-data.jsonl \
  --ood outputs/mixed-ood-data.jsonl \
  --output outputs/mixed-ood-results.json
```

The OOD design contained five prespecified regions:

1. `pressure_heavy`
2. `control_heavy`
3. `chaos_heavy`
4. `balanced`
5. `boundary`

There were:

* 20 mixtures per region
* 100 total mixtures
* 100 hands per seat
* 20,000 total simulated hands
* 200 OOD evaluation examples after aggregation across seats

The focal and reference temperatures were both 0.35.

## Primary prediction target

Continuous latent synthetic PCC weights:

```text
Pressure
Control
Chaos
```

The main comparison was between:

1. an action-frequency baseline; and
2. a contextual-history recovery model.

The primary metric was mean absolute error (MAE) on the continuous PCC mixture weights.

## Overall result

| Model                     |     MAE |    RMSE | Dominant-mode accuracy |
| ------------------------- | ------: | ------: | ---------------------: |
| Action-frequency baseline | 0.14162 | 0.18768 |                  0.845 |
| Contextual-history model  | 0.09339 | 0.12751 |                  0.865 |

Relative MAE improvement of the contextual-history model over the action-frequency baseline:

```text
34.05%
```

The prespecified overall criterion was satisfied:

```text
contextual MAE < action-frequency MAE
```

## Per-mode error

### Action-frequency baseline

| Mode     |     MAE |
| -------- | ------: |
| Pressure | 0.07505 |
| Control  | 0.15256 |
| Chaos    | 0.19725 |

### Contextual-history model

| Mode     |     MAE |
| -------- | ------: |
| Pressure | 0.05493 |
| Control  | 0.10877 |
| Chaos    | 0.11647 |

Contextual recovery improved MAE for all three latent components overall.

The largest absolute difficulty remained Chaos.

## OOD region results

### Pressure-heavy

| Model              |     MAE | Dominant-mode accuracy |
| ------------------ | ------: | ---------------------: |
| Action-frequency   | 0.14107 |                   1.00 |
| Contextual-history | 0.07514 |                   1.00 |

Contextual recovery strongly outperformed the action-frequency baseline.

### Control-heavy

| Model              |     MAE | Dominant-mode accuracy |
| ------------------ | ------: | ---------------------: |
| Action-frequency   | 0.16382 |                   0.90 |
| Contextual-history | 0.06603 |                   1.00 |

This was the strongest relative OOD advantage for the contextual model.

### Chaos-heavy

| Model              |     MAE | Dominant-mode accuracy |
| ------------------ | ------: | ---------------------: |
| Action-frequency   | 0.23157 |                   0.95 |
| Contextual-history | 0.15711 |                   1.00 |

Contextual recovery outperformed the baseline, although Chaos-heavy mixtures remained the hardest extreme region in absolute error.

### Balanced

| Model              |     MAE | Dominant-mode accuracy |
| ------------------ | ------: | ---------------------: |
| Action-frequency   | 0.04830 |                   0.60 |
| Contextual-history | 0.07549 |                  0.475 |

This was the only prespecified region in which contextual-history recovery performed worse than the action-frequency baseline.

This failure is retained as part of the frozen result.

A plausible interpretation is that near-balanced mixtures contain relatively little directional structure for contextual features to exploit, while simple aggregate action frequencies may already provide a low-variance estimate near the simplex center.

That interpretation remains a hypothesis and is not established by this experiment alone.

### Boundary

| Model              |     MAE | Dominant-mode accuracy |
| ------------------ | ------: | ---------------------: |
| Action-frequency   | 0.12334 |                  0.775 |
| Contextual-history | 0.09319 |                   0.85 |

Contextual recovery outperformed the baseline on mixtures with one component constrained near zero.

## Prespecified checks

The following checks passed:

```text
contextual_beats_action_frequency_overall = true

contextual_beats_action_frequency_in_majority_of_regions = true
```

The contextual model outperformed the action-frequency baseline in four of five OOD regions:

```text
pressure_heavy   PASS
control_heavy    PASS
chaos_heavy      PASS
balanced         FAIL
boundary         PASS
```

## Interpretation

Within this engineered synthetic PCC policy family, contextual and historical behavioral information improves recovery of continuous Pressure-Control-Chaos mixture weights under a deliberately shifted evaluation distribution.

The improvement is not confined to ordinary held-out mixtures from the same random generator.

The contextual model generalized better than an action-frequency baseline overall and in four of five prespecified OOD simplex regions.

The strongest gains occurred in Control-heavy and Pressure-heavy regions.

Chaos-heavy mixtures remained comparatively difficult, but contextual information substantially reduced their recovery error.

The balanced region produced the opposite result: simple action frequencies recovered mixture weights more accurately than the richer contextual-history model.

This suggests that the informational value of context is heterogeneous across the PCC simplex rather than uniformly beneficial.

## What this result supports

This experiment supports the narrower claim that:

> Continuous synthetic PCC mixture weights are recoverable from observable behavior beyond aggregate action frequencies, and the contextual recovery advantage generalizes to several prespecified out-of-distribution regions of the engineered PCC simplex.

It also provides evidence that contextual information is especially useful for identifying strongly structured or asymmetric mixtures.

## What this result does not establish

This experiment does **not** establish:

* that human poker behavior follows PCC;
* that Pressure, Control, and Chaos are universal latent variables;
* that these observables transfer to non-poker domains;
* that the recovery model has identified causal psychological states;
* that the contextual model is uniformly superior everywhere in the PCC simplex;
* or that the balanced-region explanation proposed above is correct.

The result concerns synthetic identifiability and generalization within the engineered PCC policy family.

## Frozen conclusion

The OOD recovery hypothesis is supported under the prespecified evaluation.

Overall:

```text
Action-frequency MAE:  0.14162
Contextual MAE:        0.09339
Relative improvement: 34.05%
```

Regional result:

```text
Contextual model wins: 4 / 5 regions
```

The balanced-region failure is retained without post-hoc modification.

No model, threshold, sampling region, or evaluation criterion should be changed retroactively for this frozen result.

## Next experiment

The next planned experiment should decompose which observable contextual features are responsible for the OOD recovery advantage.

A prospective nested feature-ablation design is recommended:

```text
M0 = action frequencies only

M1 = M0 + betting/public context

M2 = M1 + strength/equity-related context

M3 = M2 + sequential/history features
```

The same training and OOD evaluation distributions should be reused where possible.

The primary question is:

> Which additional behavioral information produces the contextual-history model's OOD recovery advantage, and does that contribution differ across simplex regions?

The balanced region should be analyzed as an informative negative case rather than tuned away.

