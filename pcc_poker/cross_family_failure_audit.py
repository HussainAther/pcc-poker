"""Read-only, post-hoc diagnostics for Beyond Strategy's frozen transfer test.

Descriptive diagnostics only: not a new confirmatory significance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ENDPOINTS = ('value_vs_tight', 'value_vs_loose', 'heldout_card_response')


def _matrix(records, field):
    a = np.asarray([r[field] for r in records], dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('invalid ' + field)
    return a


def _correlations(x, y):
    # Constant columns are reported as null, not silently assigned zero correlation.
    out = []
    for j in range(y.shape[1]):
        vals = []
        for i in range(x.shape[1]):
            if np.std(x[:, i]) <= 1e-12 or np.std(y[:, j]) <= 1e-12:
                continue
            vals.append(abs(float(np.corrcoef(x[:, i], y[:, j])[0, 1])))
        out.append(max(vals) if vals else None)
    return out


def audit(directory):
    root = Path(directory)
    rows = json.loads((root / 'measurements.json').read_text())
    frozen = json.loads((root / 'predictions_frozen.json').read_text())
    if len(rows) != 64 or len({r['id'] for r in rows}) != 64:
        raise ValueError('expected 64 unique frozen policies')
    cohorts = {c: sorted((r for r in rows if r['cohort'] == c), key=lambda r: r['replicate']) for c in ('A', 'B')}
    if any(len(v) != 32 or [r['replicate'] for r in v] != list(range(32)) for v in cohorts.values()):
        raise ValueError('invalid cohort or replicate index')
    result = {'scope': 'post-hoc descriptive diagnostics; no refitting or new significance tests',
              'frozen_evidence_modified': False, 'directions': {}}
    for direction, train_key, test_key in [('A_to_B', 'A', 'B'), ('B_to_A', 'B', 'A')]:
        train, test = cohorts[train_key], cohorts[test_key]
        xtr, xte = _matrix(train, 'baseline'), _matrix(test, 'baseline')
        ctr, cte = _matrix(train, 'candidate'), _matrix(test, 'candidate')
        y = _matrix(test, 'outcomes')
        pred = frozen[direction]
        b = np.asarray(pred['base_predictions'], dtype=float)
        a = np.asarray(pred['augmented_predictions'], dtype=float)
        if b.shape != y.shape or a.shape != y.shape or y.shape != (32, 3):
            raise ValueError('predictions/outcomes shape mismatch: ' + direction)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite predictions')
        # The frozen arrays follow replicate order; never sort by outcome or prediction.
        sq_b, sq_a = (b-y)**2, (a-y)**2
        var = np.asarray(pred['training_variance_scale'], dtype=float)
        if var.shape != (3,) or np.any(var <= 0):
            raise ValueError('invalid frozen training variance')
        gain = np.mean((sq_b-sq_a)/var, axis=0)
        frozen_gain = np.asarray(pred['improvement'], dtype=float)
        if not np.allclose(gain, frozen_gain, rtol=1e-7, atol=1e-9):
            raise ValueError('outcome/prediction alignment mismatch: ' + direction)
        std = np.std(ctr, axis=0)
        pooled = np.vstack((ctr, cte))
        scale = np.std(pooled, axis=0)
        shift = [float(abs(np.mean(cte[:, i])-np.mean(ctr[:, i])) / scale[i]) if scale[i] > 1e-12 else None for i in range(3)]
        endpoints = {}
        for j, name in enumerate(ENDPOINTS):
            d = (sq_b[:, j]-sq_a[:, j])/var[j]
            endpoints[name] = {
                'normalized_mse_improvement': float(gain[j]),
                'fraction_test_policies_helped': float(np.mean(d > 0)),
                'median_policy_improvement': float(np.median(d)),
                'worst_policy_improvement': float(np.min(d)),
                'best_policy_improvement': float(np.max(d)),
                'negative_transfer': bool(gain[j] < 0),
            }
        result['directions'][direction] = {
            'train_cohort': train_key, 'test_cohort': test_key,
            'n_train': 32, 'n_test': 32,
            'candidate_shift_pooled_sd_units': shift,
            'candidate_train_standard_deviation': std.tolist(),
            'max_absolute_candidate_baseline_correlation_train': _correlations(xtr, ctr),
            'max_absolute_candidate_baseline_correlation_test': _correlations(xte, cte),
            'endpoints': endpoints,
        }
    result['limitations'] = [
        'Exploratory analysis performed after outcomes were revealed; no confirmatory p-values.',
        'Feature correlation does not prove redundancy or causation.',
        'Cohort differences and predictive error cannot isolate the mechanism of failure.',
        'No models are retrained and no frozen files are overwritten.',
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='validation/beyond-strategy-blind-v1')
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.directory), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
