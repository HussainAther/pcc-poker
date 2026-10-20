"""Exploratory low-dimensional, training-only-scaled PCC transfer audit.

This is a post-hoc sensitivity analysis, not confirmation or model selection.
The three conventional descriptors are fixed by measurement definition:
mean action entropy (index 5), betting/raising frequency (6), and
oracle regret (7). Never use held-out labels to choose a feature or penalty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_family_failure_audit import audit as verify_alignment
from .feature_ablation_audit import _data

BASELINE_INDICES = (5, 6, 7)
FEATURES = ('pressure', 'control', 'chaos')
ENDPOINTS = ('value_vs_tight', 'value_vs_loose', 'heldout_card_response')


def standardize(train, test):
    """Training-only centering/scaling; constants are left at unit scale."""
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-10, 1., scale)
    return (train-center)/scale, (test-center)/scale


def ridge_fit_predict(train, outcomes, test, penalty):
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('penalty must be positive and finite')
    center = outcomes.mean(axis=0)
    coef = np.linalg.solve(train.T @ train + penalty*np.eye(train.shape[1]),
                           train.T @ (outcomes-center))
    return test @ coef + center


def residualize(train_base, train_candidates, test_base, test_candidates, penalty):
    """Residualize standardized candidates against standardized baseline.

    Intercept is zero after centering on the training cohort. The projection
    is fitted solely on training data, then applied unchanged to test data.
    """
    coef = np.linalg.solve(train_base.T @ train_base + penalty*np.eye(train_base.shape[1]),
                           train_base.T @ train_candidates)
    return (train_candidates-train_base @ coef,
            test_candidates-test_base @ coef)


def diagnostics(train, test):
    s = np.linalg.svd(train, compute_uv=False)
    rank = int(np.sum(s > max(train.shape)*np.finfo(float).eps*s[0])) if len(s) and s[0] else 0
    return {
        'features': int(train.shape[1]), 'numerical_rank': rank,
        'condition_number': float(s[0]/s[-1]) if rank == train.shape[1] else None,
        'test_mean_shift_training_sd_units': [float(v) for v in test.mean(axis=0)],
        'test_rms_training_sd_units': [float(v) for v in np.sqrt(np.mean(test**2, axis=0))],
    }


def audit(directory, penalties=(1., 10., 100.), residual_penalty=10.):
    root = Path(directory)
    verify_alignment(root)
    if not penalties or any(not np.isfinite(p) or p <= 0 for p in penalties):
        raise ValueError('penalties must be positive and finite')
    if not np.isfinite(residual_penalty) or residual_penalty <= 0:
        raise ValueError('residual penalty must be positive and finite')
    rows = json.loads((root/'measurements.json').read_text())
    output = {
        'scope': 'post-hoc exploratory sensitivity analysis, not a preregistered result',
        'frozen_evidence_modified': False,
        'baseline_indices': list(BASELINE_INDICES),
        'baseline_descriptors': ['mean_action_entropy', 'bet_raise_frequency', 'mean_regret'],
        'ridge_penalties': [float(p) for p in penalties],
        'residual_projection_penalty': float(residual_penalty), 'directions': {},
        'limitations': [
            'No penalty/feature selection on held-out family is justified by these reported scores.',
            'Only 32 policies per family; these are exploratory fits on existing frozen measurements.',
            'Residualized feature transfer can extrapolate under family shift; inspect diagnostics.',
            'No new independent samples, causal identification, or confirmatory significance test.',
        ],
    }
    for direction, source, target in (('A_to_B', 'A', 'B'), ('B_to_A', 'B', 'A')):
        xb = _data(rows, source, 'baseline')[:, BASELINE_INDICES]
        tb = _data(rows, target, 'baseline')[:, BASELINE_INDICES]
        xc = _data(rows, source, 'candidate')
        tc = _data(rows, target, 'candidate')
        y = _data(rows, source, 'outcomes')
        ty = _data(rows, target, 'outcomes')
        if xb.shape != (32, 3) or xc.shape != (32, 3) or y.shape != (32, 3):
            raise ValueError('unexpected input dimensions')
        b, bt = standardize(xb, tb)
        c, ct = standardize(xc, tc)
        rc, rct = residualize(b, c, bt, ct, residual_penalty)
        designs = {'baseline': (b, bt), 'all': (np.c_[b, c], np.c_[bt, ct]),
                   'residualized_all': (np.c_[b, rc], np.c_[bt, rct])}
        for i, name in enumerate(FEATURES):
            designs[name] = (np.c_[b, c[:, i]], np.c_[bt, ct[:, i]])
        variance = y.var(axis=0)
        if np.any(variance <= 0):
            raise ValueError('zero training outcome variance')
        results = {}
        for penalty in penalties:
            pred = {name: ridge_fit_predict(z, y, tz, penalty)
                    for name, (z, tz) in designs.items()}
            base_err = (pred['baseline']-ty)**2
            results[str(float(penalty))] = {
                name: {end: float(np.mean((base_err[:, j]-(p-ty)[:, j]**2)/variance[j]))
                       for j, end in enumerate(ENDPOINTS)}
                for name, p in pred.items()
            }
        output['directions'][direction] = {
            'n_train': 32, 'n_test': 32,
            'baseline_condition': diagnostics(b, bt),
            'combined_condition': diagnostics(np.c_[b, c], np.c_[bt, ct]),
            'residual_candidate_train_rms': [float(v) for v in np.sqrt(np.mean(rc**2, axis=0))],
            'residual_candidate_test_rms': [float(v) for v in np.sqrt(np.mean(rct**2, axis=0))],
            'improvement_vs_low_dim_baseline': results,
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='validation/beyond-strategy-blind-v1')
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.directory), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
