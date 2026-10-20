"""Read-only, post-hoc reliability and covariate shift diagnostics for PCC.

Bootstrap intervals here quantify empirical measurement dispersion, not
repeatability of the same policy and not inferential confidence intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_family_failure_audit import audit as check_frozen_alignment
from .feature_ablation_audit import _data
from .low_dimensional_pcc_audit import BASELINE_INDICES, FEATURES, ENDPOINTS, standardize, ridge_fit_predict


def dispersion_bootstrap(x, seed=2801, repetitions=500):
    """Within-cohort bootstrap distribution of sample SD; deterministic."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('expected finite 2D measurements and at least three policies')
    if repetitions < 20:
        raise ValueError('at least 20 bootstrap repetitions required')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(repetitions, len(x)))
    sd = x[draws].std(axis=1, ddof=1)
    return {'sample_sd': x.std(axis=0, ddof=1).tolist(),
            'bootstrap_sd_percentile_025': np.quantile(sd, .025, axis=0).tolist(),
            'bootstrap_sd_percentile_975': np.quantile(sd, .975, axis=0).tolist(),
            'n_policies': len(x)}


def shift_diagnostics(train, test, threshold=3.):
    """Train-defined z shifts and training-range exceedances; no labels used."""
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError('incompatible train and test feature matrices')
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('non-finite features')
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError('threshold must be finite and positive')
    z, tz = standardize(train, test)
    outside = (test < train.min(axis=0)) | (test > train.max(axis=0))
    return {'train_sd_zero': (train.std(axis=0) < 1e-10).tolist(),
            'abs_test_mean_z': np.abs(tz.mean(axis=0)).tolist(),
            'test_outside_train_range_fraction': outside.mean(axis=0).tolist(),
            'test_abs_z_gt_threshold_fraction': (np.abs(tz) > threshold).mean(axis=0).tolist(),
            'any_feature_outside_training_range_fraction': float(outside.any(axis=1).mean()),
            'any_feature_abs_z_gt_threshold_fraction': float((np.abs(tz) > threshold).any(axis=1).mean()),
            'threshold_training_sd': float(threshold)}


def error_shift_diagnostics(train, test, y_train, y_test, penalty=10., threshold=3.):
    """Fixed baseline model: descriptive test error by extrapolation strata.

    Model and standardization are trained only on source cohort. This
    diagnostic must not be used to choose features on the held-out cohort.
    """
    z, tz = standardize(train, test)
    prediction = ridge_fit_predict(z, y_train, tz, penalty)
    outside = ((test < train.min(axis=0)) | (test > train.max(axis=0))).any(axis=1)
    distant = (np.abs(tz) > threshold).any(axis=1)
    squared = (prediction - y_test) ** 2
    result = {}
    for label, mask in (('outside_training_range', outside), ('abs_z_above_threshold', distant)):
        result[label] = {
            'n_flagged': int(mask.sum()), 'n_unflagged': int((~mask).sum()),
            'mean_squared_error_flagged': squared[mask].mean(axis=0).tolist() if mask.any() else None,
            'mean_squared_error_unflagged': squared[~mask].mean(axis=0).tolist() if (~mask).any() else None,
        }
    return result


def audit(directory, seed=2801, repetitions=500, threshold=3., penalty=10.):
    root = Path(directory)
    check_frozen_alignment(root)
    rows = json.loads((root / 'measurements.json').read_text())
    result = {'scope': 'post-hoc descriptive measurement dispersion and shift; not new policy replications',
              'seed': seed, 'bootstrap_repetitions': repetitions, 'ridge_penalty_fixed': penalty,
              'frozen_evidence_modified': False, 'families': {}, 'directions': {},
              'features': ['mean_action_entropy', 'bet_raise_frequency', 'mean_regret', *FEATURES],
              'endpoints': list(ENDPOINTS),
              'limitations': [
                  'Across-policy bootstrap dispersion is NOT repeated-measure reliability or test-retest stability.',
                  'Shift/error associations are descriptive and do not establish why transfer fails.',
                  'Existing held-out outcomes were already inspected; do not select new models using these diagnostics.',
                  'Only 32 policies per family; no independent policy generation or confirmatory inference.']}
    values = {}
    for family in ('A', 'B'):
        base = _data(rows, family, 'baseline')[:, BASELINE_INDICES]
        candidate = _data(rows, family, 'candidate')
        outcomes = _data(rows, family, 'outcomes')
        if base.shape != (32, 3) or candidate.shape != (32, 3) or outcomes.shape != (32, 3):
            raise ValueError('unexpected frozen measurement dimensions')
        combined = np.column_stack((base, candidate))
        values[family] = (combined, outcomes)
        result['families'][family] = {
            'candidate_dispersion': dispersion_bootstrap(candidate, seed=seed + (family == 'B'), repetitions=repetitions),
            'baseline_candidate_correlation': np.corrcoef(combined, rowvar=False)[:3, 3:].tolist(),
        }
    for direction, train_family, test_family in (('A_to_B', 'A', 'B'), ('B_to_A', 'B', 'A')):
        train, yt = values[train_family]
        test, yv = values[test_family]
        result['directions'][direction] = {
            'shift': shift_diagnostics(train, test, threshold),
            'fixed_baseline_error_by_shift': error_shift_diagnostics(train[:, :3], test[:, :3], yt, yv, penalty, threshold),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='validation/beyond-strategy-blind-v1')
    parser.add_argument('--seed', type=int, default=2801)
    parser.add_argument('--bootstrap-repetitions', type=int, default=500)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.directory, seed=args.seed, repetitions=args.bootstrap_repetitions),
                     indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
