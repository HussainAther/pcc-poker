"""Patch 33: frozen, fresh-cohort representation-intervention study.

Tests whether bounding PCC features to the *training* support mitigates
A->B negative transfer. Intervention is on predictor inputs, NOT the game's
causal mechanisms. Prior studies motivated this hypothesis: this is a
protocol-first follow-up, not independent preregistration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import blind_discriminant as experiment
from .leduc_cfr import LeducCFR
from .low_dimensional_pcc_audit import ENDPOINTS, ridge_fit_predict, standardize
from .prospective_pcc_replication import SOURCES, sha, write_exclusive

PROTOCOL = {
    'id': 'pcc-representation-intervention-v4',
    'replications': 8,
    'policies_per_family': 24,
    'seed_A': 7300001,
    'seed_B': 9300001,
    'replication_stride': 100019,
    'policy_stride': 149,
    'baseline_columns': [5, 6, 7],
    'candidate_columns': [0, 1, 2],
    'ridge_penalty': 10.0,
    'endpoints': list(ENDPOINTS),
    'primary_direction': 'A_to_B',
    'primary_endpoint': 'value_vs_loose',
    'primary_statistic': 'mean cohort normalized MSE improvement: clipped PCC vs unchanged PCC',
    'primary_criterion': 'mean clipped-vs-unchanged improvement > 0',
    'bootstrap_seed': 33033,
    'bootstrap_repetitions': 5000,
    'confidence_percentiles': [2.5, 97.5],
    'interventions': {
        'unchanged': 'training-only standardized baseline+PCC',
        'clip_pcc_to_training_range': 'only standardized test PCC values clipped to per-column train min/max',
        'clip_all_to_training_range': 'standardized test baseline and PCC values clipped to training min/max (secondary)',
    },
    'no_selection': 'all comparisons fixed, no changes based on results',
}
SOURCE_NAMES = tuple(sorted(set(SOURCES + ('prospective_transfer_asymmetry.py',
                                           'prospective_representation_intervention.py',))))


def frozen_object():
    directory = Path(__file__).parent
    sources = {name: sha((directory / name).read_bytes()) for name in SOURCE_NAMES}
    return {'protocol': PROTOCOL, 'sources': sources,
            'status': 'frozen_before_policy_generation'}


def freeze(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_exclusive(directory / 'protocol.json', frozen_object())
    return {'status': 'frozen', 'protocol_file_sha256': sha((directory / 'protocol.json').read_bytes())}


def verify(directory, require_fresh=True):
    directory = Path(directory)
    observed = json.loads((directory / 'protocol.json').read_text(encoding='utf8'))
    if observed != frozen_object():
        raise RuntimeError('frozen protocol or source commitments mismatch')
    if require_fresh and any((directory / name).exists() for name in ('measurements.json', 'results.json')):
        raise FileExistsError('experiment already generated; refusing overwrite')
    return observed


def seed(family, replication, index):
    return (PROTOCOL['seed_A' if family == 'A' else 'seed_B'] +
            replication * PROTOCOL['replication_stride'] +
            index * PROTOCOL['policy_stride'])


def _arrays(rows, key, cols=None):
    result = np.asarray([r[key] for r in rows], dtype=float)
    return result if cols is None else result[:, cols]


def _clamp_to_training(train, test):
    return np.clip(test, np.min(train, axis=0), np.max(train, axis=0))


def evaluate(train, test):
    """Standardize with training statistics; change only test representations."""
    if len(train) < 3 or len(test) < 3:
        raise ValueError('not enough policies')
    xb = _arrays(train, 'baseline', PROTOCOL['baseline_columns'])
    tb = _arrays(test, 'baseline', PROTOCOL['baseline_columns'])
    xc = _arrays(train, 'candidate', PROTOCOL['candidate_columns'])
    tc = _arrays(test, 'candidate', PROTOCOL['candidate_columns'])
    y, truth = _arrays(train, 'outcomes'), _arrays(test, 'outcomes')
    if xb.shape[1] != 3 or xc.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError('unexpected shape')
    if not all(np.isfinite(a).all() for a in (xb, tb, xc, tc, y, truth)):
        raise ValueError('nonfinite data')
    b, bt = standardize(xb, tb)
    c, ct = standardize(xc, tc)
    penalty = PROTOCOL['ridge_penalty']
    fitted = np.c_[b, c]
    predictions = {
        'baseline': ridge_fit_predict(b, y, bt, penalty),
        'unchanged': ridge_fit_predict(fitted, y, np.c_[bt, ct], penalty),
        'clip_pcc_to_training_range': ridge_fit_predict(
            fitted, y, np.c_[bt, _clamp_to_training(c, ct)], penalty),
        'clip_all_to_training_range': ridge_fit_predict(
            fitted, y, np.c_[_clamp_to_training(b, bt), _clamp_to_training(c, ct)], penalty),
    }
    variance = y.var(axis=0)
    if np.any(variance <= 1e-12):
        raise ValueError('degenerate training outcomes')
    mse = {name: np.mean((prediction - truth)**2, axis=0)
           for name, prediction in predictions.items()}
    return {
        'n_train': len(train), 'n_test': len(test),
        'mse': {name: dict(zip(ENDPOINTS, values.tolist())) for name, values in mse.items()},
        'clipped_pcc_fraction': dict(zip(('pressure','control','chaos'),
            np.mean(ct != _clamp_to_training(c, ct), axis=0).tolist())),
        'normalized_improvement_vs_unchanged': {
            name: dict(zip(ENDPOINTS, ((mse['unchanged'] - values) / variance).tolist()))
            for name, values in mse.items() if name != 'unchanged'},
        'normalized_improvement_vs_baseline': {
            name: dict(zip(ENDPOINTS, ((mse['baseline'] - values) / variance).tolist()))
            for name, values in mse.items() if name != 'baseline'},
    }


def summarize(cohorts):
    if len(cohorts) < 2:
        raise ValueError('need at least two independently seeded cohorts')
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    draws = rng.integers(0, len(cohorts),
                         size=(PROTOCOL['bootstrap_repetitions'], len(cohorts)))
    result = {'n_cohorts': len(cohorts), 'directions': {}}
    for direction in ('A_to_B', 'B_to_A'):
        out = {}
        for intervention in ('clip_pcc_to_training_range', 'clip_all_to_training_range'):
            metrics = {}
            for endpoint in ENDPOINTS:
                values = np.asarray([c[direction]['normalized_improvement_vs_unchanged']
                                     [intervention][endpoint] for c in cohorts], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite cohort value')
                lo, hi = np.percentile(values[draws].mean(axis=1),
                                        PROTOCOL['confidence_percentiles'])
                metrics[endpoint] = {
                    'mean': float(values.mean()), 'cohort_values': values.tolist(),
                    'cohort_bootstrap_ci95': [float(lo), float(hi)],
                    'fraction_positive': float(np.mean(values > 0)),
                }
            out[intervention] = metrics
        result['directions'][direction] = out
    primary = result['directions'][PROTOCOL['primary_direction']][
        'clip_pcc_to_training_range'][PROTOCOL['primary_endpoint']]
    result['primary_point_estimate_criterion_met'] = primary['mean'] > 0
    result['primary_bootstrap_interval_positive'] = primary['cohort_bootstrap_ci95'][0] > 0
    return result


def run(directory):
    directory = Path(directory)
    verify(directory)
    solver = LeducCFR()
    reference = experiment.compile_measurements(solver)
    opponents = [experiment.conventional_policy(solver, parameters=p)
                 for p in ((5., -1.5, .5, 0.), (2., 1.5, .8, 0.))]
    records, cohorts = [], []
    all_seeds = set()
    for rep in range(PROTOCOL['replications']):
        group = {'A': [], 'B': []}
        for family in ('A', 'B'):
            for i in range(PROTOCOL['policies_per_family']):
                policy_seed = seed(family, rep, i)
                if policy_seed in all_seeds:
                    raise RuntimeError('duplicate seed')
                all_seeds.add(policy_seed)
                policy = (experiment.sampled_cfr(solver, policy_seed) if family == 'A'
                          else experiment.conventional_policy(solver, policy_seed))
                experiment.validate_policy(solver, policy)
                baseline, candidate = experiment.measure(solver, policy, reference)
                outcomes = [experiment.cross_value(solver, policy, opp) for opp in opponents]
                outcomes.append(experiment.matched_response(solver, policy))
                row = {'id': f'r{rep:02d}-{family}-{i:02d}', 'replication': rep,
                       'family': family, 'seed': policy_seed, 'baseline': baseline.tolist(),
                       'candidate': candidate.tolist(), 'outcomes': outcomes}
                records.append(row)
                group[family].append(row)
        cohorts.append({'A_to_B': evaluate(group['A'], group['B']),
                        'B_to_A': evaluate(group['B'], group['A'])})
        print(f'generated replication {rep+1}/{PROTOCOL["replications"]}', flush=True)
    verify(directory)
    write_exclusive(directory / 'measurements.json', records)
    report = {
        'scope': 'fresh-seed computational input intervention; not causal identification of PCC constructs',
        'protocol_sha256': sha((directory/'protocol.json').read_bytes()),
        'measurements_sha256': sha((directory/'measurements.json').read_bytes()),
        'cohort_results': cohorts, 'summary': summarize(cohorts),
        'frozen_evidence_modified': False,
        'limitations': [
            'Clipping inputs intervenes on model representation, not the game or generating mechanism.',
            'Preselected primary hypothesis motivated by inspected earlier data; not independent preregistration.',
            'Eight cohort bootstrap intervals are descriptive and may be imprecise.',
            'Training-range clipping has no guarantee of fixing joint distribution shift.',
            'No external simulator validation, no multiplicity-adjusted secondary testing.',
        ],
    }
    write_exclusive(directory/'results.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = freeze(args.output)
    elif args.command == 'run':
        report = run(args.output)
        result = {'summary': report['summary'], 'results_file': str(args.output/'results.json')}
    else:
        verify(args.output, require_fresh=False)
        result = {'status': 'protocol_and_source_verified'}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
