"""Patch 32: read-only validation and descriptive synthesis of frozen v3 cohorts.

No policy generation, model selection, or experiment file writes occur here.
Run: python -m pcc_poker.transfer_asymmetry_evidence_audit --experiment validation/pcc-transfer-asymmetry-v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prospective_pcc_replication import sha, scores
from .prospective_transfer_asymmetry import PROTOCOL, SOURCE_NAMES, source_hashes, summarize

DIRECTIONS = ('A_to_B', 'B_to_A')
FIELDS = ('improvement', 'baseline_mse', 'augmented_mse', 'conditional_ci95')


def _same_numbers(expected, observed, label="evidence", tolerance=1e-9):
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            raise ValueError(f'{label}: keys differ')
        for key in expected:
            _same_numbers(expected[key], observed[key], f'{label}.{key}', tolerance)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise ValueError(f'{label}: list shape differs')
        for i, (a, b) in enumerate(zip(expected, observed)):
            _same_numbers(a, b, f'{label}[{i}]', tolerance)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValueError(f'{label}: expected numeric value')
        if not np.isfinite(observed) or not np.isclose(expected, observed, atol=tolerance, rtol=tolerance):
            raise ValueError(f'{label}: numerical mismatch')
    elif expected != observed:
        raise ValueError(f'{label}: value differs')


def audit(directory):
    """Reject edited or inconsistent evidence; never create/modify experiment files."""
    directory = Path(directory)
    paths = {name: directory / f'{name}.json' for name in ('protocol', 'measurements', 'results')}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    data = {name: json.loads(value) for name, value in raw.items()}
    frozen, records, report = data['protocol'], data['measurements'], data['results']
    if frozen != {'protocol': PROTOCOL, 'sources': source_hashes(),
                  'status': 'frozen_before_policy_generation'}:
        raise ValueError('protocol/source commitment mismatch')
    if report.get('protocol_sha256') != sha(raw['protocol']):
        raise ValueError('protocol file hash mismatch')
    if report.get('measurements_sha256') != sha(raw['measurements']):
        raise ValueError('measurements file hash mismatch')
    expected_count = PROTOCOL['replications'] * 2 * PROTOCOL['policies_per_family_per_replication']
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError('unexpected number of measurement records')
    groups = {}
    seen_ids, seen_seeds = set(), set()
    n = PROTOCOL['policies_per_family_per_replication']
    for rep in range(PROTOCOL['replications']):
        groups[rep] = {'A': [], 'B': []}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('invalid measurement record')
        rep, family, seed, identifier = (record.get(k) for k in ('replication', 'family', 'seed', 'id'))
        if type(rep) is not int or rep not in groups or family not in ('A', 'B'):
            raise ValueError('invalid replication or family')
        if type(seed) is not int or not isinstance(identifier, str):
            raise ValueError('invalid policy identifier/seed')
        idx = len(groups[rep][family])
        expected_seed = (PROTOCOL[f'seed_base_{family}'] +
                         rep * PROTOCOL['replication_stride'] + idx * PROTOCOL['seed_stride'])
        if seed != expected_seed or identifier != f'r{rep:02d}-{family}-{idx:02d}':
            raise ValueError('policy order, seed or ID mismatch')
        if identifier in seen_ids or seed in seen_seeds:
            raise ValueError('duplicate policy seed or ID')
        seen_ids.add(identifier)
        seen_seeds.add(seed)
        for key in ('baseline', 'candidate', 'outcomes'):
            arr = np.asarray(record.get(key), dtype=float)
            if arr.ndim != 1 or not np.isfinite(arr).all():
                raise ValueError(f'nonfinite/invalid {key}')
        if len(record['baseline']) <= max(PROTOCOL['baseline_columns']) or len(record['candidate']) != 3 or len(record['outcomes']) != 3:
            raise ValueError('measurement dimensionality mismatch')
        groups[rep][family].append(record)
    for rep, group in groups.items():
        if len(group['A']) != n or len(group['B']) != n:
            raise ValueError(f'cohort {rep}: incomplete policies')
    stored = report.get('cohort_results')
    if not isinstance(stored, list) or len(stored) != PROTOCOL['replications']:
        raise ValueError('missing cohort results')
    recomputed = []
    for rep, group in groups.items():
        one = {
            'A_to_B': scores(group['A'], group['B'], PROTOCOL['bootstrap_seed'] + 2 * rep, 100),
            'B_to_A': scores(group['B'], group['A'], PROTOCOL['bootstrap_seed'] + 2 * rep + 1, 100),
        }
        # Check point estimates independently from stored MSE and training outcome variance.
        for direction, family in (('A_to_B', 'A'), ('B_to_A', 'B')):
            variance = np.var(np.array([r['outcomes'] for r in group[family]], float), axis=0)
            for j, endpoint in enumerate(PROTOCOL['endpoints']):
                implied = (one[direction]['baseline_mse'][endpoint] - one[direction]['augmented_mse'][endpoint]) / variance[j]
                _same_numbers(implied, one[direction]['improvement'][endpoint], f'cohort {rep} {direction} {endpoint}')
        _same_numbers(one, stored[rep], f'cohort[{rep}]')
        recomputed.append(one)
    summary = summarize(recomputed)
    _same_numbers(summary, report.get('summary'), 'summary')
    primary = summary['endpoints'][PROTOCOL['primary_endpoint']]
    return {
        'status': 'verified', 'frozen_evidence_modified': False,
        'protocol_sha256': sha(raw['protocol']),
        'measurements_sha256': sha(raw['measurements']),
        'results_sha256': sha(raw['results']),
        'source_files_checked': list(SOURCE_NAMES),
        'cohorts_verified': len(recomputed), 'policies_verified': len(records),
        'primary_endpoint': PROTOCOL['primary_endpoint'],
        'primary_asymmetry_mean': primary['paired_asymmetry_mean'],
        'primary_cohort_bootstrap_ci95': primary['cohort_bootstrap_ci95'],
        'primary_positive_cohort_fraction': primary['fraction_cohorts_positive'],
        'primary_point_estimate_criterion_met': summary['primary_point_estimate_criterion_met'],
        'primary_interval_excludes_zero_positive': summary['primary_interval_excludes_zero_positive'],
        'descriptive_table': [
            {'endpoint': e, 'mean_A_to_B': row['mean_improvement_A_to_B'],
             'mean_B_to_A': row['mean_improvement_B_to_A'],
             'paired_asymmetry_mean': row['paired_asymmetry_mean'],
             'cohort_bootstrap_ci95': row['cohort_bootstrap_ci95']}
            for e, row in summary['endpoints'].items()],
        'limitations': [
            'Verifier reuses the experiment fitting function; not independent implementation of every model calculation.',
            'No fresh samples or independent external validation.',
            'Eight cohort pairs; bootstrap uncertainty and secondary endpoints are descriptive.',
            'Primary hypothesis was motivated by previous observations, not independently preregistered.',
            'Positive directional asymmetry does not imply PCC improves absolute prediction.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.experiment), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
