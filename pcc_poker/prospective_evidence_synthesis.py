"""Patch 30: read-only integrity verification and descriptive evidence synthesis.

Does not regenerate policies, refit selected models, or change frozen artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import prospective_pcc_replication as p


ENDPOINTS = tuple(p.ENDPOINTS)
DIRECTIONS = (('A_to_B', 'A', 'B'), ('B_to_A', 'B', 'A'))


def _read(path):
    return json.loads(path.read_text(encoding='utf8'))


def _close(actual, expected, label, atol=1e-10):
    if not np.allclose(np.asarray(actual, float), np.asarray(expected, float),
                       atol=atol, rtol=1e-9):
        raise ValueError('reported values disagree with independently recomputed values: ' + label)


def _check_records(rows, frozen):
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError('expected 64 measurement rows')
    seen = set()
    groups = {'A': [], 'B': []}
    for row in rows:
        family = row.get('family')
        if family not in groups or row.get('id') in seen:
            raise ValueError('invalid family or duplicate policy id')
        seen.add(row['id'])
        groups[family].append(row)
    for family in groups:
        cohort = sorted(groups[family], key=lambda r: r['id'])
        seeds = frozen['protocol'][f'family_{family}_seeds']
        if [r['id'] for r in cohort] != [f'{family}-{i:02d}' for i in range(32)]:
            raise ValueError('invalid policy identifiers for family ' + family)
        if [r['seed'] for r in cohort] != seeds:
            raise ValueError('seed/measurement mismatch for family ' + family)
        for row in cohort:
            for key in ('baseline', 'candidate', 'outcomes'):
                a = np.asarray(row[key], float)
                if a.ndim != 1 or not np.isfinite(a).all():
                    raise ValueError('invalid measurement: ' + key)
            if len(row['baseline']) <= max(p.BASELINE_INDICES) or len(row['candidate']) != 3 or len(row['outcomes']) != 3:
                raise ValueError('unexpected measurement dimensions')
        groups[family] = cohort
    return groups


def audit(directory, original=None):
    directory = Path(directory)
    proto_path = directory / 'protocol.json'
    measurements_path = directory / 'measurements.json'
    results_path = directory / 'results.json'
    frozen, rows, reported = map(_read, (proto_path, measurements_path, results_path))
    if frozen != {'protocol': p.PROTOCOL, 'sources': p.hashes(),
                  'status': 'frozen_before_policy_generation'}:
        raise ValueError('protocol content or frozen source commitments do not match installed sources')
    if reported['protocol_sha256'] != p.sha(proto_path.read_bytes()):
        raise ValueError('results protocol file-byte hash mismatch')
    if reported['measurements_sha256'] != p.sha(measurements_path.read_bytes()):
        raise ValueError('results measurement file-byte hash mismatch')
    groups = _check_records(rows, frozen)
    if set(reported['results']) != {x[0] for x in DIRECTIONS}:
        raise ValueError('unexpected transfer directions')
    summaries = {}
    for i, (direction, train, test) in enumerate(DIRECTIONS):
        expected = p.scores(groups[train], groups[test],
                            p.PROTOCOL['bootstrap_seed'] + i,
                            p.PROTOCOL['bootstrap_repetitions'])
        actual = reported['results'][direction]
        if actual['n_train'] != expected['n_train'] or actual['n_test'] != expected['n_test']:
            raise ValueError('reported policy count mismatch: ' + direction)
        for metric in ('improvement', 'baseline_mse', 'augmented_mse', 'conditional_ci95'):
            if set(actual[metric]) != set(ENDPOINTS):
                raise ValueError('endpoint mismatch: ' + metric)
            for endpoint in ENDPOINTS:
                _close(actual[metric][endpoint], expected[metric][endpoint], direction + '/' + metric + '/' + endpoint)
        summaries[direction] = {endpoint: {
            'normalized_improvement': float(actual['improvement'][endpoint]),
            'baseline_mse': float(actual['baseline_mse'][endpoint]),
            'augmented_mse': float(actual['augmented_mse'][endpoint]),
            'conditional_ci95': actual['conditional_ci95'][endpoint],
        } for endpoint in ENDPOINTS}
    recomputed_primary = all(summaries[d][e]['normalized_improvement'] > 0
                             for d, _, _ in DIRECTIONS for e in ENDPOINTS)
    if reported['primary_point_estimate_criterion_met'] is not recomputed_primary:
        raise ValueError('primary criterion mismatch')
    synthesis = {
        'status': 'verified', 'frozen_evidence_modified': False,
        'integrity': {
            'freeze_cli_canonical_sha256': p.sha(p.canonical(frozen)),
            'protocol_file_byte_sha256': p.sha(proto_path.read_bytes()),
            'explanation': 'freeze CLI hashes canonical JSON; run results hash indented protocol.json bytes; unequal hashes are expected',
            'measurements_file_byte_sha256': p.sha(measurements_path.read_bytes()),
            'source_commitments_match_current_code': True,
            'reported_results_recomputed': True,
        },
        'primary_point_estimate_criterion_met': recomputed_primary,
        'prospective': summaries,
        'limitations': [
            'Same simulator and generator families; not external replication.',
            'Bootstrap intervals condition on trained models, omit training-sample uncertainty and multiplicity adjustment.',
            'This post-outcome audit verifies reproducibility, not independent preregistration.',
            'Do not select model settings based on this exploratory synthesis.',
        ],
    }
    if original is not None:
        # Only include original frozen evidence that can be directly verified.
        from .cross_family_failure_audit import audit as original_audit
        prior = original_audit(original)
        synthesis['original_frozen_descriptive'] = {
            direction: {endpoint: {'normalized_improvement': values['normalized_mse_improvement']}
                        for endpoint, values in prior['directions'][direction]['endpoints'].items()}
            for direction, _, _ in DIRECTIONS
        }
        synthesis['comparison_caveat'] = ('Original and prospective studies differ in baseline features and fitting; '
                                          'do not pool or interpret changes as an effect of fresh seeds alone.')
    return synthesis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prospective', type=Path, default=Path('validation/pcc-prospective-v2'))
    parser.add_argument('--original', type=Path, default=None)
    parser.add_argument('--report', type=Path, default=None,
                        help='optional NEW JSON report path (exclusive creation)')
    args = parser.parse_args(argv)
    result = audit(args.prospective, args.original)
    if args.report:
        p.write_exclusive(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
