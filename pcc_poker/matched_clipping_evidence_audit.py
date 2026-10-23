"""Patch 35: read-only integrity and matched-control evidence synthesis.

Usage: python -m pcc_poker.matched_clipping_evidence_audit --experiment validation/pcc-matched-clipping-v5
This never changes the frozen experiment or reruns policy generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .prospective_pcc_replication import sha
from .prospective_matched_clipping import (
    PROTOCOL, SOURCE_NAMES, MODELS, COMPARISONS, PRIMARY,
    frozen_object, seed, evaluate, summarize,
)

DIRECTIONS = ('A_to_B', 'B_to_A')
ENDPOINTS = tuple(PROTOCOL['endpoints'])


def equal(expected, found, label='value', tol=1e-9):
    """Reject nonfinite, missing and inconsistent stored experiment values."""
    if isinstance(expected, dict):
        if not isinstance(found, dict) or set(expected) != set(found):
            raise ValueError(f'{label}: keys differ')
        for key, value in expected.items():
            equal(value, found[key], f'{label}.{key}', tol)
    elif isinstance(expected, list):
        if not isinstance(found, list) or len(expected) != len(found):
            raise ValueError(f'{label}: list shape differs')
        for i, (a, b) in enumerate(zip(expected, found)):
            equal(a, b, f'{label}[{i}]', tol)
    elif isinstance(expected, bool):
        if not isinstance(found, bool) or expected != found:
            raise ValueError(f'{label}: boolean mismatch')
    elif isinstance(expected, (int, float)):
        if isinstance(found, bool) or not isinstance(found, (int, float)):
            raise ValueError(f'{label}: numeric type mismatch')
        if not np.isfinite(found) or not np.isclose(expected, found, rtol=tol, atol=tol):
            raise ValueError(f'{label}: numerical mismatch')
    elif expected != found:
        raise ValueError(f'{label}: value mismatch')


def audit(directory):
    directory = Path(directory)
    paths = {key: directory / (key + '.json') for key in ('protocol', 'measurements', 'results')}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    frozen, records, report = (json.loads(raw[k]) for k in ('protocol', 'measurements', 'results'))
    if frozen != frozen_object():
        raise ValueError('frozen protocol / source commitments mismatch')
    if report.get('protocol_sha256') != sha(raw['protocol']):
        raise ValueError('protocol hash mismatch')
    if report.get('measurements_sha256') != sha(raw['measurements']):
        raise ValueError('measurement hash mismatch')
    count = PROTOCOL['replications'] * 2 * PROTOCOL['policies_per_family']
    if not isinstance(records, list) or len(records) != count:
        raise ValueError('unexpected measurement record count')
    grouped = {r: {'A': [], 'B': []} for r in range(PROTOCOL['replications'])}
    ids, seeds = set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('invalid record')
        rep, family, ident, policy_seed = (record.get(k) for k in ('replication','family','id','seed'))
        if type(rep) is not int or rep not in grouped or family not in ('A','B'):
            raise ValueError('invalid replication or family')
        index = len(grouped[rep][family])
        if index >= PROTOCOL['policies_per_family'] or type(policy_seed) is not int:
            raise ValueError('invalid policy count or seed type')
        if ident != f'r{rep:02d}-{family}-{index:02d}' or policy_seed != seed(family,rep,index):
            raise ValueError('unexpected policy id, order or seed')
        if ident in ids or policy_seed in seeds:
            raise ValueError('duplicate policy id or seed')
        ids.add(ident); seeds.add(policy_seed)
        for field, minimum in (('baseline',max(PROTOCOL['baseline_columns'])+1),
                               ('candidate',max(PROTOCOL['candidate_columns'])+1),('outcomes',len(ENDPOINTS))):
            arr = np.asarray(record.get(field), dtype=float)
            if arr.ndim != 1 or len(arr) < minimum or not np.isfinite(arr).all():
                raise ValueError(f'invalid {field}')
        if len(record['outcomes']) != len(ENDPOINTS):
            raise ValueError('unexpected endpoint count')
        grouped[rep][family].append(record)
    if any(len(g[f]) != PROTOCOL['policies_per_family'] for g in grouped.values() for f in ('A','B')):
        raise ValueError('incomplete family cohort')
    stored = report.get('cohort_results')
    if not isinstance(stored,list) or len(stored) != PROTOCOL['replications']:
        raise ValueError('incorrect cohort results count')
    computed=[]
    for rep in range(PROTOCOL['replications']):
        group=grouped[rep]
        result={'A_to_B':evaluate(group['A'],group['B']),
                'B_to_A':evaluate(group['B'],group['A'])}
        # Separately check score arithmetic against independently stored MSE values.
        for direction, family in (('A_to_B','A'),('B_to_A','B')):
            train_y=np.array([x['outcomes'] for x in group[family]],float)
            variance=train_y.var(axis=0)
            if np.any(variance<=1e-12):
                raise ValueError('degenerate training variance')
            for comparison,(left,right) in COMPARISONS.items():
                for j,endpoint in enumerate(ENDPOINTS):
                    implied=(result[direction]['mse'][left][endpoint] - result[direction]['mse'][right][endpoint])/variance[j]
                    equal(float(implied),result[direction]['comparisons'][comparison][endpoint],
                          f'{rep}.{direction}.{comparison}.{endpoint}')
        equal(result, stored[rep], f'cohort[{rep}]')
        computed.append(result)
    summary=summarize(computed)
    equal(summary,report.get('summary'),'summary')
    primary=summary['directions'][PROTOCOL['primary_direction']][PRIMARY][PROTOCOL['primary_endpoint']]
    def row(direction,comparison,endpoint):
        return summary['directions'][direction][comparison][endpoint]
    table=[{'direction':direction,'endpoint':endpoint,
            'raw_incremental_mean':row(direction,'raw_incremental',endpoint)['mean'],
            'matched_both_incremental_mean':row(direction,PRIMARY,endpoint)['mean'],
            'matched_both_interval':row(direction,PRIMARY,endpoint)['cohort_bootstrap_ci95'],
            'baseline_clipping_effect_mean':row(direction,'baseline_clipping_effect',endpoint)['mean'],
            'pcc_clipping_effect_mean':row(direction,'pcc_clipping_effect',endpoint)['mean']}
           for direction in DIRECTIONS for endpoint in ENDPOINTS]
    return {
      'status':'verified','frozen_evidence_modified':False,
      'protocol_sha256':sha(raw['protocol']), 'measurements_sha256':sha(raw['measurements']),
      'results_sha256':sha(raw['results']), 'sources_checked':list(SOURCE_NAMES),
      'cohorts_verified':len(computed),'policies_verified':len(records),
      'primary_endpoint':PROTOCOL['primary_endpoint'],
      'primary_matched_incremental_mean':primary['mean'],
      'primary_cohort_bootstrap_ci95':primary['cohort_bootstrap_ci95'],
      'primary_positive_cohort_fraction':primary['fraction_positive'],
      'primary_criterion_met':summary['primary_point_estimate_criterion_met'],
      'primary_interval_positive':summary['primary_bootstrap_interval_positive'],
      'descriptive_table':table,
      'interpretation':('A positive clipping effect versus an unchanged model does not establish '
                        'incremental PCC value versus an equally clipped behavioral baseline.'),
      'limitations':[
        'Verifier shares experimental evaluator and summary code; not an independent fitting implementation.',
        'Only eight simulated cohort pairs; cohort bootstrap is descriptive.',
        'Protocol frozen locally after preceding results motivated the hypothesis.',
        'No external simulator validation, causal identification, or multiple-testing correction.',
        'A nonpositive primary result does not prove equivalence or absence of PCC information.',
      ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(audit(args.experiment),indent=2,allow_nan=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
