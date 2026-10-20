"""Patch 29: new-seed, protocol-first PCC transfer replication.

Freeze the protocol BEFORE executing; never alter old evidence. This is an
in-silico replication on the same game/generator families, not external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import blind_discriminant as experiment
from .low_dimensional_pcc_audit import BASELINE_INDICES, ENDPOINTS, ridge_fit_predict, standardize
from .leduc_cfr import LeducCFR

PROTOCOL = {
    'protocol_id': 'pcc-transfer-prospective-v2',
    'design': 'fresh independent policy-generator seeds, same game and generator families',
    'replicates_per_family': 32,
    'family_A_seeds': [1730001 + 131 * i for i in range(32)],
    'family_B_seeds': [1930001 + 131 * i for i in range(32)],
    'baseline_columns': list(BASELINE_INDICES),
    'candidate_columns': [0, 1, 2],
    'ridge_penalty': 10.0,
    'endpoints': list(ENDPOINTS),
    'primary_metric': 'difference in test MSE divided by training outcome variance (baseline minus augmented)',
    'primary_claim': 'positive mean improvement in both directions for every endpoint',
    'uncertainty_method': 'paired test-policy bootstrap; conditional on fitted training cohorts',
    'bootstrap_seed': 29001,
    'bootstrap_repetitions': 2000,
    'interval_percentiles': [2.5, 97.5],
    'selection': 'none: feature sets and regularization fixed before generating policies',
}

SOURCES = ('prospective_pcc_replication.py', 'blind_discriminant.py',
           'low_dimensional_pcc_audit.py', 'leduc_cfr.py', 'engine.py',
           'policies.py', 'value_floor.py', 'behavioral.py')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hashes():
    here = Path(__file__).parent
    return {name: sha((here / name).read_bytes()) for name in SOURCES}


def write_exclusive(path, obj):
    with Path(path).open('x', encoding='utf8') as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write('\n')


def freeze(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {'protocol': PROTOCOL, 'sources': hashes(),
              'status': 'frozen_before_policy_generation'}
    write_exclusive(directory / 'protocol.json', frozen)
    return frozen


def verify(directory):
    directory = Path(directory)
    frozen = json.loads((directory / 'protocol.json').read_text())
    if frozen != {'protocol': PROTOCOL, 'sources': hashes(),
                  'status': 'frozen_before_policy_generation'}:
        raise RuntimeError('protocol or source code changed since freeze')
    if any((directory / name).exists() for name in ('measurements.json', 'results.json')):
        raise FileExistsError('replication has already started; refusing overwrite')
    return frozen


def scores(train, test, seed, repetitions=2000):
    """Fixed training-only scaling; conditional paired test-policy bootstrap."""
    xb = np.asarray([r['baseline'] for r in train], float)[:, BASELINE_INDICES]
    tb = np.asarray([r['baseline'] for r in test], float)[:, BASELINE_INDICES]
    xc = np.asarray([r['candidate'] for r in train], float)
    tc = np.asarray([r['candidate'] for r in test], float)
    y = np.asarray([r['outcomes'] for r in train], float)
    ty = np.asarray([r['outcomes'] for r in test], float)
    if (xb.shape[1], xc.shape[1], y.shape[1]) != (3, 3, 3):
        raise ValueError('unexpected feature or endpoint count')
    if len(train) < 3 or len(test) < 3:
        raise ValueError('too few policies')
    b, bt = standardize(xb, tb)
    c, ct = standardize(xc, tc)
    base = ridge_fit_predict(b, y, bt, PROTOCOL['ridge_penalty'])
    aug = ridge_fit_predict(np.c_[b, c], y, np.c_[bt, ct], PROTOCOL['ridge_penalty'])
    scale = y.var(axis=0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError('degenerate training outcome variance')
    per_policy = ((ty-base)**2-(ty-aug)**2)/scale
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(test), size=(repetitions, len(test)))
    boots = per_policy[draws].mean(axis=1)
    low, high = np.percentile(boots, PROTOCOL['interval_percentiles'], axis=0)
    return {'n_train': len(train), 'n_test': len(test),
            'improvement': dict(zip(ENDPOINTS, per_policy.mean(axis=0).tolist())),
            'conditional_ci95': {k: [float(lo), float(hi)] for k, lo, hi in zip(ENDPOINTS, low, high)},
            'baseline_mse': dict(zip(ENDPOINTS, np.mean((ty-base)**2, axis=0).tolist())),
            'augmented_mse': dict(zip(ENDPOINTS, np.mean((ty-aug)**2, axis=0).tolist()))}


def run(directory):
    directory = Path(directory)
    verify(directory)
    solver = LeducCFR()
    reference = experiment.compile_measurements(solver)
    opponents = [experiment.conventional_policy(solver, parameters=p)
                 for p in ((5., -1.5, .5, 0.), (2., 1.5, .8, 0.))]
    records = []
    for family in ('A', 'B'):
        for i, seed in enumerate(PROTOCOL[f'family_{family}_seeds']):
            policy = (experiment.sampled_cfr(solver, seed) if family == 'A'
                      else experiment.conventional_policy(solver, seed))
            experiment.validate_policy(solver, policy)
            baseline, candidate = experiment.measure(solver, policy, reference)
            outcomes = [experiment.cross_value(solver, policy, opponent) for opponent in opponents]
            outcomes.append(experiment.matched_response(solver, policy))
            records.append({'id': f'{family}-{i:02d}', 'family': family, 'seed': seed,
                            'baseline': baseline.tolist(), 'candidate': candidate.tolist(),
                            'outcomes': outcomes})
        print(f'generated family {family}: {PROTOCOL["replicates_per_family"]} policies', flush=True)
    # Validate again before first artifact write; protocol must not drift.
    verify(directory)
    write_exclusive(directory / 'measurements.json', records)
    groups = {f: [r for r in records if r['family'] == f] for f in ('A', 'B')}
    results = {f'{a}_to_{b}': scores(groups[a], groups[b], PROTOCOL['bootstrap_seed']+i,
                                   PROTOCOL['bootstrap_repetitions'])
               for i, (a, b) in enumerate((('A', 'B'), ('B', 'A')))}
    primary_pass = all(v > 0 for r in results.values() for v in r['improvement'].values())
    report = {'scope': 'new-seed in-silico prospective replication, not independent lab replication',
              'protocol_sha256': sha((directory/'protocol.json').read_bytes()),
              'measurements_sha256': sha((directory/'measurements.json').read_bytes()),
              'results': results, 'primary_point_estimate_criterion_met': primary_pass,
              'uncertainty_limitations': [
                  'CIs condition on training models and do not capture training-policy sampling uncertainty.',
                  'CIs are descriptive, not multiplicity-adjusted or confirmatory significance tests.',
                  'New seeds share simulator, generator families, and measurement definitions with the original.',
                  'The frozen source hash is integrity metadata, not independent preregistration.',
                  'No new parameter or feature selection is permitted based on these results.' ]}
    write_exclusive(directory / 'results.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        result = freeze(args.output)
        print(json.dumps({'status': 'frozen', 'protocol_sha256': sha(canonical(result))}, indent=2))
    else:
        result = run(args.output)
        print(json.dumps({'primary_point_estimate_criterion_met': result['primary_point_estimate_criterion_met'],
                          'results': result['results']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
