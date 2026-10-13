"""Patch 07: held-out mixture-simplex error and regularization audit.

Recreates the Patch 06B region-held-out design; NEVER fits on a held-out
region. Reports prediction bias, per-mode error, simplex bins and penalties.
These are synthetic identifiability *diagnostics*, not proof of identification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_region_transfer import build_by_region
from .mixed import _metrics, _ridge_predict
from .mixed_ood import REGIONS
from .policies import MODES

DEFAULT_SEEDS = (1801, 1802, 1803, 1804, 1805)
FEATURES = ('M2', 'M3', 'M2_plus_matched_transitions')
PENALTIES = (0.3, 3.0, 30.0)


def simplex_bin(truth):
    """Fixed truth-only bins; no use of model predictions to determine bins."""
    v = np.asarray(truth, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.any(v < -1e-8) or abs(v.sum()-1) > 1e-6:
        raise ValueError('invalid simplex target')
    if np.min(v) <= 0.05:
        return 'edge_0_05'
    if np.max(v) >= 0.8:
        return 'corner_0_80'
    if np.max(v) - np.min(v) <= 0.15:
        return 'near_center'
    return 'interior_other'


def audit_predictions(truth, prediction):
    a, b = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or b.shape != a.shape or not len(a):
        raise ValueError('predictions and targets must be nonempty Nx3 arrays')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('nonfinite predictions or targets')
    error = b - a
    by_bin = {}
    for name in ('edge_0_05', 'corner_0_80', 'near_center', 'interior_other'):
        inds = [i for i, t in enumerate(a) if simplex_bin(t) == name]
        if inds:
            by_bin[name] = {'n': len(inds), 'mae': float(np.abs(error[inds]).mean())}
        else:
            by_bin[name] = {'n': 0, 'mae': None}
    return {
        'metrics': _metrics(a, b),
        'mean_signed_bias': {k: float(error[:, i].mean()) for i, k in enumerate(MODES)},
        'predicted_mean': {k: float(b[:, i].mean()) for i, k in enumerate(MODES)},
        'actual_mean': {k: float(a[:, i].mean()) for i, k in enumerate(MODES)},
        'mean_l1_simplex_distance': float(np.abs(error).sum(axis=1).mean()),
        'simplex_bins': by_bin,
    }


def audit_fold(training_by_region, test_examples, heldout_region, penalties=PENALTIES):
    if heldout_region not in REGIONS or not test_examples:
        raise ValueError('invalid heldout region or empty evaluation set')
    training = [ex for region in REGIONS if region != heldout_region for ex in training_by_region[region]]
    if not training:
        raise ValueError('empty training set')
    train_ids = {str(ex['mixture_id']) for ex in training}
    test_ids = {str(ex['mixture_id']) for ex in test_examples}
    if train_ids & test_ids:
        raise ValueError('mixture group leakage')
    penalties = tuple(float(p) for p in penalties)
    if not penalties or len(set(penalties)) != len(penalties) or any(not np.isfinite(p) or p <= 0 for p in penalties):
        raise ValueError('penalties must be unique positive finite values')
    truth = np.stack([ex['target'] for ex in test_examples])
    prior = np.stack([ex['target'] for ex in training]).mean(axis=0)
    out = {'train_groups': len(train_ids), 'test_groups': len(test_ids),
           'prior_only': audit_predictions(truth, np.tile(prior, (len(truth), 1))), 'models': {}}
    for feature in FEATURES:
        out['models'][feature] = {}
        for penalty in penalties:
            prediction = _ridge_predict(training, test_examples, feature, penalty=penalty)
            out['models'][feature][str(penalty)] = audit_predictions(truth, prediction)
    return out


def run(train_seed=1800, seeds=DEFAULT_SEEDS, mixtures_per_region=20, hands_per_seat=100, penalties=PENALTIES):
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or int(train_seed) in seeds:
        raise ValueError('evaluation seeds must be unique and disjoint from training')
    if mixtures_per_region < 2 or hands_per_seat < 1:
        raise ValueError('positive simulation size and >=2 mixtures required')
    training = build_by_region(int(train_seed), mixtures_per_region, hands_per_seat)
    results = []
    for seed in seeds:
        eval_by_region = build_by_region(seed, mixtures_per_region, hands_per_seat)
        results.append({'seed': seed, 'regions': {
            region: audit_fold(training, eval_by_region[region], region, penalties)
            for region in REGIONS}})
    summary = {}
    for region in REGIONS:
        rows = [item['regions'][region] for item in results]
        summary[region] = {
            'prior_mean_mae': float(np.mean([r['prior_only']['metrics']['mae'] for r in rows])),
            'models': {feature: {str(float(p)): {
                'mean_mae': float(np.mean([r['models'][feature][str(float(p))]['metrics']['mae'] for r in rows])),
                'seed_mae': [float(r['models'][feature][str(float(p))]['metrics']['mae']) for r in rows],
                'mean_signed_bias': {mode: float(np.mean([r['models'][feature][str(float(p))]['mean_signed_bias'][mode] for r in rows])) for mode in MODES}
            } for p in penalties} for feature in FEATURES}}
    return {'status': 'completed', 'analysis_status': 'synthetic_simplex_identifiability_diagnostics',
            'training_seed': int(train_seed), 'evaluation_seeds': list(seeds),
            'mixtures_per_region': mixtures_per_region, 'hands_per_seat': hands_per_seat,
            'penalties': list(penalties), 'seed_results': results, 'summary': summary,
            'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False,
                           'training_only_standardization': True, 'heldout_region_excluded_from_fit': True,
                           'target_used_as_predictor': False},
            'limitations': ('Diagnostic synthetic recovery only, not structural identifiability. '
                            'Penalties are compared on destination labels for analysis, not selected for deployment. '
                            'Never report the best held-out penalty as an unbiased tuned score. '
                            'No CFR, exploitation estimate, or human evidence. Five evaluation seeds share one training set.')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seed', type=int, default=1800)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--mixtures-per-region', type=int, default=20)
    parser.add_argument('--hands-per-seat', type=int, default=100)
    parser.add_argument('--penalties', nargs='+', type=float, default=list(PENALTIES))
    parser.add_argument('--output', default='validation/simplex-identifiability.json')
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f'refusing to overwrite existing report: {target}')
    report = run(args.train_seed, args.seeds, args.mixtures_per_region, args.hands_per_seat, args.penalties)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'summary': report['summary'], 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
