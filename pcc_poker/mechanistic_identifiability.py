"""Patch 08: synthetic observational distinguishability and solver-readiness audit.

Pairwise distances are diagnostics, not proofs of structural identification.
Nearest-neighbor transfer is a nonlinear reference, not a trained oracle or CFR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .cross_region_transfer import build_by_region
from .mixed import _metrics, _ridge_predict, _project_simplex
from .mixed_ood import REGIONS
from .engine import all_deals, initial_state, apply_action, utility


def pairwise_audit(examples, feature='M3', min_target_distance=0.2, near_feature_quantile=0.1):
    """Compare within-seat mixture separations; fixed labels used for audit only."""
    if not (0 < near_feature_quantile <= 1) or min_target_distance <= 0:
        raise ValueError('invalid pair thresholds')
    x = np.stack([np.asarray(e[feature], dtype=float) for e in examples])
    y = np.stack([np.asarray(e['target'], dtype=float) for e in examples])
    seats = [e['focal_seat'] for e in examples]
    groups = [str(e['mixture_id']) for e in examples]
    if not np.all(np.isfinite(x)) or len(examples) < 3:
        raise ValueError('need finite features from at least three examples')
    distances = []
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            if seats[i] != seats[j] or groups[i] == groups[j]:
                continue
            distances.append((float(np.linalg.norm(x[i]-x[j])), float(np.linalg.norm(y[i]-y[j]))))
    if not distances:
        raise ValueError('no independent within-seat mixture comparisons')
    distances.sort(key=lambda pair: pair[0])
    n = max(1, int(np.ceil(len(distances) * near_feature_quantile)))
    selected = distances[:n]
    return {'comparisons': len(distances), 'closest_feature_pairs': n,
            'feature_distance_threshold': distances[n-1][0],
            'closest_pairs_mean_target_l2': float(np.mean([p[1] for p in selected])),
            'closest_pairs_target_far_fraction': float(np.mean([p[1] >= min_target_distance for p in selected])),
            'note': 'Unscaled Euclidean distances; descriptive, sensitive to feature parameterization and finite-simulation noise.'}


def knn_predict(train, test, feature='M3', k=5):
    """Source-fitted standardized kNN; never uses evaluation targets."""
    if not isinstance(k, int) or k < 1 or k > len(train):
        raise ValueError('k must be between 1 and training example count')
    x = np.stack([e[feature] for e in train]).astype(float)
    z = np.stack([e[feature] for e in test]).astype(float)
    y = np.stack([e['target'] for e in train]).astype(float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
        raise ValueError('nonfinite feature')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x, z = (x-mean)/scale, (z-mean)/scale
    squared = ((z[:, None, :] - x[None, :, :])**2).sum(axis=2)
    nearest = np.argsort(squared, axis=1, kind='stable')[:, :k]
    return _project_simplex(y[nearest].mean(axis=1))


def cfr_readiness():
    """Verify game-tree properties, but do NOT claim a CFR implementation."""
    deals = all_deals()
    infos = {}
    nodes = leaves = 0
    for deck in deals:
        stack = [initial_state(deck)]
        while stack:
            state = stack.pop()
            nodes += 1
            if state.terminal:
                leaves += 1
                if abs(utility(state, 0) + utility(state, 1)) > 1e-9:
                    raise ValueError('non-zero-sum terminal utility')
                continue
            key = (state.actor, state.information_key(state.actor))
            legal = tuple(state.legal_actions())
            if key in infos and infos[key] != legal:
                raise ValueError('information set has inconsistent legal actions')
            infos[key] = legal
            stack.extend(apply_action(state, action) for action in legal)
    return {'game': 'two-player heads-up limit Leduc', 'chance_deals': len(deals),
            'traversed_nodes_including_chance_deal_repeats': nodes, 'terminal_nodes': leaves,
            'unique_player_information_sets': len(infos), 'consistent_information_set_legal_actions': True,
            'zero_sum_terminal_utilities': True, 'solver_status': 'not_implemented',
            'required_next_step': 'Implement and separately test information-set CFR, average strategies, best responses and exploitability.'}


def evaluate(training, test, k=5):
    source = [e for e in training]
    if {str(e['mixture_id']) for e in source} & {str(e['mixture_id']) for e in test}:
        raise ValueError('training and evaluation mixture groups overlap')
    truth = np.stack([e['target'] for e in test])
    return {'ridge_M3': _metrics(truth, _ridge_predict(source, test, 'M3', penalty=3.0)),
            'knn_M3': _metrics(truth, knn_predict(source, test, 'M3', k=k))}


def run(train_seed=1800, seeds=(1901, 1902, 1903, 1904, 1905), mixtures_per_region=20, hands_per_seat=100, k=5):
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or int(train_seed) in seeds:
        raise ValueError('unique disjoint evaluation seeds required')
    if mixtures_per_region < 2 or hands_per_seat < 1:
        raise ValueError('invalid simulation size')
    training = build_by_region(train_seed, mixtures_per_region, hands_per_seat)
    results = []
    for seed in seeds:
        testing = build_by_region(seed, mixtures_per_region, hands_per_seat)
        regions = {}
        for region in REGIONS:
            source = [e for r in REGIONS if r != region for e in training[r]]
            regions[region] = evaluate(source, testing[region], k=k)
            regions[region]['pairwise_M3'] = pairwise_audit(testing[region])
        results.append({'seed': seed, 'regions': regions})
    summary = {region: {model: {'mean_mae': float(np.mean([row['regions'][region][model]['mae'] for row in results])),
                                'seed_mae': [row['regions'][region][model]['mae'] for row in results]}
                        for model in ('ridge_M3', 'knn_M3')} for region in REGIONS}
    return {'status': 'completed', 'analysis_status': 'synthetic_observational_distinguishability',
            'train_seed': train_seed, 'evaluation_seeds': list(seeds), 'mixtures_per_region': mixtures_per_region,
            'hands_per_seat': hands_per_seat, 'knn_k': k, 'summary': summary, 'seed_results': results,
            'cfr_readiness': cfr_readiness(),
            'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False,
                           'heldout_regions_excluded_from_training': True, 'evaluation_targets_not_used_for_fitting': True},
            'limitations': 'Observational pair distances are not structural identifiability proofs. kNN is not an oracle. No CFR, regret, exploitability, or human validation. One fixed training seed.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-seed', type=int, default=1800)
    p.add_argument('--seeds', nargs='+', type=int, default=[1901,1902,1903,1904,1905])
    p.add_argument('--mixtures-per-region', type=int, default=20)
    p.add_argument('--hands-per-seat', type=int, default=100)
    p.add_argument('--knn-k', type=int, default=5)
    p.add_argument('--output', default='validation/mechanistic-identifiability.json')
    a = p.parse_args(argv)
    dest = Path(a.output)
    if dest.exists():
        raise FileExistsError('refusing to overwrite existing report: ' + str(dest))
    report = run(a.train_seed, a.seeds, a.mixtures_per_region, a.hands_per_seat, a.knn_k)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'summary': report['summary'], 'cfr_readiness': report['cfr_readiness'], 'output': str(dest)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
