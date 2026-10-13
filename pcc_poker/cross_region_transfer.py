"""Region-held-out synthetic transfer audit (Patch 06B).

Train on four mixture regions using one simulation seed; test on the
fifth region using disjoint future seeds. No destination-region fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .context_matched_transition_validation import make_examples
from .game_theoretic_baselines import MODEL_KEYS, attach_baselines, _evaluate
from .mixed import _metrics
from .mixed_ood import REGIONS, generate_mixed_ood_dataset
from .policies import MODES

DEFAULT_SEEDS = (1801, 1802, 1803, 1804, 1805)
DEFAULT_TRAIN_SEED = 1800


def build_by_region(seed: int, mixtures_per_region: int, hands_per_seat: int) -> dict:
    records, design = generate_mixed_ood_dataset(
        mixtures_per_region=mixtures_per_region, hands_per_seat=hands_per_seat, seed=seed)
    # Simulator IDs repeat across seeds: namespace before group splitting.
    for row in records:
        row['mixture_id'] = f"seed-{seed}:" + str(row['mixture_id'])
    examples = make_examples(records)
    attach_baselines(examples)
    output = {}
    for region in REGIONS:
        ids = {f"seed-{seed}:" + str(g['mixture_id']) for g in design['groups']
               if g['ood_region'] == region}
        subset = [example for example in examples if example['mixture_id'] in ids]
        if len({ex['mixture_id'] for ex in subset}) != mixtures_per_region or len(subset) != 2 * mixtures_per_region:
            raise ValueError(f'incomplete simulation group coverage: {region}')
        output[region] = subset
    return output


def evaluate_fold(training_by_region: dict, test_examples: list[dict], heldout_region: str) -> dict:
    if heldout_region not in REGIONS:
        raise ValueError('invalid held-out region')
    if not test_examples:
        raise ValueError('empty evaluation set')
    train = [example for region in REGIONS if region != heldout_region
             for example in training_by_region[region]]
    if not train:
        raise ValueError('no training examples')
    train_ids = {ex['mixture_id'] for ex in train}
    test_ids = {ex['mixture_id'] for ex in test_examples}
    if train_ids & test_ids:
        raise ValueError('mixture ID leakage across training and testing')
    # The only constant predictor uses the source-training mixture prior.
    prior = np.mean(np.stack([e['target'] for e in train]), axis=0)
    truth = np.stack([e['target'] for e in test_examples])
    reference = _metrics(truth, np.tile(prior, (len(test_examples), 1)))
    models = _evaluate(train, test_examples)
    return {'training_regions': [r for r in REGIONS if r != heldout_region],
            'training_groups': len(train_ids), 'evaluation_groups': len(test_ids),
            'training_prior': {m: float(prior[i]) for i, m in enumerate(MODES)},
            'prior_only': reference, 'models': models}


def summarize(results: list[dict]) -> dict:
    result = {}
    for region in REGIONS:
        folds = [row['regions'][region] for row in results]
        metrics = {}
        for name in ('prior_only',) + MODEL_KEYS:
            maes = [f[name]['mae'] if name == 'prior_only' else f['models'][name]['mae'] for f in folds]
            metrics[name] = {'mean_mae': float(np.mean(maes)), 'seed_mae': [float(v) for v in maes]}
        gains = [f['models']['M2']['mae'] - f['models']['M2_plus_matched_transitions']['mae'] for f in folds]
        result[region] = {'models': metrics, 'transition_vs_M2':
                          {'mean_mae_reduction': float(np.mean(gains)),
                           'positive_seed_fraction': float(np.mean(np.asarray(gains) > 0))}}
    return result


def run(train_seed=DEFAULT_TRAIN_SEED, seeds=DEFAULT_SEEDS, mixtures_per_region=20, hands_per_seat=100):
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(seeds) != len(set(seeds)) or int(train_seed) in seeds:
        raise ValueError('evaluation seeds must be unique, nonempty and disjoint from training seed')
    if mixtures_per_region < 2 or hands_per_seat < 1:
        raise ValueError('at least 2 mixtures per region and positive hands required')
    training = build_by_region(int(train_seed), mixtures_per_region, hands_per_seat)
    results = []
    for seed in seeds:
        evaluation = build_by_region(seed, mixtures_per_region, hands_per_seat)
        results.append({'seed': seed,
                        'regions': {region: evaluate_fold(training, evaluation[region], region)
                                    for region in REGIONS}})
    return {'status': 'completed', 'analysis_status': 'synthetic_region_held_out_transfer',
            'training_seed': int(train_seed), 'evaluation_seeds': list(seeds),
            'mixtures_per_region': mixtures_per_region, 'hands_per_seat': hands_per_seat,
            'seed_results': results, 'summary': summarize(results),
            'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False,
                           'region_excluded_from_training': True, 'train_only_standardization': True,
                           'disjoint_simulation_seeds': True, 'private_rank_used_as_predictor': False},
            'limitations': ('Synthetic simulator only. Source-region training and target-region testing are distribution-shift experiments, '
                            'not evidence of real-human construct validity or CFR/exploitability. '
                            'The training set is reused across evaluation seeds; uncertainty across seeds is not population inference.')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seed', type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--mixtures-per-region', type=int, default=20)
    parser.add_argument('--hands-per-seat', type=int, default=100)
    parser.add_argument('--output', default='validation/cross-region-transfer.json')
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f'refusing to overwrite existing report: {target}')
    report = run(args.train_seed, args.seeds, args.mixtures_per_region, args.hands_per_seat)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
