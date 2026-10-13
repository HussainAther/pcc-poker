"""Prospective five-region synthetic mixture recovery, extending Patch 05.

Fixed training dataset, new simulation seeds, and explicit region-wise metrics.
No CFR/exploitability or human validity claims are made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import load_jsonl
from .context_matched_transition_validation import make_examples
from .game_theoretic_baselines import MODEL_KEYS, attach_baselines, _evaluate
from .mixed_ood import REGIONS, generate_mixed_ood_dataset
from .policies import MODES

DEFAULT_SEEDS = (1701, 1702, 1703, 1704, 1705)


def region_metrics(examples: list[dict], models: dict) -> dict:
    """Validate truth distribution and return per-region per-model scoring."""
    if not examples:
        raise ValueError('region must contain evaluation examples')
    weights = np.stack([e['target'] for e in examples])
    return {
        'examples': len(examples),
        'target_mean': {mode: float(weights[:, i].mean()) for i, mode in enumerate(MODES)},
        'models': models,
    }


def summarize(seed_results: list[dict]) -> dict:
    output = {}
    for region in REGIONS:
        results = [s['regions'][region] for s in seed_results]
        by_model = {}
        for model in MODEL_KEYS:
            maes = [r['models'][model]['mae'] for r in results]
            by_model[model] = {
                'mean_mae': float(np.mean(maes)),
                'seed_mae': [float(x) for x in maes],
                'mean_per_mode_mae': {
                    mode: float(np.mean([r['models'][model]['per_mode_mae'][mode] for r in results]))
                    for mode in MODES
                },
                'mean_dominant_mode_accuracy': float(np.mean([
                    r['models'][model]['dominant_mode_accuracy'] for r in results
                ])),
            }
        difference = [r['models']['M2']['mae'] - r['models']['M2_plus_matched_transitions']['mae'] for r in results]
        output[region] = {
            'models': by_model,
            'matched_transition_vs_M2': {
                'mean_mae_reduction': float(np.mean(difference)),
                'seed_mae_reductions': [float(x) for x in difference],
                'positive_seed_fraction': float(np.mean(np.asarray(difference) > 0)),
            },
        }
    return output


def run(training_records: list[dict], seeds=DEFAULT_SEEDS, mixtures_per_region=20, hands_per_seat=100) -> dict:
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('seeds must be nonempty and unique')
    if mixtures_per_region < 1 or hands_per_seat < 1:
        raise ValueError('mixtures_per_region and hands_per_seat must be positive')
    train = make_examples(training_records)
    if not train:
        raise ValueError('training examples are empty')
    attach_baselines(train)
    seed_results = []
    for seed in seeds:
        records, design = generate_mixed_ood_dataset(
            mixtures_per_region=mixtures_per_region, hands_per_seat=hands_per_seat, seed=seed)
        test = make_examples(records)
        attach_baselines(test)
        region_results = {}
        for region in REGIONS:
            region_ids = {g['mixture_id'] for g in design['groups'] if g['ood_region'] == region}
            selected = [ex for ex in test if ex['mixture_id'] in region_ids]
            if len({ex['mixture_id'] for ex in selected}) != mixtures_per_region:
                raise ValueError(f'incomplete region mixture coverage: {region}')
            region_results[region] = region_metrics(selected, _evaluate(train, selected))
        seed_results.append({'seed': seed, 'regions': region_results,
                             'mixtures_per_region': mixtures_per_region})
    return {
        'status': 'completed', 'analysis_status': 'synthetic_mixture_space_generalization',
        'seeds': list(seeds), 'regions': list(REGIONS),
        'mixtures_per_region': mixtures_per_region, 'hands_per_seat': hands_per_seat,
        'seed_results': seed_results, 'summary': summarize(seed_results),
        'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False,
                       'private_rank_used_as_predictor': False, 'training_is_fixed': True},
        'limitations': ('Simulation-based synthetic mixture recovery only. Training distribution may not cover test regions. '
                        'Region results share a simulator and fixed training data; seeds do not imply independent populations. '
                        'No CFR, regret, exploitability, or causal identification. Dominant-mode accuracy can be misleading '
                        'within heavy regions. Matched-transition selection depends on actions.'),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', default='outputs/mixed-recovery-data.jsonl')
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    p.add_argument('--mixtures-per-region', type=int, default=20)
    p.add_argument('--hands-per-seat', type=int, default=100)
    p.add_argument('--output', default='validation/mixture-space-generalization.json')
    args = p.parse_args(argv)
    target = Path(args.output)
    if target.resolve() == Path(args.training).resolve():
        raise ValueError('output must not overwrite training data')
    if target.exists():
        raise FileExistsError(f'refusing to overwrite existing report: {target}')
    report = run(load_jsonl(args.training), args.seeds, args.mixtures_per_region, args.hands_per_seat)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
