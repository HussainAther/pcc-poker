"""Prospective descriptive decomposition of seat-specific transition composition.

Uses the fixed public-context bins from the preceding experiment. No models
are refit and no human observations are accessed. A diagnostic, not a causal
estimate or a test of PCC construct validity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .context_yoked_transition_validation import DEFAULT_SEEDS, public_context_transitions
from .context_matched_transition_validation import make_examples
from .history_structure_probe import MIXED_PAIR_CATEGORIES
from .paired_matched_opportunity_validation import PAIR_INDEX, _pair_examples
from .seat_yoked_transition_validation import generate_seat_yoked_evaluation


def _distribution(transitions: list[dict]) -> np.ndarray:
    counts = Counter(t['pair'] for t in transitions)
    return np.asarray([counts[p] / len(transitions) if transitions else 0.0
                       for p in MIXED_PAIR_CATEGORIES], dtype=float)


def compare_pair(seat0: dict, seat1: dict) -> dict:
    """Compare conditional pair distributions only on shared public contexts.

    Context weights are pooled overlap quotas, not seat-specific frequencies.
    With no overlap, summaries are null (not an artificial zero effect).
    """
    pools = []
    for example in (seat0, seat1):
        by_context = defaultdict(list)
        for transition in public_context_transitions(example['rows']):
            by_context[transition['context']].append(transition)
        pools.append(by_context)
    shared = sorted(set(pools[0]) & set(pools[1]))
    quotas = {key: min(len(pools[0][key]), len(pools[1][key])) for key in shared}
    quotas = {key: count for key, count in quotas.items() if count > 0}
    overlap = sum(quotas.values())
    total = [sum(map(len, pool.values())) for pool in pools]
    if not overlap:
        return {'overlap': 0, 'eligible_by_seat': total, 'common_contexts': 0,
                'conditional_tv': None, 'conditional_bet_call_gap': None,
                'conditional_bet_call_signed_gap': None,
                'unmatched_context_counts_by_seat': [total[0], total[1]]}
    # Full within-context empirical proportions; do not select using actions.
    tv = 0.0
    bet_gap = 0.0
    bet_signed = 0.0
    for key, quota in quotas.items():
        d0, d1 = (_distribution(pool[key]) for pool in pools)
        weight = quota / overlap
        tv += weight * 0.5 * float(np.abs(d0 - d1).sum())
        delta = float(d0[PAIR_INDEX] - d1[PAIR_INDEX])
        bet_gap += weight * abs(delta)
        bet_signed += weight * delta
    return {'overlap': overlap, 'eligible_by_seat': total,
            'common_contexts': len(quotas), 'conditional_tv': tv,
            'conditional_bet_call_gap': bet_gap,
            'conditional_bet_call_signed_gap': bet_signed,
            'unmatched_context_counts_by_seat': [n - overlap for n in total]}


def _summarize(pairs: list[dict]) -> dict:
    supported = [p for p in pairs if p['overlap'] > 0]
    total_overlap = sum(p['overlap'] for p in supported)
    def weighted(name):
        return (float(sum(p['overlap'] * p[name] for p in supported) / total_overlap)
                if total_overlap else None)
    return {'mixtures': len(pairs), 'supported_mixtures': len(supported),
            'zero_support_mixtures': len(pairs) - len(supported),
            'total_overlap': total_overlap,
            'overlap_weighted_conditional_tv': weighted('conditional_tv'),
            'overlap_weighted_bet_call_absolute_gap': weighted('conditional_bet_call_gap'),
            'overlap_weighted_bet_call_signed_gap': weighted('conditional_bet_call_signed_gap')}


def run_conditional_action_composition(seeds=DEFAULT_SEEDS,
                                       mixtures_per_seed: int = 20,
                                       hands_per_seat: int = 100) -> dict:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and unique')
    if mixtures_per_seed <= 0 or hands_per_seat <= 0:
        raise ValueError('mixtures_per_seed and hands_per_seat must be positive')
    reports = []
    for seed in seeds:
        records, design = generate_seat_yoked_evaluation(
            seed=seed, mixtures=mixtures_per_seed, hands_per_seat=hands_per_seat)
        comparisons = [dict(mixture_id=str(a['mixture_id']), **compare_pair(a, b))
                       for a, b in _pair_examples(make_examples(records))]
        reports.append({'seed': seed, 'design': design,
                        'summary': _summarize(comparisons), 'mixtures': comparisons})
    supported = [r['summary']['overlap_weighted_conditional_tv'] for r in reports
                 if r['summary']['overlap_weighted_conditional_tv'] is not None]
    return {'status': 'completed', 'analysis_status': 'descriptive_conditional_action_composition',
            'seeds': list(seeds), 'mixtures_per_seed': mixtures_per_seed,
            'hands_per_seat': hands_per_seat,
            'method': 'Within-mixture shared public contexts; overlap quotas weight full empirical conditional action-pair distributions; no action-based subsampling.',
            'seed_results': reports,
            'aggregate': {'supported_seeds': len(supported),
                          'mean_seed_conditional_tv': float(np.mean(supported)) if supported else None,
                          'total_zero_support_mixtures': sum(r['summary']['zero_support_mixtures'] for r in reports)},
            'provenance': {'human_data_accessed': False,
                           'previous_frozen_artifacts_modified': False},
            'limitations': 'Mixed-transition eligibility depends on observed actions; this is a descriptive conditional-composition diagnostic, not causal identification. Sparse common support and finite-sample estimation can bias comparisons.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--mixtures-per-seed', type=int, default=20)
    parser.add_argument('--hands-per-seat', type=int, default=100)
    parser.add_argument('--output', default='validation/conditional-action-composition.json')
    args = parser.parse_args(argv)
    report = run_conditional_action_composition(args.seeds, args.mixtures_per_seed, args.hands_per_seat)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'aggregate': report['aggregate'], 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
