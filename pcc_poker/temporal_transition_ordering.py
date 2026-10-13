"""Prospective descriptive seat comparison of temporal action-pair orientation.

Condition on public-context and unordered action pair simultaneously, so that
opportunity exposure and action-pair composition cannot alone create a gap.
No human data, no refitting, and no modification of the frozen evidence baseline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .context_yoked_transition_validation import DEFAULT_SEEDS, public_context_transitions
from .context_matched_transition_validation import make_examples
from .paired_matched_opportunity_validation import _pair_examples
from .seat_yoked_transition_validation import generate_seat_yoked_evaluation


def directed_transitions(rows: list[dict]) -> list[dict]:
    """Add chronological direction using only the two observed actions."""
    by_hand = defaultdict(list)
    for row in rows:
        by_hand[row['hand_id']].append(row)
    directions = {}
    for hand_id, hand_rows in by_hand.items():
        ordered = sorted(hand_rows, key=lambda r: r['decision_index'])
        for left, right in zip(ordered, ordered[1:]):
            directions[(hand_id, left['decision_index'], right['decision_index'])] = (
                left['action'], right['action'])
    result = []
    for transition in public_context_transitions(rows):
        key = (transition['hand_id'], transition['left_decision_index'],
               transition['right_decision_index'])
        direction = directions[key]
        # Alphabetical ordering supplies a fixed, seat-independent orientation.
        result.append({**transition, 'forward': int(direction == transition['pair']),
                       'directed_pair': direction})
    return result


def _strata(example: dict) -> dict:
    grouped = defaultdict(list)
    for transition in directed_transitions(example['rows']):
        grouped[(transition['context'], transition['pair'])].append(transition['forward'])
    return grouped


def compare_pair(seat0: dict, seat1: dict, permutations: int = 199,
                 seed: int = 0) -> dict:
    """Evaluate orientation differences conditioned on public context and pair.

    A within-stratum label-exchange permutation is descriptive and assumes
    exchangeability; it does not correct for dependence among hands or players.
    """
    if permutations < 0:
        raise ValueError('permutations must be nonnegative')
    pools = (_strata(seat0), _strata(seat1))
    shared = sorted(set(pools[0]) & set(pools[1]))
    quotas = {k: min(len(pools[0][k]), len(pools[1][k])) for k in shared}
    overlap = sum(quotas.values())
    eligible = [sum(map(len, pool.values())) for pool in pools]
    if overlap == 0:
        return {'overlap': 0, 'eligible_by_seat': eligible, 'common_strata': 0,
                'orientation_gap': None, 'signed_orientation_gap': None,
                'permutation_p_value': None, 'permutations': permutations,
                'unmatched_by_seat': eligible}
    differences = []
    for key in shared:
        a = np.asarray(pools[0][key], dtype=float)
        b = np.asarray(pools[1][key], dtype=float)
        differences.append((quotas[key], float(a.mean() - b.mean()), a, b))
    observed = sum(q * abs(d) for q, d, _, _ in differences) / overlap
    signed = sum(q * d for q, d, _, _ in differences) / overlap
    p_value = None
    if permutations:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(permutations):
            statistic = 0.0
            for q, _, a, b in differences:
                shuffled = rng.permutation(np.concatenate((a, b)))
                delta = float(shuffled[:len(a)].mean() - shuffled[len(a):].mean())
                statistic += q * abs(delta)
            if statistic / overlap >= observed - 1e-12:
                count += 1
        p_value = (count + 1) / (permutations + 1)
    return {'overlap': overlap, 'eligible_by_seat': eligible,
            'common_strata': len(shared), 'orientation_gap': float(observed),
            'signed_orientation_gap': float(signed),
            'permutation_p_value': p_value, 'permutations': permutations,
            'unmatched_by_seat': [n - overlap for n in eligible]}


def summarize(comparisons: list[dict]) -> dict:
    supported = [c for c in comparisons if c['overlap']]
    weight = sum(c['overlap'] for c in supported)
    def weighted(field):
        return (sum(c['overlap'] * c[field] for c in supported) / weight
                if weight else None)
    return {'mixtures': len(comparisons), 'supported_mixtures': len(supported),
            'zero_support_mixtures': len(comparisons) - len(supported),
            'total_overlap': weight, 'overlap_weighted_orientation_gap': weighted('orientation_gap'),
            'overlap_weighted_signed_orientation_gap': weighted('signed_orientation_gap')}


def run_temporal_transition_ordering(seeds=DEFAULT_SEEDS, mixtures_per_seed: int = 20,
                                     hands_per_seat: int = 100,
                                     permutations: int = 199) -> dict:
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('seeds must be nonempty and unique')
    if mixtures_per_seed <= 0 or hands_per_seat <= 0 or permutations < 0:
        raise ValueError('mixtures and hands must be positive; permutations nonnegative')
    reports = []
    for seed in seeds:
        records, design = generate_seat_yoked_evaluation(
            seed=seed, mixtures=mixtures_per_seed, hands_per_seat=hands_per_seat)
        comparisons = [dict(mixture_id=str(a['mixture_id']),
                            **compare_pair(a, b, permutations=permutations,
                                           seed=seed * 1000003 + i))
                       for i, (a, b) in enumerate(_pair_examples(make_examples(records)))]
        reports.append({'seed': seed, 'design': design, 'summary': summarize(comparisons),
                        'mixtures': comparisons})
    supported = [r['summary']['overlap_weighted_orientation_gap'] for r in reports
                 if r['summary']['overlap_weighted_orientation_gap'] is not None]
    return {'status': 'completed', 'analysis_status': 'descriptive_temporal_ordering',
            'seeds': list(seeds), 'mixtures_per_seed': mixtures_per_seed,
            'hands_per_seat': hands_per_seat, 'permutations': permutations,
            'method': 'Within-mixture shared (public context, unordered action pair) strata; overlap-quota weighted orientation disparity; within-stratum orientation-label permutations.',
            'seed_results': reports,
            'aggregate': {'supported_seeds': len(supported),
                          'mean_seed_orientation_gap': float(np.mean(supported)) if supported else None,
                          'total_zero_support_mixtures': sum(r['summary']['zero_support_mixtures'] for r in reports)},
            'provenance': {'human_data_accessed': False,
                           'previous_frozen_artifacts_modified': False},
            'limitations': 'Descriptive analysis of immediate transitions only; not a test of long-range temporal dynamics or causal direction. Eligibility conditions on observed actions. Permutation p-values assume within-stratum exchangeability, ignore cross-hand dependence, and are unadjusted for multiple comparisons. Sparse common support may limit inference.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--mixtures-per-seed', type=int, default=20)
    parser.add_argument('--hands-per-seat', type=int, default=100)
    parser.add_argument('--permutations', type=int, default=199)
    parser.add_argument('--output', default='validation/temporal-transition-ordering.json')
    args = parser.parse_args(argv)
    report = run_temporal_transition_ordering(args.seeds, args.mixtures_per_seed,
                                             args.hands_per_seat, args.permutations)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'aggregate': report['aggregate'],
                      'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
