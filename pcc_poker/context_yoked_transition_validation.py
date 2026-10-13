"""Prospective public-context matching of seat-yoked transition opportunities.

The matching protocol is fixed before inspecting seeds 1501--1505. Features
are fit using the existing training data; evaluation outcomes never select bins.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .analyze import load_jsonl
from .history_structure_probe import MIXED_PAIR_CATEGORIES
from .paired_matched_opportunity_validation import (
    PAIR_INDEX, _pair_examples, _evaluate_representation,
    attach_paired_features, aggregate_seed_results,
)
from .context_matched_transition_validation import make_examples
from .seat_opportunity_decomposition import _stable_transition_key
from .seat_yoked_transition_validation import generate_seat_yoked_evaluation

DEFAULT_SEEDS = (1501, 1502, 1503, 1504, 1505)


def _bin(value: float, edges: tuple[float, ...]) -> int:
    return sum(float(value) > edge for edge in edges)


def public_context_transitions(rows: list[dict]) -> list[dict]:
    """Capture right-decision public state, never private rank or target labels."""
    by_hand = defaultdict(list)
    for row in rows:
        by_hand[row['hand_id']].append(row)
    result = []
    for hand, hand_rows in by_hand.items():
        ordered = sorted(hand_rows, key=lambda row: row['decision_index'])
        for left, right in zip(ordered, ordered[1:]):
            if float(right['to_call']) <= 0:
                continue
            pair = tuple(sorted((left['action'], right['action'])))
            if pair not in MIXED_PAIR_CATEGORIES:
                continue
            result.append({
                'hand_id': hand,
                'left_decision_index': left['decision_index'],
                'right_decision_index': right['decision_index'],
                'round_index': int(right['round_index']),
                'pair': pair,
                'context': (
                    int(right['round_index']),
                    _bin(right['to_call'], (2, 5, 10)),
                    _bin(right['pot'], (4, 10, 20)),
                ),
            })
    return result


def _feature(transitions: list[dict]) -> np.ndarray:
    counts = Counter(t['pair'] for t in transitions)
    n = len(transitions)
    return np.array([counts[p] / n if n else 0.0 for p in MIXED_PAIR_CATEGORIES])


def attach_context_features(examples: list[dict]) -> None:
    """Exact public-context quotas within each mixture; stable label-blind picks."""
    for seat0, seat1 in _pair_examples(examples):
        pools = {}
        for seat, example in ((0, seat0), (1, seat1)):
            by_context = defaultdict(list)
            for transition in public_context_transitions(example['rows']):
                by_context[transition['context']].append(transition)
            pools[seat] = by_context
        keys = sorted(set(pools[0]) | set(pools[1]))
        quotas = {key: min(len(pools[0][key]), len(pools[1][key])) for key in keys}
        for seat, example in ((0, seat0), (1, seat1)):
            selected = []
            for key in keys:
                selected.extend(sorted(pools[seat][key], key=_stable_transition_key)[:quotas[key]])
            block = _feature(selected)
            offset = len(example['M2_features'])
            features = np.concatenate([example['M2_features'], block])
            ablated = features.copy()
            ablated[offset + PAIR_INDEX] = 0.0
            example['context_matched_features'] = features
            example['context_matched_ablated'] = ablated
            eligible = sum(len(v) for v in pools[seat].values())
            matched = len(selected)
            example['context_matching_diagnostics'] = {
                'eligible': eligible,
                'selected': matched,
                'retained_fraction': matched / eligible if eligible else None,
                'common_contexts': sum(q > 0 for q in quotas.values()),
                'matched_context_counts': {str(k): q for k, q in quotas.items() if q},
                'bet_call_share': float(block[PAIR_INDEX]),
            }


def run_context_yoked_transition_validation(training_records: list[dict],
                                             seeds=DEFAULT_SEEDS,
                                             mixtures_per_seed: int = 20,
                                             hands_per_seat: int = 100) -> dict:
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and unique')
    if mixtures_per_seed <= 0 or hands_per_seat <= 0:
        raise ValueError('mixtures_per_seed and hands_per_seat must be positive')
    train = make_examples(training_records)
    if not train:
        raise ValueError('training data contain no examples')
    attach_paired_features(train)
    attach_context_features(train)
    reports = []
    for seed in seeds:
        records, design = generate_seat_yoked_evaluation(
            seed=seed, mixtures=mixtures_per_seed, hands_per_seat=hands_per_seat)
        test = make_examples(records)
        attach_paired_features(test)
        attach_context_features(test)
        ordinary = _evaluate_representation(train, test, 'ordinary')
        count = _evaluate_representation(train, test, 'paired_matched')
        context = _evaluate_representation(train, test, 'context_matched')
        diagnostics = [e['context_matching_diagnostics'] for e in test]
        retained = [d['retained_fraction'] for d in diagnostics if d['retained_fraction'] is not None]
        exact = all(a['context_matching_diagnostics']['matched_context_counts'] ==
                    b['context_matching_diagnostics']['matched_context_counts']
                    for a, b in _pair_examples(test))
        ordinary_gap = ordinary['absolute_seat_gap']
        count_gap = count['absolute_seat_gap']
        attenuation = None if np.isclose(ordinary_gap, 0.0) else 1.0 - count_gap / ordinary_gap
        analysis = {
            'gap_attenuation': attenuation,
            'ordinary': ordinary, 'paired_matched': count, 'context_matched': context,
            'context_diagnostics': {
                'mean_retained_fraction': float(np.mean(retained)) if retained else None,
                'total_selected': sum(d['selected'] for d in diagnostics),
                'exact_context_count_match': exact,
                'zero_common_support_examples': sum(d['selected'] == 0 for d in diagnostics),
            },
        }
        reports.append({'seed': seed, 'design': design, 'analysis': analysis})
    aggregate = aggregate_seed_results(reports)
    gaps = np.array([r['analysis']['context_matched']['absolute_seat_gap'] for r in reports])
    baseline = np.array([r['analysis']['paired_matched']['absolute_seat_gap'] for r in reports])
    differences = baseline - gaps
    aggregate['context_matched'] = {
        'mean_absolute_seat_gap': float(np.mean(gaps)),
        'mean_gap_reduction_vs_count_only': float(np.mean(differences)),
        'positive_reduction_seed_fraction': float(np.mean(differences > 0)),
        'seed_gap_reductions': differences.tolist(),
    }
    return {
        'status': 'completed',
        'analysis_status': 'prospective_public_context_yoked_transition_validation',
        'seeds': list(seeds), 'mixtures_per_seed': mixtures_per_seed,
        'hands_per_seat': hands_per_seat,
        'matching_rule': 'within each mixture: exact round, fixed to_call bins (<=2, <=5, <=10, >10), fixed pot bins (<=4, <=10, <=20, >20); select by stable identifiers',
        'seed_results': reports, 'aggregate': aggregate,
        'provenance': {'human_data_accessed': False,
                       'previous_frozen_artifacts_modified': False,
                       'matching_rule_tuned_after_results': False},
        'limitations': 'Eligibility requires a mixed transition and is action-dependent; matching does not identify causal seat effects. Sparse common support must be reported.',
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', default='outputs/mixed-recovery-data.jsonl')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--mixtures-per-seed', type=int, default=20)
    parser.add_argument('--hands-per-seat', type=int, default=100)
    parser.add_argument('--output', default='validation/context-yoked-transition-validation.json')
    args = parser.parse_args(argv)
    report = run_context_yoked_transition_validation(
        load_jsonl(args.training), args.seeds, args.mixtures_per_seed, args.hands_per_seat)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'aggregate': report['aggregate'],
                      'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
