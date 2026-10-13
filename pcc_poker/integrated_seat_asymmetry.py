"""Integrate pre-existing seat diagnostics without conflating their estimands.

Inputs must have matching prospectively specified designs. No evidence files are
modified, and the three descriptive measurements are not additive causal effects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .temporal_transition_ordering import compare_pair


SOURCES = ('context', 'composition', 'temporal')
METRICS = {
    'ordinary_seat_gap': ('context', 'analysis', 'ordinary', 'absolute_seat_gap'),
    'count_matched_seat_gap': ('context', 'analysis', 'paired_matched', 'absolute_seat_gap'),
    'context_matched_seat_gap': ('context', 'analysis', 'context_matched', 'absolute_seat_gap'),
    'conditional_action_tv': ('composition', 'summary', 'overlap_weighted_conditional_tv'),
    'conditional_bet_call_gap': ('composition', 'summary', 'overlap_weighted_bet_call_absolute_gap'),
    'temporal_orientation_gap': ('temporal', 'summary', 'overlap_weighted_orientation_gap'),
}


def _get(record, path):
    for key in path:
        record = record[key]
    return record


def _validate(reports):
    if set(reports) != set(SOURCES):
        raise ValueError('all three experiment reports are required')
    seeds = reports['context']['seeds']
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('seeds must be nonempty and unique')
    for name, report in reports.items():
        if report.get('status') != 'completed':
            raise ValueError(f'{name}: experiment is not completed')
        for field in ('seeds', 'mixtures_per_seed', 'hands_per_seat'):
            if report.get(field) != reports['context'].get(field):
                raise ValueError(f'{name}: mismatched {field}')
        observed = [item['seed'] for item in report['seed_results']]
        if observed != seeds:
            raise ValueError(f'{name}: seed_results do not match seed order')
        if report.get('provenance', {}).get('human_data_accessed') is not False:
            raise ValueError(f'{name}: provenance not verified as synthetic-only')
    return seeds


def _interval(values, iterations=2000, seed=20261013):
    """Seed-resampling percentile interval, descriptive (not a formal CI)."""
    if not values:
        return None
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(values, dtype=float), size=(iterations, len(values)), replace=True)
    return [float(x) for x in np.percentile(draws.mean(axis=1), [2.5, 97.5])]


def _example(directions, seat):
    rows = []
    for hand, direction in enumerate(directions):
        for index, action in enumerate(direction):
            rows.append({'hand_id': f'{seat}-{hand}', 'decision_index': index,
                         'action': action, 'round_index': 0,
                         'to_call': 2 if index else 0, 'pot': 6})
    return {'rows': rows}


def orientation_sensitivity():
    """Locked-in unit contrast; confirms sensitivity, not empirical power."""
    left = _example([('bet', 'call')] * 10, 0)
    same = _example([('bet', 'call')] * 10, 1)
    reversed_order = _example([('call', 'bet')] * 10, 1)
    null = compare_pair(left, same, permutations=0)
    positive = compare_pair(left, reversed_order, permutations=0)
    passed = (null['overlap'] == positive['overlap'] == 10
              and null['orientation_gap'] == 0.0
              and positive['orientation_gap'] == 1.0)
    return {'status': 'passed' if passed else 'failed', 'null_gap': null['orientation_gap'],
            'injected_reversal_gap': positive['orientation_gap'],
            'overlap_per_comparison': 10,
            'scope': 'Fully separated two-action ordering contrast; not a power analysis for observed data.'}


def integrate(reports, bootstrap_iterations=2000):
    if bootstrap_iterations <= 0:
        raise ValueError('bootstrap_iterations must be positive')
    seeds = _validate(reports)
    records = []
    for index, seed in enumerate(seeds):
        values = {'seed': seed}
        for metric, (source, *path) in METRICS.items():
            result = _get(reports[source]['seed_results'][index], path)
            values[metric] = None if result is None else float(result)
        values['context_vs_count_gap_change'] = (
            values['count_matched_seat_gap'] - values['context_matched_seat_gap'])
        records.append(values)
    summary = {}
    for metric in (*METRICS, 'context_vs_count_gap_change'):
        valid = [record[metric] for record in records if record[metric] is not None]
        summary[metric] = {
            'supported_seeds': len(valid),
            'mean': float(np.mean(valid)) if valid else None,
            'seed_resampling_interval_95': _interval(valid, bootstrap_iterations),
        }
    sensitivity = orientation_sensitivity()
    return {
        'status': 'completed' if sensitivity['status'] == 'passed' else 'sensitivity_failed',
        'analysis_status': 'descriptive_nonadditive_seat_diagnostics',
        'design': {field: reports['context'][field] for field in
                   ('seeds', 'mixtures_per_seed', 'hands_per_seat')},
        'seed_results': records, 'summary': summary,
        'temporal_sensitivity': sensitivity,
        'support': {
            source: {'zero_support_mixtures': sum(
                item['summary']['zero_support_mixtures'] for item in reports[source]['seed_results'])}
            for source in ('composition', 'temporal')},
        'provenance': {'human_data_accessed': False, 'input_reports_read_only': True,
                       'frozen_evidence_modified': False},
        'limitations': ('Metrics use different denominators and estimands, so they must NOT be summed, '
                        'attributed causal shares, or interpreted as a formal decomposition. '
                        'Seed bootstrap intervals are descriptive and unstable with five seeds. '
                        'Temporal positive control is a maximal reversal contrast and does not '
                        'establish sensitivity to weak or long-range effects.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', default='validation/context-yoked-transition-validation.json')
    parser.add_argument('--composition', default='validation/conditional-action-composition.json')
    parser.add_argument('--temporal', default='validation/temporal-transition-ordering.json')
    parser.add_argument('--bootstrap-iterations', type=int, default=2000)
    parser.add_argument('--output', default='validation/integrated-seat-asymmetry.json')
    args = parser.parse_args(argv)
    reports = {name: json.loads(Path(getattr(args, name)).read_text(encoding='utf-8'))
               for name in SOURCES}
    result = integrate(reports, bootstrap_iterations=args.bootstrap_iterations)
    destination = Path(args.output)
    if destination.resolve() in {Path(getattr(args, name)).resolve() for name in SOURCES}:
        parser.error('output cannot overwrite an input report')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'summary': result['summary'],
                      'temporal_sensitivity': result['temporal_sensitivity'],
                      'output': str(destination)}, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
