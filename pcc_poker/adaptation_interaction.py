"""Post-hoc, paired factorial diagnostics for Patch 12 adaptive CFR ablations.

Read-only analysis: cannot establish causal effects beyond the tested switches,
and no held-out parameter tuning or independent per-hand inference is implied.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

CONDITIONS = ('frozen', 'opponent_only', 'history_only', 'adaptive')
CONTRASTS = {
    'full_vs_frozen': {'adaptive': 1, 'frozen': -1},
    'opponent_at_frozen_history': {'opponent_only': 1, 'frozen': -1},
    'history_at_frozen_opponent': {'history_only': 1, 'frozen': -1},
    'opponent_at_adaptive_history': {'adaptive': 1, 'history_only': -1},
    'history_at_adaptive_opponent': {'adaptive': 1, 'opponent_only': -1},
    'factorial_interaction': {'adaptive': 1, 'opponent_only': -1, 'history_only': -1, 'frozen': 1},
}


def _mean(values):
    return sum(values) / len(values)


def paired_interval(values, *, samples=2000, seed=1, confidence=.95):
    """Percentile bootstrap over *replicate-level paired contrasts*, not hands."""
    if len(values) < 2 or samples < 1 or not 0 < confidence < 1:
        raise ValueError('need >=2 replicates, positive bootstrap samples and valid confidence')
    if not all(math.isfinite(x) for x in values):
        raise ValueError('non-finite values')
    rng = random.Random(seed)
    size = len(values)
    draws = sorted(_mean([values[rng.randrange(size)] for _ in range(size)]) for _ in range(samples))
    alpha = (1 - confidence) / 2
    return [draws[max(0, min(samples - 1, int(alpha * (samples - 1))))],
            draws[max(0, min(samples - 1, int((1 - alpha) * (samples - 1))))]]


def _validate(report):
    if report.get('status') != 'completed' or 'budgets' not in report or 'design' not in report:
        raise ValueError('not a completed Patch 12 report')
    design = report['design']
    seeds = design.get('seeds')
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('invalid replicate seeds')
    if set(report.get('ablations', {})) != set(CONDITIONS):
        raise ValueError('missing factorial conditions')
    expected_switches = {'frozen': [False, False], 'opponent_only': [True, False],
                         'history_only': [False, True], 'adaptive': [True, True]}
    if report['ablations'] != expected_switches:
        raise ValueError('unexpected factorial switch mapping')
    if not design.get('matched_deals_across_conditions') or not design.get('independent_learning_state_per_condition_and_seat'):
        raise ValueError('paired independent design required')
    for budget, record in report['budgets'].items():
        if not budget.isdigit() or int(budget) < 1:
            raise ValueError('invalid CFR budget')
        for case, entry in record.get('cases', {}).items():
            reps = entry.get('replicates', [])
            if [r.get('seed') for r in reps] != seeds:
                raise ValueError('missing, repeated or reordered replicate seeds: ' + case)
            for rep in reps:
                if set(rep.get('conditions', {})) != set(CONDITIONS):
                    raise ValueError('missing condition in replicate')
                for name in CONDITIONS:
                    value = rep['conditions'][name]
                    payoff = value.get('seat_average_payoff')
                    seat0, seat1 = value.get('seat0_mean'), value.get('seat1_mean')
                    curve = value.get('cumulative_quarter_half_full')
                    if not isinstance(curve, list) or len(curve) != 3:
                        raise ValueError('expected three cumulative checkpoints')
                    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [payoff, seat0, seat1, *curve]):
                        raise ValueError('non-finite or missing payoff')
                    if abs(payoff - (seat0 + seat1) / 2) > 1e-9 or abs(payoff - curve[-1]) > 1e-9:
                        raise ValueError('inconsistent mean payoff')
    return seeds


def analyze(report, *, bootstrap_samples=2000, bootstrap_seed=2301):
    seeds = _validate(report)
    if bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be positive')
    output = {}
    for budget, block in report['budgets'].items():
        cases = {}
        for case, entry in block['cases'].items():
            reps = entry['replicates']
            contrasts = {}
            for contrast_index, (name, coefficients) in enumerate(CONTRASTS.items()):
                values = [sum(coeff * rep['conditions'][condition]['seat_average_payoff'] for condition, coeff in coefficients.items()) for rep in reps]
                lo, hi = paired_interval(values, samples=bootstrap_samples, seed=bootstrap_seed + int(budget) * 101 + contrast_index * 17 + sum(map(ord, case)))
                contrasts[name] = {'mean': _mean(values), 'seed_values': values, 'paired_replicate_interval_95': [lo, hi],
                                   'interval_excludes_zero': lo > 0 or hi < 0,
                                   'positive_seed_fraction': sum(v > 0 for v in values) / len(values)}
            curves = {}
            for condition in CONDITIONS:
                per_rep = [r['conditions'][condition]['cumulative_quarter_half_full'] for r in reps]
                mean_curve = [_mean([c[j] for c in per_rep]) for j in range(3)]
                curves[condition] = {'cumulative_quarter_half_full': mean_curve,
                                     'second_half_minus_first_half_mean': 2 * (mean_curve[2] - mean_curve[1]) - mean_curve[1]}
            cases[case] = {'contrasts': contrasts, 'learning_curve_descriptive': curves}
        output[budget] = {'cases': cases}
    return {'status': 'completed', 'analysis_status': 'posthoc_paired_factorial_descriptive',
            'design': {'seeds': seeds, 'budgets': list(report['budgets']), 'bootstrap_samples': bootstrap_samples,
                       'bootstrap_seed': bootstrap_seed, 'original_design': report['design']},
            'budgets': output,
            'limitations': ('Post-hoc contrasts from five independent replicates. Percentile intervals are descriptive and not multiple-comparison adjusted; '
                            'an interval excluding zero is not definitive evidence. Cumulative learning checkpoints overlap. '
                            'Second-half minus first-half uses differences of cumulative means, not independent hand measurements. '
                            'Effects are specific to these frozen CFR opponents, policies and tournament conditions.'),
            'provenance': {'source_report_read_only': True, 'frozen_evidence_modified': False, 'human_data_accessed': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='validation/adaptive-cfr-ablations.json')
    parser.add_argument('--output', default='validation/adaptation-interaction.json')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2301)
    args = parser.parse_args(argv)
    source, output = Path(args.input), Path(args.output)
    if output.resolve() == source.resolve() or output.exists():
        parser.error('refusing to overwrite an input or existing report')
    result = analyze(json.loads(source.read_text(encoding='utf-8')), bootstrap_samples=args.bootstrap_samples,
                     bootstrap_seed=args.bootstrap_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    compact = {budget: {case: {k: round(v['mean'], 5) for k, v in data['contrasts'].items() if k in ('full_vs_frozen', 'factorial_interaction')}
                        for case, data in block['cases'].items()} for budget, block in result['budgets'].items()}
    print(json.dumps({'status': 'completed', 'contrasts': compact, 'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
