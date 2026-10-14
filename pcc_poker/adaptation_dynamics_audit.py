"""Patch 17: read-only, post-hoc seat and learning-checkpoint audit of Patch 16.

No additional tournament runs. Bootstrap units are independent seed replicates.
Quarter/half checkpoints are cumulative; disjoint segment means are recovered
algebraically using the original integer checkpoint lengths.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .adaptation_interaction import paired_interval

BUDGETS = ('500', '2000')
CONDITIONS = ('adaptive', 'frozen')


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    return sum(values) / len(values)


def _summarize(values, *, seed, samples):
    return {'mean': _mean(values), 'interval_95': paired_interval(values, seed=seed, samples=samples),
            'positive_replicate_fraction': sum(x > 0 for x in values) / len(values),
            'seed_values': values}


def _validate(report):
    if report.get('status') != 'completed' or report.get('analysis_status') != 'prospective_balanced_adaptation_benefit_replication':
        raise ValueError('requires completed Patch 16 report')
    design = report.get('design', {})
    seeds = design.get('seeds')
    n = design.get('hands_per_seat')
    if not isinstance(seeds, list) or len(seeds) < 2 or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('invalid seed schedule')
    if type(n) is not int or n < 4 or n // 4 == 0 or n // 2 == n // 4:
        raise ValueError('requires at least four hands per seat for disjoint checkpoints')
    if not design.get('paired_deals') or not design.get('independent_agent_state_by_condition_and_seat'):
        raise ValueError('missing paired-design guarantees')
    results = report.get('results', {})
    if set(results) != set(BUDGETS):
        raise ValueError('unexpected budget set')
    for budget in BUDGETS:
        entry = results[budget]
        if entry.get('role') != ('primary' if budget == '2000' else 'secondary'):
            raise ValueError('inconsistent endpoint role')
        reps = entry.get('replicates', [])
        if [r.get('seed') for r in reps] != seeds:
            raise ValueError('replicate schedule mismatch')
        for rep in reps:
            if set(rep.get('conditions', {})) != set(CONDITIONS):
                raise ValueError('missing condition')
            for label in CONDITIONS:
                c = rep['conditions'][label]
                vals = [c.get('seat0_mean'), c.get('seat1_mean'), c.get('seat_average_payoff')]
                curve = c.get('cumulative_quarter_half_full')
                if not isinstance(curve, list) or len(curve) != 3:
                    raise ValueError('missing cumulative checkpoints')
                if not all(_numeric(v) for v in vals + curve):
                    raise ValueError('non-finite payoff')
                if not math.isclose(vals[2], (vals[0] + vals[1]) / 2, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('inconsistent seat average')
                if not math.isclose(vals[2], curve[2], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('inconsistent full-run checkpoint')
            a, f = rep['conditions']['adaptive'], rep['conditions']['frozen']
            checks = {'paired_adaptation_gain': a['seat_average_payoff'] - f['seat_average_payoff'],
                      'seat0_adaptation_gain': a['seat0_mean'] - f['seat0_mean'],
                      'seat1_adaptation_gain': a['seat1_mean'] - f['seat1_mean']}
            if any(not _numeric(rep.get(k)) or not math.isclose(rep[k], v, rel_tol=1e-9, abs_tol=1e-9)
                   for k, v in checks.items()):
                raise ValueError('inconsistent stored adaptation contrast')
        source = entry.get('full_vs_frozen', {})
        calculated = [r['paired_adaptation_gain'] for r in reps]
        if len(source.get('seed_values', [])) != len(reps) or any(
                not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
                for x, y in zip(source['seed_values'], calculated)):
            raise ValueError('source contrast mismatch')
        if not _numeric(source.get('mean')) or not math.isclose(source['mean'], _mean(calculated), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('source mean mismatch')
    return seeds, n


def audit(report, *, bootstrap_samples=2000, bootstrap_seed=2700):
    if type(bootstrap_samples) is not int or bootstrap_samples < 1 or type(bootstrap_seed) is not int:
        raise ValueError('invalid bootstrap settings')
    seeds, n = _validate(report)
    q, h = max(1, n // 4), max(1, n // 2)
    result = {}
    for budget in BUDGETS:
        reps = report['results'][budget]['replicates']
        gains = [r['paired_adaptation_gain'] for r in reps]
        seat0 = [r['seat0_adaptation_gain'] for r in reps]
        seat1 = [r['seat1_adaptation_gain'] for r in reps]
        quarter, middle, late = [], [], []
        for rep in reps:
            curves = [rep['conditions'][label]['cumulative_quarter_half_full'] for label in CONDITIONS]
            a, f = curves
            delta = [x - y for x, y in zip(a, f)]
            quarter.append(delta[0])
            middle.append((h * delta[1] - q * delta[0]) / (h - q))
            late.append((n * delta[2] - h * delta[1]) / (n - h))
        stats = lambda vals, offset: _summarize(vals, seed=bootstrap_seed + int(budget) * 10 + offset, samples=bootstrap_samples)
        loo = [(seeds[i], _mean(gains[:i] + gains[i + 1:])) for i in range(len(seeds))]
        result[budget] = {
            'role': 'primary' if budget == '2000' else 'secondary',
            'full_vs_frozen': stats(gains, 1),
            'seat0_gain': stats(seat0, 2), 'seat1_gain': stats(seat1, 3),
            'seat0_minus_seat1_gain': stats([x - y for x, y in zip(seat0, seat1)], 4),
            'segments': {'first_quarter': stats(quarter, 5), 'second_quarter': stats(middle, 6),
                         'second_half': stats(late, 7),
                         'second_half_minus_first_quarter': stats([x - y for x, y in zip(late, quarter)], 8)},
            'leave_one_seed_out': {'means_by_excluded_seed': [{'seed': s, 'mean': v} for s, v in loo],
                                   'min_mean': min(v for _, v in loo), 'max_mean': max(v for _, v in loo)},
            'top_abs_gain_seed': seeds[max(range(len(gains)), key=lambda i: abs(gains[i]))],
        }
    return {
        'status': 'completed', 'analysis_status': 'post_hoc_patch16_adaptation_dynamics_audit',
        'source_design_sha256': report.get('design_sha256'),
        'design': {'seeds': seeds, 'hands_per_seat': n, 'checkpoint_hand_counts': [q, h, n],
                   'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
                   'read_only': True, 'post_hoc': True},
        'results': result,
        'limitations': ['Post-hoc descriptive audit; not a second prospective hypothesis test.',
                        'Bootstrap resamples seeds, not dependent hands.',
                        'Seat effects are mean payoff contrasts; no seat-specific learning curves were recorded.',
                        'Segment estimates reconstructed from cumulative paired seat-average checkpoints; no per-hand observations available.',
                        'Second-half vs first-quarter is descriptive; neither segment independently identifies learning causation.',
                        'One fixed CFR training sequence; no cross-game generalization.'],
        'provenance': {'input_read_only': True, 'policy_changes': False, 'solver_changes': False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='validation/prospective-adaptation-benefit.json')
    parser.add_argument('--output', default='validation/adaptation-dynamics-audit.json')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2700)
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        parser.error('refusing to overwrite existing report: ' + str(target))
    report = audit(json.loads(Path(args.input).read_text(encoding='utf-8')),
                   bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'results': {
        k: {'gain': v['full_vs_frozen']['mean'], 'gain_interval': v['full_vs_frozen']['interval_95'],
            'seat_difference': v['seat0_minus_seat1_gain']['mean'],
            'late_minus_early': v['segments']['second_half_minus_first_quarter']['mean'],
            'leave_one_out_range': [v['leave_one_seed_out']['min_mean'], v['leave_one_seed_out']['max_mean']]}
        for k, v in report['results'].items()}, 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
