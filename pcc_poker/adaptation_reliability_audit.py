"""Read-only post-hoc reliability audit of the Patch 14 prospective report.

This is a secondary analysis, not a preregistered confirmatory test.
The Patch 14 report retains per-replicate condition means but no seat-level or
within-run checkpoints; those unavailable analyses must not be reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .adaptation_interaction import paired_interval

CONDITIONS = ('adaptive', 'frozen', 'opponent_only', 'history_only')
CONTRASTS = {
    'full_vs_frozen': {'adaptive': 1, 'frozen': -1},
    'opponent_only_vs_frozen': {'opponent_only': 1, 'frozen': -1},
    'history_only_vs_frozen': {'history_only': 1, 'frozen': -1},
    'full_vs_opponent_only': {'adaptive': 1, 'opponent_only': -1},
    'full_vs_history_only': {'adaptive': 1, 'history_only': -1},
    'factorial_interaction': {'adaptive': 1, 'opponent_only': -1, 'history_only': -1, 'frozen': 1},
}


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def audit(report, *, bootstrap_samples=2000, bootstrap_seed=2500):
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive integer')
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError('bootstrap_seed must be integer')
    if report.get('analysis_status') != 'prospective_balanced_adaptation_replication':
        raise ValueError('expected Patch 14 prospective report')
    design = report['design']
    canonical = json.dumps(design, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != report['design_sha256']:
        raise ValueError('design fingerprint mismatch')
    seeds = design['seeds']
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError('expected unique replicate seeds')
    budgets = (str(design['secondary_cfr_iterations']), str(design['primary_cfr_iterations']))
    if set(report['results']) != set(budgets):
        raise ValueError('unexpected budget results')
    result = {}
    for budget_index, budget in enumerate(budgets):
        section = report['results'][budget]
        reps = section['replicates']
        if len(reps) != len(seeds) or [r['seed'] for r in reps] != list(seeds):
            raise ValueError('missing or out-of-order replicate seeds')
        vectors = {name: [] for name in CONTRASTS}
        absolute = []
        for rep in reps:
            vals = rep['conditions']
            if set(vals) != set(CONDITIONS):
                raise ValueError('missing or extra adaptation condition')
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in vals.values()):
                raise ValueError('nonfinite or invalid payoff')
            for name, coeffs in CONTRASTS.items():
                vectors[name].append(sum(vals[c] * weight for c, weight in coeffs.items()))
            absolute.append(vals['adaptive'])
            if not _close(vectors['full_vs_frozen'][-1], rep['full_vs_frozen']) or not _close(vectors['factorial_interaction'][-1], rep['primary_interaction']):
                raise ValueError('stored replicate contrast inconsistent with condition payoffs')
        for label, vector in (('interaction', vectors['factorial_interaction']), ('full_vs_frozen', vectors['full_vs_frozen']), ('absolute_adaptive_payoff', absolute)):
            source = section[label]
            if len(source['seed_values']) != len(vector) or any(not _close(a, b) for a, b in zip(vector, source['seed_values'])) or not _close(sum(vector)/len(vector), source['mean']):
                raise ValueError('stored summary inconsistent with replicate values')
        comparisons = {}
        for contrast_index, (name, values) in enumerate(vectors.items()):
            lo, hi = paired_interval(values, samples=bootstrap_samples, seed=bootstrap_seed + budget_index * 100 + contrast_index)
            mean = sum(values) / len(values)
            comparisons[name] = {'mean': mean, 'paired_replicate_interval_95': [lo, hi],
                                 'interval_excludes_zero': lo > 0 or hi < 0,
                                 'positive_replicates': sum(x > 0 for x in values),
                                 'negative_replicates': sum(x < 0 for x in values),
                                 'zero_replicates': sum(x == 0 for x in values),
                                 'seed_values': values}
        result[budget] = {'role': section['role'], 'replicate_count': len(reps), 'contrasts': comparisons,
                          'absolute_adaptive_payoff_mean': sum(absolute)/len(absolute)}
    return {'status': 'completed', 'analysis_status': 'posthoc_patch14_reliability_audit',
            'source_design_sha256': report['design_sha256'],
            'bootstrap': {'samples': bootstrap_samples, 'seed': bootstrap_seed, 'unit': 'independent replicate'},
            'results': result,
            'unavailable_from_source': {'seat_specific_payoffs': 'Patch 14 stores only seat-averaged per-replicate payoffs',
                                        'within_tournament_learning_curves': 'Patch 14 stores no time checkpoints or per-hand returns'},
            'interpretation': 'All contrasts except the original Patch 14 primary interaction are secondary/post-hoc; intervals are descriptive and unadjusted for multiplicity. Sign of point estimate does not establish significance. No new tournament data.',
            'provenance': {'input_read_only': True, 'agents_modified': False, 'simulation_rerun': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='validation/prospective-adaptation-replication.json')
    parser.add_argument('--output', default='validation/adaptation-reliability-audit.json')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2500)
    args = parser.parse_args(argv)
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        parser.error('input and output must differ')
    if target.exists():
        parser.error('refusing to overwrite existing report: ' + str(target))
    result = audit(json.loads(source.read_text(encoding='utf-8')),
                   bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'results': {
        k: {name: {'mean': round(v['mean'], 6), 'interval_95': v['paired_replicate_interval_95']}
            for name, v in entry['contrasts'].items() if name in ('full_vs_frozen', 'factorial_interaction')}
        for k, entry in result['results'].items()}, 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
