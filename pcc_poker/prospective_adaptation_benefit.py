"""Patch 16: fresh-seed prospective Balanced PCC adaptation-benefit replication.

Primary endpoint: paired seat-averaged payoff(adaptive) - payoff(frozen)
against the 2000-iteration CFR average strategy. One-sided decision: primary
95% *two-sided* percentile replicate-bootstrap lower endpoint > 0.
The design is selected after observing Patch 15, and is prospective only
with respect to the new tournament replicates. No solver/agent tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .adaptive_cfr_ablations import ablation_replicate
from .adaptation_interaction import paired_interval
from .engine import all_deals
from .leduc_cfr import LeducCFR
from .policies import PURE_MIXTURES

PRIMARY_BUDGET = 2000
SECONDARY_BUDGET = 500
DEFAULT_SEEDS = tuple(range(2601, 2631))
TEMPERATURE = 0.35
PRIMARY_ENDPOINT = 'paired_adaptive_minus_frozen_seat_averaged_net_chips_per_hand'


def _valid_seeds(seeds):
    return len(seeds) >= 2 and len(seeds) == len(set(seeds)) and all(
        isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds
    )


def _summary(values, seed, samples):
    lo, hi = paired_interval(values, samples=samples, seed=seed)
    return {'mean': sum(values) / len(values), 'seed_values': values,
            'paired_replicate_interval_95': [lo, hi],
            'positive_replicate_fraction': sum(v > 0 for v in values) / len(values),
            'interval_excludes_zero_positive': lo > 0}


def experiment(*, seeds=DEFAULT_SEEDS, hands_per_seat=200, bootstrap_samples=2000, bootstrap_seed=2600):
    seeds = tuple(seeds)
    if not _valid_seeds(seeds):
        raise ValueError('require at least two unique nonnegative integer seeds')
    if not isinstance(hands_per_seat, int) or isinstance(hands_per_seat, bool) or hands_per_seat < 1:
        raise ValueError('hands_per_seat must be a positive integer')
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be a positive integer')
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError('bootstrap_seed must be an integer')
    if set(seeds) & set(range(2401, 2421)) or set(seeds) & set(range(2201, 2206)):
        raise ValueError('replicate seeds overlap previous Patch 12/14 experiments')
    design = {
        'design_version': 'pcc-poker-patch16-v1',
        'primary_case': 'balanced', 'primary_cfr_iterations': PRIMARY_BUDGET,
        'secondary_cfr_iterations': SECONDARY_BUDGET,
        'primary_endpoint': PRIMARY_ENDPOINT,
        'primary_decision_rule': '95% two-sided percentile paired-replicate bootstrap interval lower bound > 0',
        'primary_alpha_nominal_two_sided': 0.05,
        'secondary_inference': 'descriptive; no multiplicity-adjusted claim',
        'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
        'temperature': TEMPERATURE, 'bootstrap_samples': bootstrap_samples,
        'bootstrap_seed': bootstrap_seed, 'paired_deals': True,
        'independent_agent_state_by_condition_and_seat': True,
        'cfr_fixed_in_each_match': True, 'policy_tuning_on_replication_data': False,
    }
    digest = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    decks = all_deals()
    schedules = {}
    for seed in seeds:
        rng = random.Random(seed)
        schedules[seed] = [decks[rng.randrange(len(decks))] for _ in range(hands_per_seat)]
    solver = LeducCFR()
    trained = 0
    results = {}
    for budget in (SECONDARY_BUDGET, PRIMARY_BUDGET):
        solver.train(budget - trained)
        trained = budget
        strategy = solver.average_strategy()
        reps = []
        for seed in seeds:
            conditions = ablation_replicate(PURE_MIXTURES['balanced'], strategy, schedules[seed], seed, TEMPERATURE)
            # Preserve per-seat and three within-run cumulative checkpoints for future descriptive audits.
            condensed = {}
            for name in ('adaptive', 'frozen'):
                c = conditions[name]
                condensed[name] = {k: c[k] for k in ('seat0_mean', 'seat1_mean', 'seat_average_payoff', 'cumulative_quarter_half_full')}
                if not math.isclose(condensed[name]['seat_average_payoff'],
                                    (condensed[name]['seat0_mean'] + condensed[name]['seat1_mean']) / 2,
                                    rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('inconsistent seat average')
            delta = condensed['adaptive']['seat_average_payoff'] - condensed['frozen']['seat_average_payoff']
            reps.append({'seed': seed, 'conditions': condensed, 'paired_adaptation_gain': delta,
                         'seat0_adaptation_gain': condensed['adaptive']['seat0_mean'] - condensed['frozen']['seat0_mean'],
                         'seat1_adaptation_gain': condensed['adaptive']['seat1_mean'] - condensed['frozen']['seat1_mean']})
        deltas = [r['paired_adaptation_gain'] for r in reps]
        absvals = [r['conditions']['adaptive']['seat_average_payoff'] for r in reps]
        results[str(budget)] = {
            'role': 'primary' if budget == PRIMARY_BUDGET else 'secondary',
            'full_vs_frozen': _summary(deltas, bootstrap_seed + budget + 1, bootstrap_samples),
            'absolute_adaptive_payoff': _summary(absvals, bootstrap_seed + budget + 2, bootstrap_samples),
            'replicates': reps,
        }
    primary = results[str(PRIMARY_BUDGET)]['full_vs_frozen']
    return {
        'status': 'completed', 'analysis_status': 'prospective_balanced_adaptation_benefit_replication',
        'design': design, 'design_sha256': digest, 'results': results,
        'primary_decision': {'rule': design['primary_decision_rule'],
                             'criterion_met': primary['paired_replicate_interval_95'][0] > 0,
                             'mean': primary['mean'],
                             'interval_95': primary['paired_replicate_interval_95']},
        'limitations': 'Fixed single CFR training sequence. Repeated hands are dependent within replicate; replicate is bootstrap unit. Previously selected hypothesis, new replicate seeds. Percentile bootstrap is nominal, not a formal guarantee. Secondary comparisons descriptive. Positive relative gain does not imply positive absolute PCC payoff or equilibrium play.',
        'provenance': {'earlier_reports_read_only': True, 'frozen_evidence_modified': False,
                       'human_data_accessed': False, 'policies_modified': False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--hands-per-seat', type=int, default=200)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2600)
    parser.add_argument('--output', default='validation/prospective-adaptation-benefit.json')
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        parser.error('refusing to overwrite existing report: ' + str(target))
    report = experiment(seeds=args.seeds, hands_per_seat=args.hands_per_seat,
                        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'design_sha256': report['design_sha256'],
                      'primary_decision': report['primary_decision'],
                      'secondary_gain': report['results'][str(SECONDARY_BUDGET)]['full_vs_frozen'],
                      'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
