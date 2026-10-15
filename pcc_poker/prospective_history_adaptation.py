"""Patch 22: prospective history-only adaptation-benefit replication.

Fresh seeds, one precommitted primary endpoint, unmodified PCC/CFR policies.
The hypothesis was selected after the exploratory Patch 21 study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .adaptation_interaction import paired_interval
from .adaptive_cfr_ablations import ABLATIONS, ablation_replicate
from .engine import all_deals
from .leduc_cfr import LeducCFR
from .policies import PURE_MIXTURES

SEEDS = tuple(range(3601, 3631))
HANDS_PER_SEAT = 200
CFR_BUDGET = 2000
TEMPERATURE = 0.35
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 3600
PRIMARY_ARMS = ('history_only', 'frozen')
SECONDARY_CONTRASTS = {
    'full_vs_frozen': ('adaptive', 'frozen'),
    'opponent_only_vs_frozen': ('opponent_only', 'frozen'),
    'full_vs_history_only': ('adaptive', 'history_only'),
}
PREVIOUS_RANGES = ((1501, 1505), (1601, 1605), (1701, 1705),
                   (1800, 1805), (1901, 1905), (2101, 2105),
                   (2201, 2205), (2401, 2420), (2601, 2630),
                   (2801, 2830), (3001, 3030), (3201, 3230),
                   (3401, 3430))


def validate(seeds, hands_per_seat, bootstrap_samples, bootstrap_seed):
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds) or
            any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('need at least two distinct nonnegative integer seeds')
    if any(lo <= s <= hi for s in seeds for lo, hi in PREVIOUS_RANGES):
        raise ValueError('seeds overlap prior experimental schedules')
    if type(hands_per_seat) is not int or hands_per_seat < 4:
        raise ValueError('hands per seat must be >= 4')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be a positive integer')
    if type(bootstrap_seed) is not int:
        raise ValueError('bootstrap seed must be an integer')


def summarize(values, samples, seed):
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid paired replicate values')
    interval = paired_interval(values, samples=samples, seed=seed)
    return {'mean': sum(values) / len(values),
            'seed_values': list(values),
            'paired_replicate_interval_95': list(interval),
            'positive_replicate_fraction': sum(x > 0 for x in values) / len(values)}


def experiment(*, seeds=SEEDS, hands_per_seat=HANDS_PER_SEAT,
               bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED):
    seeds = tuple(seeds)
    validate(seeds, hands_per_seat, bootstrap_samples, bootstrap_seed)
    if set(ABLATIONS) != {'adaptive', 'frozen', 'opponent_only', 'history_only'}:
        raise ValueError('factorial arms unavailable')
    if ABLATIONS['history_only'] != (False, True) or ABLATIONS['frozen'] != (False, False):
        raise ValueError('primary component switches changed')
    if 'balanced' not in PURE_MIXTURES:
        raise ValueError('balanced mixture unavailable')
    design = {
        'version': 'patch22-prospective-history-only-v1',
        'hypothesis_selected_from': 'patch21_exploratory_component_attribution',
        'scope': 'fresh_seed_prospective_replication_not_new_policy_optimization',
        'primary_endpoint': 'history_only_minus_frozen_seat_averaged_chips_per_hand',
        'primary_hypothesis': 'mean_history_only_minus_frozen_greater_than_zero',
        'primary_decision_rule': 'two_sided_95pct_paired_seed_bootstrap_lower_bound_gt_zero',
        'secondary_comparisons': {k: list(v) for k, v in SECONDARY_CONTRASTS.items()},
        'secondary_inference': 'exploratory_no_multiplicity_adjustment',
        'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
        'cfr_iterations': CFR_BUDGET, 'temperature': TEMPERATURE,
        'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
        'mixture': 'balanced', 'arms': {k: list(v) for k, v in ABLATIONS.items()},
        'matched_chance_deals_across_arms_and_seats': True,
        'independent_policy_states_per_arm_and_seat': True,
        'fixed_cfr_average_strategy': True,
    }
    signature = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    solver = LeducCFR()
    solver.train(CFR_BUDGET)
    strategy = solver.average_strategy()
    deals = all_deals()
    if not deals:
        raise ValueError('empty chance-deal enumeration')
    replicates = []
    for seed in seeds:
        rng = random.Random(seed)
        schedule = [deals[rng.randrange(len(deals))] for _ in range(hands_per_seat)]
        arms = ablation_replicate(PURE_MIXTURES['balanced'], strategy, schedule, seed, TEMPERATURE)
        if set(arms) != set(ABLATIONS):
            raise ValueError('missing experimental arm')
        for name, record in arms.items():
            a, b, avg = (record['seat0_mean'], record['seat1_mean'], record['seat_average_payoff'])
            if not all(math.isfinite(x) for x in (a, b, avg)) or abs((a+b)/2 - avg) > 1e-9:
                raise ValueError('inconsistent payoffs in ' + name)
        replicates.append({'seed': seed, 'conditions': arms})

    def difference(left, right, field='seat_average_payoff'):
        return [r['conditions'][left][field] - r['conditions'][right][field] for r in replicates]

    primary = summarize(difference(*PRIMARY_ARMS), bootstrap_samples, bootstrap_seed)
    secondary = {
        name: summarize(difference(*pair), bootstrap_samples, bootstrap_seed + 1 + idx)
        for idx, (name, pair) in enumerate(SECONDARY_CONTRASTS.items())
    }
    primary_seats = {
        str(seat): summarize(difference(*PRIMARY_ARMS, f'seat{seat}_mean'),
                             bootstrap_samples, bootstrap_seed + 100 + seat)
        for seat in (0, 1)
    }
    return {'status': 'completed', 'design': design, 'design_sha256': signature,
            'primary': primary,
            'primary_decision': {'rule': design['primary_decision_rule'],
                                 'criterion_met': primary['paired_replicate_interval_95'][0] > 0},
            'primary_by_seat': primary_seats, 'secondary': secondary,
            'replicates': replicates,
            'limitations': 'Seed-cluster bootstrap reflects sampled tournament variability under a fixed CFR strategy. Primary question selected from Patch 21. No cross-game generalization or causal attribution to EBID. Secondary analyses unadjusted.',
            'provenance': {'original_policy_unchanged': True, 'cfr_solver_unchanged': True,
                           'prior_results_unchanged': True, 'human_data_accessed': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands-per-seat', type=int, default=HANDS_PER_SEAT)
    parser.add_argument('--bootstrap-samples', type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--output', default='validation/prospective-history-adaptation.json')
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        parser.error('refusing to overwrite existing report: ' + str(destination))
    result = experiment(hands_per_seat=args.hands_per_seat,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.bootstrap_seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'design_sha256': result['design_sha256'],
                      'primary': {'mean': result['primary']['mean'],
                                  'interval_95': result['primary']['paired_replicate_interval_95']},
                      'primary_decision': result['primary_decision'],
                      'secondary': {key: {'mean': value['mean'],
                                          'interval_95': value['paired_replicate_interval_95']}
                                    for key, value in result['secondary'].items()},
                      'output': str(destination)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
