"""Patch 20: prospective replication of the Patch 19 ten-hand memory effect.

Primary hypothesis and rule were selected from Patch 19, before new seeds.
No policy modification, opponent retraining across replicates, or hindsight tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .adaptation_interaction import paired_interval
from .adaptation_timescale_intervention import CFR_BUDGET, TEMPERATURE, seat_payoffs
from .engine import all_deals
from .leduc_cfr import LeducCFR

SEEDS = tuple(range(3201, 3231))
PRIMARY_PERIOD = 10
SECONDARY_PERIODS = (25, 100)
HANDS_PER_SEAT = 200
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 3200
PREVIOUS_SEED_RANGES = ((2201, 2205), (2401, 2420), (2601, 2630),
                        (2801, 2830), (3001, 3030))


def _validate(seeds, hands_per_seat, bootstrap_samples, bootstrap_seed):
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds) or
            any(type(x) is not int or x < 0 for x in seeds)):
        raise ValueError('require at least two distinct nonnegative integer seeds')
    if any(start <= seed <= end for seed in seeds for start, end in PREVIOUS_SEED_RANGES):
        raise ValueError('seeds overlap previous PCC adaptation experiments')
    if type(hands_per_seat) is not int or hands_per_seat < max(SECONDARY_PERIODS):
        raise ValueError('hands per seat must support all fixed reset periods')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be positive')
    if type(bootstrap_seed) is not int:
        raise ValueError('bootstrap seed must be an integer')


def _summary(values, samples, seed):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('replicate contrasts must be finite')
    low, high = paired_interval(values, samples=samples, seed=seed)
    return {'mean': sum(values) / len(values), 'seed_values': values,
            'paired_replicate_interval_95': [low, high],
            'positive_replicate_fraction': sum(v > 0 for v in values) / len(values)}


def experiment(*, seeds=SEEDS, hands_per_seat=HANDS_PER_SEAT,
               bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED):
    seeds = tuple(seeds)
    _validate(seeds, hands_per_seat, bootstrap_samples, bootstrap_seed)
    arms = {'continuous': None, 'reset_10': PRIMARY_PERIOD,
            'reset_25': 25, 'reset_100': 100}
    design = {
        'design_version': 'patch20-prospective-memory-retention-v1',
        'primary_hypothesis': 'continuous_payoff > both_reset_every_10_payoff',
        'primary_endpoint': 'paired_seat_averaged_net_chips_per_hand_continuous_minus_reset10',
        'primary_decision_rule': 'two-sided_95pct_paired_replicate_percentile_bootstrap_lower_bound_gt_zero',
        'hypothesis_selected_from': 'patch19_exploratory_results',
        'secondary_periods': list(SECONDARY_PERIODS),
        'secondary_inference': 'descriptive_only_unadjusted_for_multiplicity',
        'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
        'cfr_iterations': CFR_BUDGET, 'temperature': TEMPERATURE,
        'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
        'matched_chance_deals_across_arms_and_seats': True,
        'reset_both_learned_modules_without_reseeding_random_streams': True,
        'independent_learning_state_per_arm_and_seat': True,
        'fixed_cfr_average_strategy': True,
        'arms': {name: {'opponent_period': period, 'history_period': period}
                 for name, period in arms.items()},
    }
    digest = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    solver = LeducCFR()
    solver.train(CFR_BUDGET)
    strategy = solver.average_strategy()
    deals = all_deals()
    if not deals:
        raise ValueError('no chance deals')
    replicates = []
    for seed in seeds:
        rng = random.Random(seed)
        schedule = [deals[rng.randrange(len(deals))] for _ in range(hands_per_seat)]
        outcomes = {}
        for name, period in arms.items():
            seat_results = [seat_payoffs(strategy, schedule, seed, seat, period, period)
                            for seat in (0, 1)]
            if any(len(xs) != hands_per_seat or not all(math.isfinite(v) for v in xs)
                   for xs in seat_results):
                raise ValueError('non-finite or incomplete per-hand results')
            seat_means = [sum(xs) / hands_per_seat for xs in seat_results]
            outcomes[name] = {'seat0_mean': seat_means[0], 'seat1_mean': seat_means[1],
                              'seat_average_payoff': sum(seat_means) / 2}
        replicates.append({'seed': seed, 'arms': outcomes})
    summaries = {}
    for ix, name in enumerate(('reset_10', 'reset_25', 'reset_100')):
        differences = [r['arms']['continuous']['seat_average_payoff'] -
                       r['arms'][name]['seat_average_payoff'] for r in replicates]
        summaries[name] = _summary(differences, bootstrap_samples, bootstrap_seed + ix)
    primary = summaries['reset_10']
    return {
        'status': 'completed', 'design': design, 'design_sha256': digest,
        'primary': primary,
        'primary_decision': {'criterion_met': primary['paired_replicate_interval_95'][0] > 0,
                             'rule': design['primary_decision_rule']},
        'secondary': {'reset_25': summaries['reset_25'], 'reset_100': summaries['reset_100']},
        'replicates': replicates,
        'limitations': 'Prospective only with respect to fresh tournament seeds; the primary hypothesis was selected using Patch 19. Bootstrap clusters by seed. No multiplicity-adjusted secondary inference, no claims about other environments, and fixed CFR solution.',
        'provenance': {'prior_reports_read_only': True, 'pcc_policy_unchanged': True,
                       'cfr_solver_unchanged': True, 'human_data_accessed': False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands-per-seat', type=int, default=HANDS_PER_SEAT)
    parser.add_argument('--bootstrap-samples', type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--output', default='validation/prospective-memory-retention.json')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('refusing to overwrite existing report: ' + str(output))
    result = experiment(hands_per_seat=args.hands_per_seat,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.bootstrap_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'design_sha256': result['design_sha256'],
                      'primary': result['primary'], 'primary_decision': result['primary_decision'],
                      'secondary': {k: {'mean': v['mean'], 'interval_95': v['paired_replicate_interval_95']}
                                    for k, v in result['secondary'].items()},
                      'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
