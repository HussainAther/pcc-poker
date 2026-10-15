"""Patch 21: fresh-seed, factorial PCC adaptation component attribution.

Exploratory component comparisons; no policy or solver changes. All payoff
uncertainty is clustered by independent seed, never by individual hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .adaptation_interaction import CONTRASTS, paired_interval
from .adaptive_cfr_ablations import ABLATIONS, ablation_replicate
from .engine import all_deals
from .leduc_cfr import LeducCFR
from .policies import PURE_MIXTURES

SEEDS = tuple(range(3401, 3431))
CFR_BUDGET = 2000
HANDS_PER_SEAT = 200
TEMPERATURE = 0.35
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 3400
OLD_SEED_RANGES = ((1501, 1505), (1601, 1605), (1701, 1705),
                   (1800, 1805), (1901, 1905), (2101, 2105),
                   (2201, 2205), (2401, 2420), (2601, 2630),
                   (2801, 2830), (3001, 3030), (3201, 3230))


def validate(seeds, hands_per_seat, temperature, bootstrap_samples, bootstrap_seed):
    if (len(seeds) < 2 or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('require two or more distinct nonnegative integer seeds')
    if any(a <= s <= b for s in seeds for a, b in OLD_SEED_RANGES):
        raise ValueError('seed overlaps previous experiment')
    if type(hands_per_seat) is not int or hands_per_seat < 4:
        raise ValueError('hands per seat must be >= 4')
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be positive')
    if type(bootstrap_seed) is not int:
        raise ValueError('bootstrap seed must be integer')


def summarize(values, *, samples, seed):
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid replicate values')
    return {'mean': sum(values) / len(values), 'seed_values': values,
            'paired_replicate_interval_95': paired_interval(values, samples=samples, seed=seed),
            'positive_replicate_fraction': sum(x > 0 for x in values) / len(values)}


def experiment(*, seeds=SEEDS, hands_per_seat=HANDS_PER_SEAT,
               temperature=TEMPERATURE, bootstrap_samples=BOOTSTRAP_SAMPLES,
               bootstrap_seed=BOOTSTRAP_SEED):
    seeds = tuple(seeds)
    validate(seeds, hands_per_seat, temperature, bootstrap_samples, bootstrap_seed)
    if 'balanced' not in PURE_MIXTURES:
        raise ValueError('missing balanced mixture')
    if set(ABLATIONS) != {'adaptive', 'frozen', 'opponent_only', 'history_only'}:
        raise ValueError('factorial arms unavailable')
    design = {
        'version': 'patch21-fresh-seed-factorial-v1',
        'scope': 'exploratory_mechanism_attribution_not_confirmatory_synergy',
        'previously_supported_result': 'patch16_full_adaptive_vs_frozen_cfr2000',
        'cfr_iterations': CFR_BUDGET, 'mixture': 'balanced',
        'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
        'temperature': temperature, 'bootstrap_samples': bootstrap_samples,
        'bootstrap_seed': bootstrap_seed, 'arms': {k: list(v) for k, v in ABLATIONS.items()},
        'matched_deals_across_arms_and_seats': True,
        'independent_learning_state_per_arm_and_seat': True,
        'fixed_cfr_average_strategy': True,
        'no_multiple_testing_correction': True,
    }
    signature = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    solver = LeducCFR()
    solver.train(CFR_BUDGET)
    strategy = solver.average_strategy()
    deals = all_deals()
    if not deals:
        raise ValueError('empty deal enumeration')
    reps = []
    for seed in seeds:
        rng = random.Random(seed)
        schedule = [deals[rng.randrange(len(deals))] for _ in range(hands_per_seat)]
        arms = ablation_replicate(PURE_MIXTURES['balanced'], strategy, schedule, seed, temperature)
        if set(arms) != set(ABLATIONS):
            raise ValueError('missing ablation condition')
        for name, record in arms.items():
            a, b, c = record['seat0_mean'], record['seat1_mean'], record['seat_average_payoff']
            if not all(math.isfinite(x) for x in (a, b, c)) or abs(c - (a+b)/2) > 1e-9:
                raise ValueError('inconsistent seat payoffs in ' + name)
        reps.append({'seed': seed, 'conditions': arms})
    contrasts = {}
    for index, (name, coefficients) in enumerate(CONTRASTS.items()):
        xs = [sum(weight * r['conditions'][arm]['seat_average_payoff']
                  for arm, weight in coefficients.items()) for r in reps]
        contrasts[name] = summarize(xs, samples=bootstrap_samples, seed=bootstrap_seed + index)
    seat_contrasts = {}
    for index, seat in enumerate((0, 1)):
        field = f'seat{seat}_mean'
        seat_contrasts[str(seat)] = summarize([
            r['conditions']['adaptive'][field] - r['conditions']['frozen'][field]
            for r in reps], samples=bootstrap_samples, seed=bootstrap_seed + 100 + index)
    return {'status': 'completed', 'design_sha256': signature, 'design': design,
            'contrasts': contrasts, 'seat_full_vs_frozen': seat_contrasts,
            'replicates': reps,
            'interpretation': 'Individual component contrasts are conditional effects of enabling a switch; the factorial interaction is not a general synergy theorem.',
            'limitations': 'Exploratory contrasts with unadjusted intervals. Finite sampled poker payoffs, independent by seed rather than hand; frozen CFR opponent. No generalization to other mixtures, opponents, or EBID mathematical theory.',
            'provenance': {'prior_reports_unchanged': True, 'policies_unchanged': True,
                           'solver_unchanged': True, 'human_data_accessed': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands-per-seat', type=int, default=HANDS_PER_SEAT)
    parser.add_argument('--bootstrap-samples', type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument('--bootstrap-seed', type=int, default=BOOTSTRAP_SEED)
    parser.add_argument('--output', default='validation/adaptation-mechanism-attribution.json')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('refusing to overwrite existing report: ' + str(output))
    result = experiment(hands_per_seat=args.hands_per_seat,
                        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'design_sha256': result['design_sha256'],
                      'contrasts': {k: {'mean': v['mean'], 'interval_95': v['paired_replicate_interval_95']}
                                    for k, v in result['contrasts'].items()},
                      'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
