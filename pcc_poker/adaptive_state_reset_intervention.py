"""Patch 18: prospective retained-state intervention for Balanced PCC.

The sole primary contrast is continuous learning minus a policy reset every 50
hands, against a fixed CFR-2000 average strategy. A reset occurs *before* hands
51, 101, 151, ... and resets both PCC opponent and action-history state by
constructing a new policy while preserving its RNG state. All arms use
identical decks and independent, persistent CFR action streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .adaptation_interaction import paired_interval
from .adaptive_cfr_tournament import select_action
from .engine import all_deals, apply_action, initial_state, utility
from .leduc_cfr import LeducCFR
from .policies import PCCPolicy, PURE_MIXTURES

DEFAULT_SEEDS = tuple(range(2801, 2831))
CFR_BUDGET = 2000
RESET_PERIOD = 50
TEMPERATURE = 0.35


def play_reset_hand(deck, pcc, seat, strategy, cfr_rng):
    """Execute one hand, updating both PCC learning mechanisms."""
    state = initial_state(deck)
    while not state.terminal:
        if state.actor == seat:
            action = pcc.decide(state).action
        else:
            key = (state.actor, state.information_key(state.actor))
            action = select_action(state.legal_actions(), strategy[key], cfr_rng)
            pcc.opponent_model.observe(state, action)
        state = apply_action(state, action)
    return float(utility(state, seat))


def seat_arm(weights, strategy, decks, seed, seat, reset_period, temperature=TEMPERATURE):
    """Return per-hand payoffs. None means uninterrupted persistent state."""
    if not decks or seat not in (0, 1) or (reset_period is not None and (
        not isinstance(reset_period, int) or isinstance(reset_period, bool) or reset_period < 1)):
        raise ValueError('invalid seat, decks, or reset period')
    # Stable independent streams across seats and arms; neither RNG resets.
    base = seed * 8 + seat * 2
    cfr_rng = random.Random(base + 1)
    pcc = None
    payoffs = []
    for index, deck in enumerate(decks):
        if pcc is None or (reset_period is not None and index % reset_period == 0):
            rng_state = pcc.rng.getstate() if pcc is not None else None
            pcc = PCCPolicy(weights, seed=base, temperature=temperature)
            if rng_state is not None:
                pcc.rng.setstate(rng_state)
        payoffs.append(play_reset_hand(deck, pcc, seat, strategy, cfr_rng))
    return payoffs


def replicate(strategy, decks, seed, reset_period=RESET_PERIOD):
    weights = PURE_MIXTURES['balanced']
    output = {}
    for arm, period in (('continuous', None), ('periodic_reset', reset_period), ('reset_each_hand', 1)):
        seats = [seat_arm(weights, strategy, decks, seed, seat, period) for seat in (0, 1)]
        n = len(decks)
        quarter = max(1, n // 4)
        half = max(1, n // 2)
        def paired_mean(start, end):
            return sum(sum(vals[start:end]) for vals in seats) / (2 * (end - start))
        output[arm] = {
            'seat0_mean': sum(seats[0]) / n,
            'seat1_mean': sum(seats[1]) / n,
            'seat_average_payoff': paired_mean(0, n),
            'segments': {'first_quarter': paired_mean(0, quarter),
                         'second_quarter': paired_mean(quarter, half) if quarter < half else None,
                         'second_half': paired_mean(half, n) if half < n else None},
        }
    return output


def _summary(values, seed, samples):
    lo, hi = paired_interval(values, samples=samples, seed=seed)
    return {'mean': sum(values) / len(values), 'seed_values': values,
            'paired_replicate_interval_95': [lo, hi],
            'positive_replicate_fraction': sum(x > 0 for x in values) / len(values)}


def experiment(*, seeds=DEFAULT_SEEDS, hands_per_seat=200, reset_period=RESET_PERIOD,
               bootstrap_samples=2000, bootstrap_seed=2800):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(
        not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds):
        raise ValueError('require distinct nonnegative integer seeds')
    if set(seeds) & (set(range(2201, 2206)) | set(range(2401, 2421)) | set(range(2601, 2631))):
        raise ValueError('fresh seeds must not overlap prior adaptation experiments')
    if not isinstance(hands_per_seat, int) or isinstance(hands_per_seat, bool) or hands_per_seat < 4:
        raise ValueError('hands_per_seat must be >= 4')
    if (not isinstance(reset_period, int) or isinstance(reset_period, bool) or
            reset_period < 2 or reset_period >= hands_per_seat):
        raise ValueError('reset period must be >=2 and less than hands per seat')
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples < 1:
        raise ValueError('invalid bootstrap samples')
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError('invalid bootstrap seed')
    design = {'design_version': 'patch18-retained-state-reset-v1', 'primary_pcc': 'balanced',
              'primary_cfr_budget': CFR_BUDGET, 'primary_contrast': 'continuous_minus_periodic_reset',
              'primary_decision_rule': 'two-sided 95% paired-replicate percentile bootstrap lower bound > 0',
              'seeds': list(seeds), 'hands_per_seat': hands_per_seat, 'reset_period': reset_period,
              'temperature': TEMPERATURE, 'bootstrap_samples': bootstrap_samples,
              'bootstrap_seed': bootstrap_seed, 'matched_deals': True,
              'cfr_rng_persistent_across_resets': True, 'pcc_rng_persistent_across_resets': True,
              'secondary_reset_each_hand': True,
              'policy_optimization': False}
    digest = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    solver = LeducCFR()
    solver.train(CFR_BUDGET)
    strategy = solver.average_strategy()
    deals = all_deals()
    reps = []
    for seed in seeds:
        rng = random.Random(seed)
        schedule = [deals[rng.randrange(len(deals))] for _ in range(hands_per_seat)]
        arms = replicate(strategy, schedule, seed, reset_period)
        gain = arms['continuous']['seat_average_payoff'] - arms['periodic_reset']['seat_average_payoff']
        reps.append({'seed': seed, 'arms': arms, 'continuous_minus_periodic_reset': gain,
                     'continuous_minus_reset_each_hand': arms['continuous']['seat_average_payoff'] - arms['reset_each_hand']['seat_average_payoff'],
                     'seat0_gain': arms['continuous']['seat0_mean'] - arms['periodic_reset']['seat0_mean'],
                     'seat1_gain': arms['continuous']['seat1_mean'] - arms['periodic_reset']['seat1_mean']})
    primary = _summary([r['continuous_minus_periodic_reset'] for r in reps], bootstrap_seed + 1, bootstrap_samples)
    secondary = _summary([r['continuous_minus_reset_each_hand'] for r in reps], bootstrap_seed + 2, bootstrap_samples)
    return {'status': 'completed', 'design': design, 'design_sha256': digest,
            'primary': primary, 'primary_decision': {'criterion_met': primary['paired_replicate_interval_95'][0] > 0,
            'rule': design['primary_decision_rule']}, 'secondary_continuous_minus_reset_each_hand': secondary,
            'replicates': reps,
            'limitations': 'Reset recreates both PCC learned-state modules while preserving its RNG state. Common random streams do not guarantee identical action trajectories after policies diverge. Checkpoints are descriptive; bootstrap resamples independent replicate blocks. No inference of exploitability or PCC superiority.',
            'provenance': {'prior_reports_read_only': True, 'pcc_policy_unchanged': True, 'human_data_accessed': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--hands-per-seat', type=int, default=200)
    parser.add_argument('--reset-period', type=int, default=RESET_PERIOD)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2800)
    parser.add_argument('--output', default='validation/adaptive-state-reset-intervention.json')
    args = parser.parse_args(argv)
    dest = Path(args.output)
    if dest.exists():
        parser.error('refusing to overwrite existing report: ' + str(dest))
    report = experiment(seeds=args.seeds, hands_per_seat=args.hands_per_seat,
                        reset_period=args.reset_period, bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.bootstrap_seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'primary': report['primary'],
                      'primary_decision': report['primary_decision'],
                      'secondary': report['secondary_continuous_minus_reset_each_hand']['mean'],
                      'output': str(dest)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
