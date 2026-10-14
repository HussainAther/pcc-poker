"""Patch 19: exploratory retained-memory timescale and component-reset experiment.

The primary purpose is mechanistic exploration after Patch 18's negative
continuous-vs-50-hand result. No confirmatory inference is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .adaptation_interaction import paired_interval
from .adaptive_cfr_tournament import select_action
from .engine import all_deals, apply_action, initial_state, utility
from .leduc_cfr import LeducCFR
from .policies import OpponentModel, PCCPolicy, PURE_MIXTURES

DEFAULT_SEEDS = tuple(range(3001, 3031))
DEFAULT_PERIODS = (1, 10, 25, 50, 100)
DEFAULT_COMPONENT_PERIOD = 50
CFR_BUDGET = 2000
TEMPERATURE = 0.35


def conditions(periods=DEFAULT_PERIODS, component_period=DEFAULT_COMPONENT_PERIOD):
    """Condition definitions: (name, opponent-reset interval, history-reset interval)."""
    if not isinstance(periods, (tuple, list)) or not periods or any(
        type(x) is not int or x < 1 for x in periods
    ) or len(set(periods)) != len(periods):
        raise ValueError('periods must be distinct positive integers')
    if type(component_period) is not int or component_period < 1:
        raise ValueError('component period must be positive')
    output = [('continuous', None, None)]
    output += [(f'both_every_{x}', x, x) for x in periods]
    output += [('opponent_only_reset', component_period, None),
               ('history_only_reset', None, component_period)]
    return output


def play_hand(deck, pcc, seat, strategy, cfr_rng):
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


def seat_payoffs(strategy, decks, seed, seat, opponent_period, history_period,
                 temperature=TEMPERATURE):
    """Reset only requested learned modules; never reseed the policy or CFR RNG."""
    if not decks or seat not in (0, 1) or any(
        period is not None and (type(period) is not int or period < 1)
        for period in (opponent_period, history_period)
    ):
        raise ValueError('invalid seat, decks, or reset period')
    base = seed * 8 + seat * 2
    pcc = PCCPolicy(PURE_MIXTURES['balanced'], seed=base, temperature=temperature)
    cfr_rng = random.Random(base + 1)
    result = []
    for index, deck in enumerate(decks):
        if index > 0:
            if opponent_period is not None and index % opponent_period == 0:
                pcc.opponent_model = OpponentModel()
            if history_period is not None and index % history_period == 0:
                pcc.action_history = OpponentModel()
        result.append(play_hand(deck, pcc, seat, strategy, cfr_rng))
    return result


def _summarize(values, bootstrap_seed, bootstrap_samples):
    low, high = paired_interval(values, samples=bootstrap_samples, seed=bootstrap_seed)
    return {'mean': sum(values) / len(values), 'seed_values': values,
            'paired_replicate_interval_95': [low, high],
            'positive_replicate_fraction': sum(x > 0 for x in values) / len(values)}


def experiment(*, seeds=DEFAULT_SEEDS, hands_per_seat=200, periods=DEFAULT_PERIODS,
               component_period=DEFAULT_COMPONENT_PERIOD, bootstrap_samples=2000,
               bootstrap_seed=3000):
    seeds = tuple(seeds)
    periods = tuple(periods)
    arms = conditions(periods, component_period)
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(x) is not int or x < 0 for x in seeds)):
        raise ValueError('require at least two unique nonnegative integer seeds')
    if set(seeds) & (set(range(2201, 2206)) | set(range(2401, 2421)) |
                     set(range(2601, 2631)) | set(range(2801, 2831))):
        raise ValueError('seeds overlap previous adaptation experiments')
    if type(hands_per_seat) is not int or hands_per_seat < 4:
        raise ValueError('hands per seat must be >= 4')
    if max(periods + (component_period,)) > hands_per_seat:
        raise ValueError('reset periods cannot exceed hand count')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1 or type(bootstrap_seed) is not int:
        raise ValueError('invalid bootstrap configuration')
    design = {'design_version': 'patch19-exploratory-timescale-v1',
              'interpretation': 'exploratory_not_preregistered_confirmation',
              'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
              'cfr_iterations': CFR_BUDGET, 'temperature': TEMPERATURE,
              'periods': list(periods), 'component_period': component_period,
              'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
              'conditions': [dict(name=n, opponent_reset_period=o, history_reset_period=h)
                             for n, o, h in arms],
              'reset_before_hand': 'period+1, 2*period+1, ...',
              'pcc_and_cfr_rng_preserved_across_resets': True,
              'same_chance_deals_across_arms_and_seats': True,
              'independent_learning_state_per_arm_and_seat': True}
    digest = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    solver = LeducCFR()
    solver.train(CFR_BUDGET)
    strategy = solver.average_strategy()
    decks = all_deals()
    reps = []
    for seed in seeds:
        rng = random.Random(seed)
        schedule = [decks[rng.randrange(len(decks))] for _ in range(hands_per_seat)]
        arm_values = {}
        for name, opponent_period, history_period in arms:
            seat_values = [seat_payoffs(strategy, schedule, seed, seat, opponent_period, history_period)
                           for seat in (0, 1)]
            n = hands_per_seat
            q = n // 4
            h = n // 2
            def segment(start, end):
                return sum(sum(x[start:end]) for x in seat_values) / (2 * (end - start))
            arm_values[name] = {
                'seat0_mean': sum(seat_values[0]) / n,
                'seat1_mean': sum(seat_values[1]) / n,
                'seat_average_payoff': segment(0, n),
                'segments': {'first_quarter': segment(0, q),
                             'second_quarter': segment(q, h),
                             'second_half': segment(h, n)}}
        reps.append({'seed': seed, 'arms': arm_values})
    summaries = {}
    for index, (name, _, _) in enumerate(arms):
        absolute = [r['arms'][name]['seat_average_payoff'] for r in reps]
        relative = [r['arms']['continuous']['seat_average_payoff'] - v for r, v in
                    zip(reps, absolute)]
        summaries[name] = {
            'absolute_payoff': _summarize(absolute, bootstrap_seed + 10 * index + 1, bootstrap_samples),
            'continuous_minus_condition': _summarize(relative, bootstrap_seed + 10 * index + 2, bootstrap_samples),
        }
    return {'status': 'completed', 'design': design, 'design_sha256': digest,
            'summaries': summaries, 'replicates': reps,
            'limitations': 'Exploratory comparisons selected after Patch 18. Intervals descriptive, unadjusted for multiple comparisons. Common RNG streams do not guarantee matched realized actions. Reset changes future policy behavior; results are conditional on a single fixed CFR solution and game setting.',
            'provenance': {'prior_reports_read_only': True, 'pcc_policy_unchanged': True,
                           'cfr_solver_unchanged': True, 'human_data_accessed': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--hands-per-seat', type=int, default=200)
    parser.add_argument('--periods', nargs='+', type=int, default=list(DEFAULT_PERIODS))
    parser.add_argument('--component-period', type=int, default=DEFAULT_COMPONENT_PERIOD)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=3000)
    parser.add_argument('--output', default='validation/adaptation-timescale-intervention.json')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('refusing to overwrite existing report: ' + str(output))
    report = experiment(seeds=args.seeds, hands_per_seat=args.hands_per_seat,
                        periods=args.periods, component_period=args.component_period,
                        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps({'status': report['status'], 'design_sha256': report['design_sha256'],
                      'continuous_minus_condition': {name: {'mean': result['continuous_minus_condition']['mean'],
                      'interval_95': result['continuous_minus_condition']['paired_replicate_interval_95']}
                      for name, result in report['summaries'].items()}, 'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
