"""Paired adaptive PCC ablations against fixed CFR average strategies.

This is a sampled payoff study, not an exploitability measurement. CFR learns
before play; each PCC condition has independent learning state per seat/replicate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from .adaptive_cfr_tournament import bootstrap_interval, select_action
from .engine import all_deals, apply_action, initial_state, utility
from .leduc_cfr import LeducCFR
from .policies import PCCPolicy, PURE_MIXTURES

ABLATIONS = {
    'adaptive': (True, True),
    'frozen': (False, False),
    'opponent_only': (True, False),
    'history_only': (False, True),
}


def play_ablation_hand(deck, pcc, seat, strategy, cfr_rng, learn_opponent, learn_history):
    if seat not in (0, 1):
        raise ValueError('seat must be zero or one')
    state = initial_state(deck)
    while not state.terminal:
        if state.actor == seat:
            # PCCPolicy.decide updates action history after selecting an action;
            # roll back that update when action-history adaptation is disabled.
            if not learn_history:
                before = {k: dict(v) for k, v in pcc.action_history.context_actions.items()}
            action = pcc.decide(state).action
            if not learn_history:
                pcc.action_history.context_actions.clear()
                for context, counts in before.items():
                    pcc.action_history.context_actions[context].update(counts)
        else:
            key = (state.actor, state.information_key(state.actor))
            action = select_action(state.legal_actions(), strategy[key], cfr_rng)
            if learn_opponent:
                pcc.opponent_model.observe(state, action)
        state = apply_action(state, action)
    return float(utility(state, seat))


def ablation_replicate(weights, strategies, decks, seed, temperature=0.35):
    if not decks:
        raise ValueError('decks must be nonempty')
    outputs = {}
    for label, (learn_opponent, learn_history) in ABLATIONS.items():
        seat_means, curves = [], []
        for seat in (0, 1):
            # Same initial streams across conditions, separate mutable policies.
            pcc = PCCPolicy(weights, seed=seed * 4 + seat * 2, temperature=temperature)
            rng = random.Random(seed * 4 + seat * 2 + 1)
            payoffs = [play_ablation_hand(deck, pcc, seat, strategies, rng, learn_opponent, learn_history) for deck in decks]
            seat_means.append(float(np.mean(payoffs)))
            n = len(payoffs)
            curves.append([float(np.mean(payoffs[:max(1, n // 4)])), float(np.mean(payoffs[:max(1, n // 2)])), float(np.mean(payoffs))])
        outputs[label] = {'seat0_mean': seat_means[0], 'seat1_mean': seat_means[1],
                          'seat_average_payoff': float(np.mean(seat_means)),
                          'cumulative_quarter_half_full': [float(np.mean([curves[0][j], curves[1][j]])) for j in range(3)]}
    return outputs


def experiment(budgets=(100, 500, 2000), seeds=(2201, 2202, 2203, 2204, 2205), hands_per_seat=200,
               temperature=0.35, bootstrap_samples=2000):
    budgets, seeds = tuple(map(int, budgets)), tuple(map(int, seeds))
    if not budgets or min(budgets) < 1 or len(set(budgets)) != len(budgets):
        raise ValueError('budgets must be distinct positive integers')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or hands_per_seat < 1:
        raise ValueError('require distinct replicate seeds and positive hands')
    if not math.isfinite(temperature) or temperature <= 0 or bootstrap_samples < 1:
        raise ValueError('invalid temperature/bootstrap design')
    deals = all_deals()
    # Cache chance schedules across budgets, policies, and ablations.
    schedules = {}
    for s in seeds:
        rng = random.Random(s)
        schedules[s] = [deals[rng.randrange(len(deals))] for _ in range(hands_per_seat)]
    solver = LeducCFR()
    report = {}
    prior = 0
    for budget in sorted(budgets):
        solver.train(budget - prior)
        prior = budget
        strategy = solver.average_strategy()
        cases = {}
        for case_index, (name, weights) in enumerate(PURE_MIXTURES.items()):
            reps = []
            for seed in seeds:
                paired = ablation_replicate(weights, strategy, schedules[seed], seed, temperature)
                reps.append({'seed': seed, 'conditions': paired})
            conditions = {}
            for label in ABLATIONS:
                vals = [r['conditions'][label]['seat_average_payoff'] for r in reps]
                frozen = [r['conditions']['frozen']['seat_average_payoff'] for r in reps]
                diff = [a - b for a, b in zip(vals, frozen)]
                conditions[label] = {
                    'mean_pcc_payoff': float(np.mean(vals)),
                    'replicate_interval_95': bootstrap_interval(vals, 123400 + budget + case_index * 17, bootstrap_samples),
                    'delta_vs_frozen': float(np.mean(diff)),
                    'delta_vs_frozen_interval_95': bootstrap_interval(diff, 234500 + budget + case_index * 17, bootstrap_samples),
                    'mean_cumulative_quarter_half_full': [float(np.mean([r['conditions'][label]['cumulative_quarter_half_full'][j] for r in reps])) for j in range(3)]}
            cases[name] = {'conditions': conditions, 'replicates': reps}
        report[str(budget)] = {'cfr_evaluation': solver.evaluate(), 'cases': cases}
    return {'status': 'completed', 'design': {'budgets': list(budgets), 'seeds': list(seeds),
            'hands_per_seat': hands_per_seat, 'temperature': temperature,
            'bootstrap_samples': bootstrap_samples, 'matched_deals_across_conditions': True,
            'independent_learning_state_per_condition_and_seat': True,
            'cfr_fixed_within_tournament': True}, 'ablations': {k: list(v) for k, v in ABLATIONS.items()},
            'budgets': report,
            'limitations': 'Paired replicate bootstrap intervals use replicate means, not independent hands. Early/late cumulative means are descriptive, not independent learning-rate estimates. No exploitability inference for PCC.',
            'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgets', type=int, nargs='+', default=[100, 500, 2000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2201, 2202, 2203, 2204, 2205])
    parser.add_argument('--hands-per-seat', type=int, default=200)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--temperature', type=float, default=0.35)
    parser.add_argument('--output', default='validation/adaptive-cfr-ablations.json')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('refusing to overwrite existing report: ' + str(output))
    result = experiment(args.budgets, args.seeds, args.hands_per_seat, args.temperature, args.bootstrap_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    compact = {budget: {case: {name: round(v['delta_vs_frozen'], 5) for name, v in values['conditions'].items()}
                        for case, values in entry['cases'].items()} for budget, entry in result['budgets'].items()}
    print(json.dumps({'status': 'completed', 'delta_vs_frozen': compact, 'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
