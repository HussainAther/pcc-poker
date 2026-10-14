"""Seeded adaptive PCC versus fixed finite-iteration CFR Leduc tournament.

The opponent model is updated after every observed CFR action. PCC's action
history is updated by PCCPolicy.decide. No CFR learning occurs during matches.
The unit of uncertainty is an independent seat-paired tournament replicate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from .engine import all_deals, apply_action, initial_state, utility
from .leduc_cfr import LeducCFR
from .policies import PCCPolicy, PURE_MIXTURES


def select_action(actions, probabilities, rng):
    values = np.asarray(probabilities, dtype=float)
    if len(actions) != len(values) or not len(actions) or not np.all(np.isfinite(values)) or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError('invalid action probabilities')
    threshold = rng.random()
    total = 0.0
    for action, prob in zip(actions, values):
        total += float(prob)
        if threshold <= total:
            return action
    return actions[-1]


def play_hand(deck, pcc, pcc_seat, cfr_strategy, cfr_rng):
    if pcc_seat not in (0, 1):
        raise ValueError('pcc_seat must be 0 or 1')
    state = initial_state(deck)
    while not state.terminal:
        actor = state.actor
        if actor == pcc_seat:
            action = pcc.decide(state).action
        else:
            key = (actor, state.information_key(actor))
            action = select_action(state.legal_actions(), cfr_strategy[key], cfr_rng)
            pcc.opponent_model.observe(state, action)
        state = apply_action(state, action)
    return float(utility(state, pcc_seat))


def paired_replicate(weights, cfr_strategy, decks, seed, temperature=0.35):
    """Identical chance deals per seat, independent learning and random streams."""
    if len(decks) == 0:
        raise ValueError('at least one deal is required')
    scores = []
    for seat in (0, 1):
        pcc = PCCPolicy(weights, seed=seed * 4 + 2 * seat, temperature=temperature)
        opponent_rng = random.Random(seed * 4 + 2 * seat + 1)
        scores.append([play_hand(deck, pcc, seat, cfr_strategy, opponent_rng) for deck in decks])
    return {'seat0_mean': float(np.mean(scores[0])), 'seat1_mean': float(np.mean(scores[1])),
            'seat_average_payoff': float((np.mean(scores[0]) + np.mean(scores[1])) / 2),
            'seat0_payoffs': scores[0], 'seat1_payoffs': scores[1],
            'cfr_seat_average_payoff': float(-(np.mean(scores[0]) + np.mean(scores[1])) / 2)}


def bootstrap_interval(values, seed, samples=2000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or samples < 1:
        raise ValueError('invalid bootstrap input')
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def tournament(iterations=100, seeds=(2101, 2102, 2103, 2104, 2105), hands_per_seat=200, temperature=0.35, bootstrap_samples=2000):
    seeds = tuple(int(x) for x in seeds)
    if iterations < 1 or hands_per_seat < 1 or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('positive iterations/hands and at least two distinct seeds required')
    if not math.isfinite(temperature) or temperature <= 0 or bootstrap_samples < 1:
        raise ValueError('invalid temperature or bootstrap samples')
    solver = LeducCFR()
    solver.train(iterations)
    strategy = solver.average_strategy()
    deals = all_deals()
    cases = {}
    for label, weights in PURE_MIXTURES.items():
        replicates = []
        for seed in seeds:
            deck_rng = random.Random(seed)
            decks = [deals[deck_rng.randrange(len(deals))] for _ in range(hands_per_seat)]
            result = paired_replicate(weights, strategy, decks, seed, temperature)
            replicates.append({'seed': seed, 'seat0_mean': result['seat0_mean'],
                               'seat1_mean': result['seat1_mean'], 'seat_average_payoff': result['seat_average_payoff']})
        means = [x['seat_average_payoff'] for x in replicates]
        cases[label] = {'weights': list(weights), 'replicates': replicates,
                        'mean_pcc_payoff': float(np.mean(means)),
                        'mean_cfr_payoff': float(-np.mean(means)),
                        'replicate_bootstrap_interval_95': bootstrap_interval(means, 99173 + sum(ord(c) for c in label), bootstrap_samples),
                        'positive_replicate_fraction': float(np.mean(np.asarray(means) > 0))}
    return {'status': 'completed', 'game': 'two-player heads-up limit Leduc',
            'design': {'cfr_iterations': iterations, 'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
                       'temperature': temperature, 'bootstrap_samples': bootstrap_samples,
                       'chance_deals_paired_across_seats': True, 'adaptive_learning_reset_by_replicate_and_seat': True,
                       'fixed_cfr_average_strategy': True},
            'cfr_reference': solver.evaluate(), 'cases': cases,
            'scope': 'Adaptive online PCC versus fixed finite-iteration CFR; seat-paired sampled match payoff, not exploitability or proof of PCC optimality.',
            'limitations': 'Adaptive opponent learning is online within each replicate; sample hands are dependent within replicate. Bootstrap resamples independent paired replicates, not hands. PCC mixture labels are simulator-defined.',
            'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2101, 2102, 2103, 2104, 2105])
    parser.add_argument('--hands-per-seat', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.35)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output', default='validation/adaptive-cfr-tournament.json')
    args = parser.parse_args(argv)
    path = Path(args.output)
    if path.exists():
        parser.error('refusing to overwrite existing report: ' + str(path))
    report = tournament(args.iterations, args.seeds, args.hands_per_seat, args.temperature, args.bootstrap_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'cases': {k: {'mean_pcc_payoff': v['mean_pcc_payoff'], 'interval_95': v['replicate_bootstrap_interval_95']} for k, v in report['cases'].items()}, 'output': str(path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
