"""Prospective held-out comparison of public decision proxies and sequence features.

This is NOT a CFR solver or an exploitability estimator. Pot odds are public
normative proxies, not an optimal policy; target mixture weights are synthetic.
Training comes only from the pre-existing mixed dataset, evaluation from fresh
seat-yoked simulations. No hidden oracle columns enter the predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import load_jsonl
from .context_matched_transition_validation import make_examples
from .context_yoked_transition_validation import attach_context_features
from .mixed import _metrics, _ridge_predict
from .mixed_ood_ablation import _nested_features
from .paired_matched_opportunity_validation import attach_paired_features
from .seat_yoked_transition_validation import generate_seat_yoked_evaluation

DEFAULT_SEEDS = (1601, 1602, 1603, 1604, 1605)
MODEL_KEYS = ('public_decision_proxy', 'M0', 'M1', 'M2', 'M3', 'M2_plus_matched_transitions')


def public_decision_proxy(rows: list[dict]) -> np.ndarray:
    """Public-only pot-odds and betting exposure, never private value/equity."""
    if not rows:
        raise ValueError('empty example')
    calls = np.array([max(float(r['to_call']), 0.0) for r in rows])
    pots = np.array([max(float(r['pot']), 0.0) for r in rows])
    odds = np.divide(calls, pots + calls, out=np.zeros_like(calls), where=(pots + calls) > 0)
    facing = calls > 0
    actions = [r['action'] for r in rows]
    return np.asarray([
        float(np.mean(facing)),
        float(np.mean(odds)),
        float(np.mean(odds[facing])) if np.any(facing) else 0.0,
        float(np.mean(np.asarray(actions) == 'fold')),
        float(np.mean(np.asarray(actions) == 'call')),
        float(np.mean(np.asarray(actions) == 'raise')),
        float(np.mean(np.asarray(actions) == 'bet')),
        float(np.mean([int(r['round_index']) for r in rows])),
        float(int(rows[0]['focal_seat'])),
    ], dtype=float)


def attach_baselines(examples: list[dict]) -> None:
    """Produce fixed-dimensional, observable-only feature sets."""
    attach_paired_features(examples)
    attach_context_features(examples)
    for ex in examples:
        blocks = _nested_features(ex['rows'])
        ex['public_decision_proxy'] = public_decision_proxy(ex['rows'])
        for name in ('M0', 'M1', 'M2', 'M3'):
            ex[name] = blocks[name]
        # Existing M2 and public-context-matched UNORDERED action pairs.
        ex['M2_plus_matched_transitions'] = ex['context_matched_features']


def _evaluate(train: list[dict], test: list[dict]) -> dict:
    truth = np.stack([e['target'] for e in test])
    groups = [str(e['mixture_id']) for e in test]
    if set(groups) & {str(e['mixture_id']) for e in train}:
        raise ValueError('training and evaluation mixture IDs overlap')
    result = {}
    for key in MODEL_KEYS:
        predicted = _ridge_predict(train, test, key)
        result[key] = _metrics(truth, predicted)
    return result


def run(training_records: list[dict], seeds=DEFAULT_SEEDS,
        mixtures_per_seed: int = 20, hands_per_seat: int = 100) -> dict:
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be nonempty and unique')
    if mixtures_per_seed < 1 or hands_per_seat < 1:
        raise ValueError('mixtures_per_seed and hands_per_seat must be positive')
    training = make_examples(training_records)
    if not training:
        raise ValueError('training examples are empty')
    attach_baselines(training)
    seed_results = []
    for seed in seeds:
        records, design = generate_seat_yoked_evaluation(
            seed=seed, mixtures=mixtures_per_seed, hands_per_seat=hands_per_seat)
        evaluation = make_examples(records)
        if not evaluation:
            raise ValueError(f'no evaluation examples for seed {seed}')
        attach_baselines(evaluation)
        seed_results.append({'seed': seed, 'design': design,
                             'models': _evaluate(training, evaluation)})
    summary = {}
    for name in MODEL_KEYS:
        errors = [result['models'][name]['mae'] for result in seed_results]
        summary[name] = {'mean_mae': float(np.mean(errors)),
                         'seed_mae': [float(e) for e in errors]}
    improvements = [row['models']['M2']['mae'] - row['models']['M2_plus_matched_transitions']['mae']
                    for row in seed_results]
    summary['incremental_matched_transition_vs_M2'] = {
        'mean_mae_reduction': float(np.mean(improvements)),
        'seed_mae_reductions': [float(x) for x in improvements],
        'positive_seed_fraction': float(np.mean(np.array(improvements) > 0)),
    }
    return {
        'status': 'completed', 'analysis_status': 'public_proxy_held_out_synthetic_baselines',
        'seeds': list(seeds), 'mixtures_per_seed': mixtures_per_seed,
        'hands_per_seat': hands_per_seat, 'seed_results': seed_results, 'summary': summary,
        'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False,
                       'private_rank_used_as_predictor': False, 'mixture_ids_disjoint': True},
        'limitations': ('No CFR, regret minimization, exploitability, or causal identification. '
                        'Pot-odds features are decision-theoretic proxies only. '
                        'Matched transitions use action-dependent selection; synthetic labels only; '
                        'five seeds are insufficient for strong inferential claims.'),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', default='outputs/mixed-recovery-data.jsonl')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--mixtures-per-seed', type=int, default=20)
    parser.add_argument('--hands-per-seat', type=int, default=100)
    parser.add_argument('--output', default='validation/game-theoretic-baselines.json')
    args = parser.parse_args(argv)
    report = run(load_jsonl(args.training), args.seeds, args.mixtures_per_seed, args.hands_per_seat)
    target = Path(args.output)
    training_path = Path(args.training).resolve()
    if target.resolve() == training_path:
        raise ValueError('output must not overwrite training data')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'summary': report['summary'], 'output': str(target)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
