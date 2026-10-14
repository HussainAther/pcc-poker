"""Prospective, frozen-design replication of the Balanced PCC factorial interaction.

Primary: Balanced PCC against CFR-2000. Secondary: CFR-500. Neither
policy parameters nor contrast definitions are adapted to evaluation outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .adaptive_cfr_ablations import ABLATIONS, ablation_replicate
from .adaptation_interaction import paired_interval
from .engine import all_deals
from .leduc_cfr import LeducCFR
from .policies import PURE_MIXTURES

PRIMARY_BUDGET = 2000
SECONDARY_BUDGET = 500
DEFAULT_SEEDS = tuple(range(2401, 2421))
TEMPERATURE = 0.35
PRIMARY_CONTRAST = 'adaptive - opponent_only - history_only + frozen'


def _mean(values):
    return sum(values) / len(values)


def _summarize(values, *, bootstrap_seed, bootstrap_samples):
    return {'mean': _mean(values), 'seed_values': values,
            'paired_replicate_interval_95': paired_interval(values, samples=bootstrap_samples, seed=bootstrap_seed),
            'positive_replicate_fraction': sum(x > 0 for x in values) / len(values)}


def experiment(*, seeds=DEFAULT_SEEDS, hands_per_seat=200, bootstrap_samples=2000, bootstrap_seed=2400):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds):
        raise ValueError('at least two unique nonnegative integer replicate seeds required')
    if hands_per_seat < 1 or bootstrap_samples < 1:
        raise ValueError('hands and bootstrap samples must be positive')
    # Predeclare analyses before looking at any replicate payoff.
    design = {'primary_case': 'balanced', 'primary_cfr_iterations': PRIMARY_BUDGET,
              'secondary_cfr_iterations': SECONDARY_BUDGET, 'primary_contrast': PRIMARY_CONTRAST,
              'seeds': list(seeds), 'hands_per_seat': hands_per_seat,
              'temperature': TEMPERATURE, 'bootstrap_samples': bootstrap_samples,
              'bootstrap_seed': bootstrap_seed, 'paired_chance_deals': True,
              'separate_agent_state_per_condition_and_seat': True,
              'same_seed_across_conditions': True, 'cfr_fixed_during_match': True,
              'no_parameter_tuning_on_test_replicates': True}
    digest = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    decks = all_deals()
    schedules = {}
    for seed in seeds:
        rng = random.Random(seed)
        schedules[seed] = [decks[rng.randrange(len(decks))] for _ in range(hands_per_seat)]
    solver = LeducCFR()
    previous = 0
    outputs = {}
    for budget in (SECONDARY_BUDGET, PRIMARY_BUDGET):
        solver.train(budget - previous)
        previous = budget
        strategy = solver.average_strategy()
        reps = []
        for seed in seeds:
            conditions = ablation_replicate(PURE_MIXTURES['balanced'], strategy, schedules[seed], seed, TEMPERATURE)
            vals = {k: conditions[k]['seat_average_payoff'] for k in ABLATIONS}
            reps.append({'seed': seed, 'conditions': vals,
                         'primary_interaction': vals['adaptive'] - vals['opponent_only'] - vals['history_only'] + vals['frozen'],
                         'full_vs_frozen': vals['adaptive'] - vals['frozen']})
        interaction = [r['primary_interaction'] for r in reps]
        full = [r['full_vs_frozen'] for r in reps]
        adaptive = [r['conditions']['adaptive'] for r in reps]
        outputs[str(budget)] = {
            'role': 'primary' if budget == PRIMARY_BUDGET else 'secondary',
            'interaction': _summarize(interaction, bootstrap_seed=bootstrap_seed + budget + 1, bootstrap_samples=bootstrap_samples),
            'full_vs_frozen': _summarize(full, bootstrap_seed=bootstrap_seed + budget + 2, bootstrap_samples=bootstrap_samples),
            'absolute_adaptive_payoff': _summarize(adaptive, bootstrap_seed=bootstrap_seed + budget + 3, bootstrap_samples=bootstrap_samples),
            'replicates': reps}
    return {'status': 'completed', 'analysis_status': 'prospective_balanced_adaptation_replication',
            'design': design, 'design_sha256': digest, 'results': outputs,
            'interpretation_rule': 'Primary contrast is 2000-iteration factorial interaction; evaluate its paired replicate interval before secondary findings. No multiplicity-adjusted confirmatory claim.',
            'limitations': 'Replicate bootstrap intervals are descriptive; learning-dependent hands are not independent. CFR baselines share one trained solver; no full adaptive PCC exploitability claim. A post-observation design selected from Patch 13 requires fresh replication and careful reporting.',
            'provenance': {'prior_reports_read_only': True, 'frozen_evidence_modified': False, 'human_data_accessed': False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    parser.add_argument('--hands-per-seat', type=int, default=200)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=2400)
    parser.add_argument('--output', default='validation/prospective-adaptation-replication.json')
    args = parser.parse_args(argv)
    path = Path(args.output)
    if path.exists():
        parser.error('refusing to overwrite existing report: ' + str(path))
    result = experiment(seeds=args.seeds, hands_per_seat=args.hands_per_seat,
                        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'design_sha256': result['design_sha256'],
                      'primary': result['results'][str(PRIMARY_BUDGET)]['interaction'],
                      'secondary': result['results'][str(SECONDARY_BUDGET)]['interaction'],
                      'output': str(path)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
