"""Full-tree CFR and exact information-set best response for the local Leduc engine.

Utilities are net chips per hand.  Exploitability is NashConv / 2 (chips/player).
This module is independent of PCC mixture-recovery labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .engine import all_deals, initial_state, apply_action, utility


@dataclass
class Node:
    state: object
    children: tuple[int, ...]
    key: tuple | None
    actions: tuple[str, ...]
    depth: int


def game_tree():
    nodes = []
    roots = []
    infos = defaultdict(list)
    def visit(state, depth):
        idx = len(nodes)
        nodes.append(None)
        actions = state.legal_actions()
        key = (state.actor, state.information_key(state.actor)) if actions else None
        children = tuple(visit(apply_action(state, a), depth + 1) for a in actions)
        nodes[idx] = Node(state, children, key, actions, depth)
        if key is not None:
            infos[key].append(idx)
        return idx
    for deck in all_deals():
        roots.append(visit(initial_state(deck), 0))
    for key, ids in infos.items():
        if any(nodes[i].actions != nodes[ids[0]].actions for i in ids):
            raise ValueError(f'inconsistent legal actions: {key}')
        if len({nodes[i].depth for i in ids}) != 1:
            raise ValueError(f'information set spans depths: {key}')
    return nodes, roots, infos


def regret_matching(regrets):
    positive = np.maximum(regrets, 0)
    total = float(positive.sum())
    return positive / total if total > 0 else np.full(len(regrets), 1.0 / len(regrets))


class LeducCFR:
    def __init__(self):
        self.nodes, self.roots, self.infos = game_tree()
        self.regrets = {key: np.zeros(len(self.nodes[ids[0]].actions)) for key, ids in self.infos.items()}
        self.strategy_sums = {key: np.zeros_like(v) for key, v in self.regrets.items()}
        self.iterations = 0

    def iterate(self):
        strategy = {k: regret_matching(v) for k, v in self.regrets.items()}
        increments = {k: np.zeros_like(v) for k, v in self.regrets.items()}
        def traverse(i, reach0, reach1, chance):
            node = self.nodes[i]
            if node.key is None:
                return utility(node.state, 0)
            actor = node.state.actor
            probs = strategy[node.key]
            if actor == 0:
                values = np.array([traverse(c, reach0 * p, reach1, chance) for c, p in zip(node.children, probs)])
                expected = float(probs @ values)
                increments[node.key] += chance * reach1 * (values - expected)
                self.strategy_sums[node.key] += chance * reach0 * probs
            else:
                values = np.array([traverse(c, reach0, reach1 * p, chance) for c, p in zip(node.children, probs)])
                expected = float(probs @ values)
                increments[node.key] += chance * reach0 * (expected - values)
                self.strategy_sums[node.key] += chance * reach1 * probs
            return expected
        chance = 1.0 / len(self.roots)
        for i in self.roots:
            traverse(i, 1.0, 1.0, chance)
        for key in self.regrets:
            self.regrets[key] += increments[key]
        self.iterations += 1

    def average_strategy(self):
        return {k: (v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))) for k, v in self.strategy_sums.items()}

    def train(self, n):
        if n < 1:
            raise ValueError('iterations must be positive')
        for _ in range(n):
            self.iterate()

    def profile_value(self, policy):
        def walk(i):
            node = self.nodes[i]
            if node.key is None:
                return utility(node.state, 0)
            return sum(float(p) * walk(c) for p, c in zip(policy[node.key], node.children))
        return sum(walk(i) for i in self.roots) / len(self.roots)

    def best_response_value(self, policy, player):
        """Exact pure infoset best response to opponent's fixed behavioral strategy."""
        if player not in (0, 1):
            raise ValueError('player must be 0 or 1')
        weights = np.zeros(len(self.nodes))
        for root in self.roots:
            weights[root] = 1 / len(self.roots)
        for i, node in enumerate(self.nodes):
            if node.key is None:
                continue
            if node.state.actor == player:
                for c in node.children:
                    weights[c] += weights[i]
            else:
                for p, c in zip(policy[node.key], node.children):
                    weights[c] += weights[i] * p
        values = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            if node.key is None:
                values[i] = utility(node.state, player)
        grouped = defaultdict(list)
        for k, ids in self.infos.items():
            grouped[self.nodes[ids[0]].depth].append((k, ids))
        max_depth = max(n.depth for n in self.nodes)
        # At each depth, opponent nodes are evaluated after all children; player
        # decisions share a single maximizing action over every hidden state.
        for depth in range(max_depth, -1, -1):
            for key, ids in grouped.get(depth, ()):
                actor = key[0]
                if actor == player:
                    score = np.zeros(len(self.nodes[ids[0]].children))
                    for i in ids:
                        for a, c in enumerate(self.nodes[i].children):
                            score[a] += weights[i] * values[c]
                    best = int(np.argmax(score))
                    for i in ids:
                        values[i] = values[self.nodes[i].children[best]]
                else:
                    for i in ids:
                        node = self.nodes[i]
                        values[i] = sum(float(p) * values[c] for p, c in zip(policy[key], node.children))
        return float(sum(values[i] for i in self.roots) / len(self.roots))

    def evaluate(self):
        policy = self.average_strategy()
        value0 = self.profile_value(policy)
        br0 = self.best_response_value(policy, 0)
        br1 = self.best_response_value(policy, 1)
        nash_conv = br0 + br1
        if nash_conv < -1e-8:
            raise ValueError('invalid negative NashConv')
        return {'iterations': self.iterations, 'profile_value_player0': value0,
                'best_response_player0': br0, 'best_response_player1': br1,
                'nash_conv': max(0., nash_conv), 'exploitability': max(0., nash_conv / 2)}


def run(budgets=(1, 10, 50, 100)):
    budgets = tuple(int(n) for n in budgets)
    if not budgets or any(n <= 0 for n in budgets) or sorted(set(budgets)) != list(budgets):
        raise ValueError('budgets must be strictly increasing positive integers')
    solver = LeducCFR()
    results = []
    for budget in budgets:
        solver.train(budget - solver.iterations)
        results.append(solver.evaluate())
    return {'status': 'completed', 'game': 'two-player heads-up limit Leduc',
            'chance_deals': len(solver.roots), 'information_sets': len(solver.infos),
            'metrics_unit': 'net chips per hand', 'exploitability_definition': 'NashConv / 2',
            'checkpoints': results,
            'provenance': {'human_data_accessed': False, 'frozen_evidence_modified': False},
            'limitations': 'Solver benchmark only; not a PCC mixture-recovery predictor. No external solver cross-validation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgets', nargs='+', type=int, default=[1, 10, 50, 100])
    parser.add_argument('--output', default='validation/leduc-cfr-convergence.json')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('refusing to overwrite existing report: ' + str(output))
    report = run(args.budgets)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checkpoints': report['checkpoints'], 'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
