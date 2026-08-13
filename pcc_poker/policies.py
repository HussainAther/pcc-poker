"""PCC action-selection objectives for Leduc poker."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random

import numpy as np

from .engine import State, equity


MODES = ("pressure", "control", "chaos")


class OpponentModel:
    def __init__(self) -> None:
        self.context_actions: dict[str, Counter] = defaultdict(Counter)

    def context(self, state: State) -> str:
        facing = "facing" if state.to_call else "open"
        return f"r{state.round_index}|{facing}"

    def observe(self, state: State, action: str) -> None:
        self.context_actions[self.context(state)][action] += 1

    def probability(self, state: State, action: str, legal: tuple[str, ...] | None = None) -> float:
        actions = legal or state.legal_actions()
        counts = self.context_actions[self.context(state)]
        total = sum(counts[item] + 1 for item in actions)
        return (counts[action] + 1) / total

    def fold_probability(self, state: State) -> float:
        counts = self.context_actions[f"r{state.round_index}|facing"]
        return (counts["fold"] + 1) / (sum(counts.values()) + 3)


@dataclass(frozen=True)
class Decision:
    action: str
    probabilities: dict[str, float]
    component_scores: dict[str, dict[str, float]]
    weights: dict[str, float]
    equity: float


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    values = np.array([scores[action] for action in scores], dtype=float) / max(temperature, 1e-4)
    values -= values.max(); exp = np.exp(values); exp /= exp.sum()
    return dict(zip(scores, exp.tolist()))


def component_scores(
    state: State,
    opponent_model: OpponentModel,
    actor_history: OpponentModel | None = None,
) -> dict[str, dict[str, float]]:
    legal = state.legal_actions(); eq = equity(state, state.actor)
    pot = max(state.pot, 1); call_cost = state.to_call
    actor_history = actor_history or OpponentModel()
    baseline = {action: actor_history.probability(state, action, legal) for action in legal}
    scores = {mode: {} for mode in MODES}
    for action in legal:
        aggressive = action in {"bet", "raise"}
        passive = action in {"check", "call"}
        commitment = (state.bet_size + call_cost) / pot if aggressive else call_cost / pot if action == "call" else 0
        fold_leverage = opponent_model.fold_probability(state) if aggressive else 0
        scores["pressure"][action] = 1.3 * aggressive + fold_leverage + 0.45 * commitment - 0.4 * (action == "fold")

        pot_odds = call_cost / (pot + call_cost) if call_cost else 0
        showdown_value = eq - pot_odds
        robust = showdown_value if action in {"call", "check"} else (2 * eq - 1) + 0.55 * fold_leverage
        risk = commitment * (1 - eq)
        flexibility = 0.25 if passive and action != "fold" else 0
        scores["control"][action] = robust + flexibility - 0.8 * risk

        surprise = -math.log(max(baseline[action], 1e-9))
        performance_floor = max(-0.75, robust)
        scores["chaos"][action] = surprise + 0.65 * performance_floor - 1.2 * (action == "fold" and eq > 0.6)
    return scores


class PCCPolicy:
    def __init__(
        self,
        weights: tuple[float, float, float],
        seed: int = 0,
        temperature: float = 0.35,
        label: str | None = None,
    ) -> None:
        values = np.asarray(weights, dtype=float)
        if values.shape != (3,) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("PCC weights must be three nonnegative values with positive sum")
        self.weights = values / values.sum()
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.label = label or MODES[int(self.weights.argmax())]
        self.opponent_model = OpponentModel()
        self.action_history = OpponentModel()

    def decide(self, state: State) -> Decision:
        components = component_scores(state, self.opponent_model, self.action_history)
        combined = {
            action: sum(self.weights[index] * components[mode][action] for index, mode in enumerate(MODES))
            for action in state.legal_actions()
        }
        probabilities = _softmax(combined, self.temperature)
        threshold = self.rng.random(); cumulative = 0.0; selected = next(iter(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if threshold <= cumulative:
                selected = action; break
        self.action_history.observe(state, selected)
        return Decision(
            selected, probabilities, components,
            dict(zip(MODES, self.weights.tolist())), equity(state, state.actor),
        )


PURE_MIXTURES = {
    "pressure": (0.8, 0.1, 0.1),
    "control": (0.1, 0.8, 0.1),
    "chaos": (0.1, 0.1, 0.8),
    "balanced": (1 / 3, 1 / 3, 1 / 3),
}
