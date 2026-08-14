"""Independent policy mechanisms for cross-family PCC transfer tests.

These policies do not call ``policies.component_scores`` and do not combine PCC
scores. Each mechanism first creates its own action distribution; continuous
mixture weights then combine those distributions at the probability level.
"""

from __future__ import annotations

import math
import random

import numpy as np

from .engine import State, equity
from .policies import Decision, MODES, OpponentModel


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in values.values())
    if total <= 1e-12:
        return {action: 1 / len(values) for action in values}
    return {action: max(value, 0.0) / total for action, value in values.items()}


def _softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    actions = tuple(values)
    scores = np.asarray([values[action] for action in actions], dtype=float)
    scores = scores / max(temperature, 1e-4)
    scores -= scores.max()
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum()
    return dict(zip(actions, probabilities.tolist()))


class IndependentMixturePolicy:
    """Blend separately normalized coercive, value, and novelty mechanisms."""

    family_name = "independent_probability_mixture"

    def __init__(
        self,
        weights: tuple[float, float, float],
        seed: int = 0,
        temperature: float = 0.35,
        label: str | None = None,
    ) -> None:
        values = np.asarray(weights, dtype=float)
        if values.shape != (3,) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("weights must be three nonnegative values")
        self.weights = values / values.sum()
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.label = label or f"independent-{int(self.weights.argmax())}"
        self.opponent_model = OpponentModel()
        self.action_history = OpponentModel()

    def _coercive_distribution(self, state: State) -> dict[str, float]:
        """Prioritize initiative and opponent fold leverage, independent of cards."""
        fold_leverage = self.opponent_model.fold_probability(state)
        scores = {}
        for action in state.legal_actions():
            if action in {"bet", "raise"}:
                scores[action] = 1.6 + fold_leverage
            elif action == "call":
                scores[action] = 0.25
            elif action == "check":
                scores[action] = 0.0
            else:
                scores[action] = -1.0
        return _softmax(scores, self.temperature)

    def _action_value(self, state: State, action: str) -> float:
        """One-step expected-chip proxy used only by the value mechanism."""
        eq = equity(state, state.actor)
        pot = max(state.pot, 1)
        call_cost = state.to_call
        fold_probability = self.opponent_model.fold_probability(state)
        if action == "fold":
            return 0.0
        if action == "check":
            return (2 * eq - 1) * pot
        if action == "call":
            return eq * (pot + call_cost) - call_cost
        commitment = call_cost + state.bet_size
        contested = (2 * eq - 1) * (pot + commitment)
        return fold_probability * pot + (1 - fold_probability) * contested

    def _value_distribution(self, state: State) -> dict[str, float]:
        values = {
            action: self._action_value(state, action)
            for action in state.legal_actions()
        }
        return _softmax(values, max(self.temperature * 1.5, 0.1))

    def _novelty_distribution(self, state: State) -> dict[str, float]:
        """Favor rare actions only when near the best one-step value."""
        legal = state.legal_actions()
        action_values = {
            action: self._action_value(state, action) for action in legal
        }
        best_value = max(action_values.values())
        tolerance = max(0.75, 0.2 * state.pot)
        novelty = {}
        for action in legal:
            history_probability = self.action_history.probability(
                state, action, legal
            )
            surprise = -math.log(max(history_probability, 1e-9))
            value_gap = max(best_value - action_values[action], 0.0)
            eligible = value_gap <= tolerance
            novelty[action] = math.exp(surprise) if eligible else 0.02
        return _normalize(novelty)

    def decide(self, state: State) -> Decision:
        component_probabilities = {
            "pressure": self._coercive_distribution(state),
            "control": self._value_distribution(state),
            "chaos": self._novelty_distribution(state),
        }
        combined = {
            action: sum(
                self.weights[index]
                * component_probabilities[mode].get(action, 0.0)
                for index, mode in enumerate(MODES)
            )
            for action in state.legal_actions()
        }
        probabilities = _normalize(combined)
        threshold = self.rng.random()
        cumulative = 0.0
        selected = next(iter(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if threshold <= cumulative:
                selected = action
                break
        self.action_history.observe(state, selected)
        return Decision(
            action=selected,
            probabilities=probabilities,
            component_scores=component_probabilities,
            weights=dict(zip(MODES, self.weights.tolist())),
            equity=equity(state, state.actor),
        )


class AdaptiveMixturePolicy(IndependentMixturePolicy):
    """PCC family whose Control component explicitly exploits learned responses.

    Pressure remains broadly aggressive and Chaos remains novelty-seeking. The
    Control component changes its open aggression with the opponent's observed
    round-specific fold rate, while retaining a card-value safety constraint.
    """

    family_name = "adaptive_response_mixture"

    def _adaptive_control_distribution(self, state: State) -> dict[str, float]:
        legal = state.legal_actions()
        eq = equity(state, state.actor)
        fold_probability = self.opponent_model.fold_probability(state)
        scores = {}
        for action in legal:
            value = self._action_value(state, action)
            scaled_value = value / max(state.pot + state.bet_size, 1)
            if action in {"bet", "raise"}:
                # Time aggression to learned fold vulnerability. Weak hands may
                # exploit a reliable fold, while strong hands still value-build.
                response_timing = 4.0 * (fold_probability - 1 / 3)
                card_safety = 0.8 * (2 * eq - 1)
                scores[action] = scaled_value + response_timing + card_safety
            elif action in {"check", "call"}:
                # Preserve optionality when the opponent has resisted pressure.
                resistance = 1.2 * (1.0 - fold_probability)
                scores[action] = scaled_value + resistance + 0.35 * eq
            else:
                scores[action] = scaled_value + 0.8 * (1.0 - eq)
        return _softmax(scores, max(self.temperature * 1.25, 0.12))

    def decide(self, state: State) -> Decision:
        component_probabilities = {
            "pressure": self._coercive_distribution(state),
            "control": self._adaptive_control_distribution(state),
            "chaos": self._novelty_distribution(state),
        }
        combined = {
            action: sum(
                self.weights[index]
                * component_probabilities[mode].get(action, 0.0)
                for index, mode in enumerate(MODES)
            )
            for action in state.legal_actions()
        }
        probabilities = _normalize(combined)
        threshold = self.rng.random()
        cumulative = 0.0
        selected = next(iter(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if threshold <= cumulative:
                selected = action
                break
        self.action_history.observe(state, selected)
        return Decision(
            action=selected,
            probabilities=probabilities,
            component_scores=component_probabilities,
            weights=dict(zip(MODES, self.weights.tolist())),
            equity=equity(state, state.actor),
        )
