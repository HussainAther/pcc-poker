"""Independent synthetic value floor for validating a Chaos observable.

This module intentionally does not use PCC weights, component scores, policy
labels, learned opponent models, or observed terminal outcomes.  In Leduc it
computes information-set action values under a fixed *uniform legal-action*
continuation model.  That deliberately simple reference is used only to test
whether a surprisal metric can distinguish useful mixing from value-destroying
randomness before any human-data analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from .behavioral import information_states
from .engine import State, apply_action, utility


@lru_cache(maxsize=None)
def _uniform_continuation_value(state: State, observer: int) -> float:
    if state.terminal:
        return utility(state, observer)
    legal = state.legal_actions()
    probability = 1.0 / len(legal)
    return sum(
        probability * _uniform_continuation_value(apply_action(state, action), observer)
        for action in legal
    )


class UniformContinuationValueModel:
    """Exact Leduc Q-values under a fixed, non-PCC continuation reference."""

    def action_values(self, state: State) -> dict[str, float]:
        observer = state.actor
        possibilities = information_states(state, observer)
        return {
            action: sum(
                probability
                * _uniform_continuation_value(apply_action(concrete, action), observer)
                for concrete, probability in possibilities
            )
            for action in state.legal_actions()
        }


@dataclass(frozen=True)
class PerformanceFloor:
    chosen_value: float
    best_value: float
    regret: float
    tolerance: float
    adequacy: float


@dataclass(frozen=True)
class EffectiveChaosCandidate:
    normalized_surprisal: float
    adequacy: float
    effective_surprisal: float
    regret: float
    chosen_value: float
    best_value: float


def performance_floor(values: dict[str, float], chosen_action: str, *, tolerance: float) -> PerformanceFloor:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if chosen_action not in values:
        raise ValueError("chosen_action must have a value")
    chosen = float(values[chosen_action])
    best = max(float(value) for value in values.values())
    regret = max(0.0, best - chosen)
    # Linear, preregistrable floor: full credit for a best action, zero once
    # regret reaches the declared tolerance. No PCC labels enter this mapping.
    adequacy = max(0.0, 1.0 - regret / tolerance)
    return PerformanceFloor(chosen, best, regret, float(tolerance), adequacy)


def effective_chaos_candidate(
    normalized_surprisal: float,
    values: dict[str, float],
    chosen_action: str,
    *,
    tolerance: float,
) -> EffectiveChaosCandidate:
    if not math.isfinite(normalized_surprisal) or normalized_surprisal < 0:
        raise ValueError("normalized_surprisal must be finite and nonnegative")
    floor = performance_floor(values, chosen_action, tolerance=tolerance)
    return EffectiveChaosCandidate(
        normalized_surprisal=float(normalized_surprisal),
        adequacy=floor.adequacy,
        effective_surprisal=float(normalized_surprisal) * floor.adequacy,
        regret=floor.regret,
        chosen_value=floor.chosen_value,
        best_value=floor.best_value,
    )


def measure_synthetic_effective_chaos(
    state: State,
    chosen_action: str,
    normalized_surprisal: float,
    *,
    tolerance: float | None = None,
    value_model: UniformContinuationValueModel | None = None,
) -> EffectiveChaosCandidate:
    """Score one synthetic Leduc decision with an independent value floor."""
    if chosen_action not in state.legal_actions():
        raise ValueError("chosen_action must be legal")
    model = value_model or UniformContinuationValueModel()
    values = model.action_values(state)
    # Default scale is fixed from public game state, not fitted to PCC labels.
    if tolerance is None:
        tolerance = float(max(state.pot + state.bet_size, 1))
    return effective_chaos_candidate(
        normalized_surprisal, values, chosen_action, tolerance=tolerance
    )
