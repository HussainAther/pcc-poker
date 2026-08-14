"""Label-free public-state PCC candidate observables for future human poker work.

These measurements are deliberately conservative.  They operate only on
``PublicDecisionState`` objects reconstructed from sanitized PHH records and do
not use hidden policy labels, private cards, outcomes, source identifiers, or
future actions.

The quantities are *candidate behavioral observables*, not ground-truth PCC
labels:

* pressure_index: observable commitment/escalation imposed by the chosen action;
* history_alignment: how much a frozen history-conditioned action model improves
  the probability assigned to the chosen action over a static public-state model;
* behavioral_surprisal: surprise of the chosen action under the frozen static
  public-state model.

An ``effective_surprisal`` / Chaos construct is intentionally not defined here.
That requires a separately validated value or performance-floor model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .handhq_features import PublicDecisionState


ACTIONS = ("fold", "check_call", "bet_raise")


def _bucket(value: float, scale: float) -> int:
    if scale <= 1e-12:
        return 0
    ratio = max(0.0, value / scale)
    if ratio <= 0.05:
        return 0
    if ratio <= 0.25:
        return 1
    if ratio <= 0.75:
        return 2
    return 3


def _static_context(state: PublicDecisionState) -> tuple:
    scale = max(state.pot_size, state.effective_stack, 1.0)
    return (
        state.street_index,
        state.active_players,
        bool(state.to_call > 1e-12),
        _bucket(state.to_call, scale),
        _bucket(state.pot_size, max(state.effective_stack, 1.0)),
        state.raises_this_street,
        tuple(state.legal_actions),
    )


def _opponent_history_signature(state: PublicDecisionState) -> tuple:
    """Summarize only public actions by players other than the focal actor."""
    counts = Counter()
    last_action = "none"
    last_amount_bucket = 0
    scale = max(state.pot_size, state.effective_stack, 1.0)
    for kind, actor, amount in state.prior_action_sequence:
        if actor is None or actor == state.actor_position:
            continue
        if kind in ACTIONS:
            counts[kind] += 1
            last_action = kind
            last_amount_bucket = _bucket(float(amount or 0.0), scale)
    return (
        min(counts["fold"], 3),
        min(counts["check_call"], 3),
        min(counts["bet_raise"], 3),
        last_action,
        last_amount_bucket,
    )


def _temporal_context(state: PublicDecisionState) -> tuple:
    return (_static_context(state), _opponent_history_signature(state))


def _legal_probability(counts: Counter, legal: Sequence[str], action: str, smoothing: float) -> float:
    if action not in legal:
        return 0.0
    denominator = sum(counts[a] + smoothing for a in legal)
    return (counts[action] + smoothing) / denominator


class FrozenPublicStateActionModel:
    """Smoothed action-frequency models fit on a separate calibration set."""

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.smoothing = float(smoothing)
        self.static_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.temporal_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.global_counts: Counter = Counter()
        self._fitted = False

    @classmethod
    def fit(cls, states: Iterable[PublicDecisionState], smoothing: float = 1.0) -> "FrozenPublicStateActionModel":
        model = cls(smoothing=smoothing)
        count = 0
        for state in states:
            action = state.observed_action
            if action not in ACTIONS:
                continue
            model.static_counts[_static_context(state)][action] += 1
            model.temporal_counts[_temporal_context(state)][action] += 1
            model.global_counts[action] += 1
            count += 1
        if count == 0:
            raise ValueError("calibration set contains no supported decisions")
        model._fitted = True
        return model

    def _probability(self, state: PublicDecisionState, *, temporal: bool) -> float:
        if not self._fitted:
            raise RuntimeError("model must be fitted before evaluation")
        action = state.observed_action
        legal = state.legal_actions
        context = _temporal_context(state) if temporal else _static_context(state)
        table = self.temporal_counts if temporal else self.static_counts
        counts = table.get(context)
        # Deterministic backoff prevents zero-data contexts from turning the
        # evaluation set into accidental model training.
        if not counts or sum(counts.values()) == 0:
            counts = self.global_counts
        return _legal_probability(counts, legal, action, self.smoothing)

    def static_probability(self, state: PublicDecisionState) -> float:
        return self._probability(state, temporal=False)

    def temporal_probability(self, state: PublicDecisionState) -> float:
        return self._probability(state, temporal=True)


@dataclass(frozen=True)
class HumanPCCObservable:
    pressure_index: float
    commitment_fraction: float
    escalation_indicator: float
    behavioral_surprisal: float
    normalized_surprisal: float
    static_action_probability: float
    temporal_action_probability: float
    history_alignment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pressure_index": self.pressure_index,
            "commitment_fraction": self.commitment_fraction,
            "escalation_indicator": self.escalation_indicator,
            "behavioral_surprisal": self.behavioral_surprisal,
            "normalized_surprisal": self.normalized_surprisal,
            "static_action_probability": self.static_action_probability,
            "temporal_action_probability": self.temporal_action_probability,
            "history_alignment": self.history_alignment,
        }


def _pressure_measure(state: PublicDecisionState) -> tuple[float, float, float]:
    action = state.observed_action
    if action != "bet_raise":
        return 0.0, 0.0, 0.0
    target = float(state.observed_amount or state.actor_street_contribution)
    incremental = max(0.0, target - state.actor_street_contribution)
    denominator = max(state.effective_stack, state.actor_stack_remaining, 1e-12)
    commitment = min(1.0, incremental / denominator)
    escalation = 1.0 if target > state.current_bet + 1e-12 else 0.0
    # Equal-weight descriptive index. It is intentionally transparent rather
    # than tuned against PCC labels or human outcomes.
    pressure = 0.5 * commitment + 0.5 * escalation
    return pressure, commitment, escalation


def measure_observable(state: PublicDecisionState, model: FrozenPublicStateActionModel) -> HumanPCCObservable:
    """Measure one evaluation decision using a previously frozen model."""
    p_static = max(model.static_probability(state), 1e-12)
    p_temporal = max(model.temporal_probability(state), 1e-12)
    surprisal = -math.log(p_static)
    max_surprisal = math.log(max(len(state.legal_actions), 1))
    normalized = surprisal / max_surprisal if max_surprisal > 1e-12 else 0.0
    pressure, commitment, escalation = _pressure_measure(state)
    return HumanPCCObservable(
        pressure_index=pressure,
        commitment_fraction=commitment,
        escalation_indicator=escalation,
        behavioral_surprisal=surprisal,
        normalized_surprisal=normalized,
        static_action_probability=p_static,
        temporal_action_probability=p_temporal,
        history_alignment=math.log(p_temporal) - math.log(p_static),
    )


def evaluate_observables(
    calibration_states: Iterable[PublicDecisionState],
    evaluation_states: Iterable[PublicDecisionState],
    *,
    smoothing: float = 1.0,
) -> tuple[HumanPCCObservable, ...]:
    """Fit on calibration decisions and score a disjoint evaluation iterable."""
    model = FrozenPublicStateActionModel.fit(calibration_states, smoothing=smoothing)
    return tuple(measure_observable(state, model) for state in evaluation_states)
