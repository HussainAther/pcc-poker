"""Label-free behavioral measurements from public state and counterfactual value."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import math

from .engine import RANKS, State, apply_action, utility


def _record_context(record: dict) -> tuple:
    return (
        int(record["actor"]),
        int(record["round_index"]),
        bool(record["to_call"] > 0),
        int(record["pot"]),
        tuple(record["legal_actions"]),
    )


def _state_context(state: State) -> tuple:
    return (
        state.actor,
        state.round_index,
        bool(state.to_call > 0),
        state.pot,
        state.legal_actions(),
    )


def _information_record_context(record: dict) -> tuple:
    """Public context used to ask whether private information improves prediction."""
    return (
        int(record["actor"]),
        int(record["round_index"]),
        record.get("public_rank"),
        int(record["to_call"]),
        int(record["pot"]),
        tuple(record["legal_actions"]),
    )


def _information_state_context(state: State) -> tuple:
    return (
        state.actor,
        state.round_index,
        state.public,
        state.to_call,
        state.pot,
        state.legal_actions(),
    )


class PublicActionModel:
    """Smoothed action frequencies conditioned only on public betting context."""

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.smoothing = smoothing
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.information_public_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.information_private_counts: dict[tuple, Counter] = defaultdict(Counter)

    @classmethod
    def from_records(
        cls, records: list[dict], smoothing: float = 1.0
    ) -> "PublicActionModel":
        model = cls(smoothing)
        for record in records:
            model.counts[_record_context(record)][record["action"]] += 1
            if "private_rank" in record:
                public_context = _information_record_context(record)
                model.information_public_counts[public_context][record["action"]] += 1
                private_context = (public_context, int(record["private_rank"]))
                model.information_private_counts[private_context][record["action"]] += 1
        return model

    def probabilities(self, state: State) -> dict[str, float]:
        legal = state.legal_actions()
        counts = self.counts[_state_context(state)]
        denominator = sum(counts[action] + self.smoothing for action in legal)
        return {
            action: (counts[action] + self.smoothing) / denominator
            for action in legal
        }

    def information_probabilities(
        self, state: State, prior_strength: float = 3.0
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return public and own-card-conditioned action probabilities.

        The conditioned estimate shrinks toward the public estimate. It uses the
        acting player's own card, never the simulator's opponent card or deck.
        """
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        legal = state.legal_actions()
        public_context = _information_state_context(state)
        public_counts = self.information_public_counts[public_context]
        public_total = sum(public_counts[action] + self.smoothing for action in legal)
        public = {
            action: (public_counts[action] + self.smoothing) / public_total
            for action in legal
        }
        private_counts = self.information_private_counts[
            (public_context, state.private[state.actor])
        ]
        private_total = sum(private_counts[action] for action in legal)
        conditioned = {
            action: (
                private_counts[action] + prior_strength * public[action]
            ) / (private_total + prior_strength)
            for action in legal
        }
        return public, conditioned


def information_states(state: State, observer: int) -> list[tuple[State, float]]:
    """Enumerate states consistent with the observer's cards and public history.

    Only the acting player's private rank and the public rank are retained. The
    simulator's actual opponent card and unrevealed deck order are discarded.
    """
    if observer not in (0, 1):
        raise ValueError("observer must be player 0 or 1")
    own = state.private[observer]
    remaining = Counter({rank: 2 for rank in RANKS})
    remaining[own] -= 1
    if state.public is not None:
        remaining[state.public] -= 1

    possibilities = []
    cards_left = sum(remaining.values())
    opponent_index = 1 - observer
    for opponent_rank in RANKS:
        opponent_count = remaining[opponent_rank]
        if opponent_count <= 0:
            continue
        opponent_probability = opponent_count / cards_left
        after_opponent = remaining.copy()
        after_opponent[opponent_rank] -= 1
        private = list(state.private)
        private[observer] = own
        private[opponent_index] = opponent_rank

        if state.public is not None:
            concrete = replace(state, private=tuple(private), deck=())
            possibilities.append((concrete, opponent_probability))
            continue

        future_total = cards_left - 1
        for public_rank in RANKS:
            public_count = after_opponent[public_rank]
            if public_count <= 0:
                continue
            probability = opponent_probability * public_count / future_total
            concrete = replace(
                state,
                private=tuple(private),
                deck=(public_rank,),
            )
            possibilities.append((concrete, probability))

    total = sum(probability for _, probability in possibilities)
    return [(concrete, probability / total) for concrete, probability in possibilities]


@dataclass(frozen=True)
class BehavioralMeasurement:
    action_values: dict[str, float]
    chosen_value: float
    best_value: float
    regret: float
    control_efficiency: float
    public_action_probability: float
    private_conditioned_action_probability: float
    private_information_gain: float
    predictive_control: float
    response_entropy: float
    response_entropy_normalized: float
    response_compression: float
    predicted_fold_probability: float
    commitment_ratio: float
    pressure_index: float
    action_probability: float
    action_surprisal: float
    effective_surprisal: float

    def as_dict(self) -> dict:
        return {
            "action_values": self.action_values,
            "chosen_value": self.chosen_value,
            "best_value": self.best_value,
            "regret": self.regret,
            "control_efficiency": self.control_efficiency,
            "public_action_probability": self.public_action_probability,
            "private_conditioned_action_probability": self.private_conditioned_action_probability,
            "private_information_gain": self.private_information_gain,
            "predictive_control": self.predictive_control,
            "response_entropy": self.response_entropy,
            "response_entropy_normalized": self.response_entropy_normalized,
            "response_compression": self.response_compression,
            "predicted_fold_probability": self.predicted_fold_probability,
            "commitment_ratio": self.commitment_ratio,
            "pressure_index": self.pressure_index,
            "action_probability": self.action_probability,
            "action_surprisal": self.action_surprisal,
            "effective_surprisal": self.effective_surprisal,
        }


class CounterfactualOracle:
    """Exact Leduc action values under a fixed public continuation model."""

    def __init__(self, action_model: PublicActionModel) -> None:
        self.action_model = action_model

    @lru_cache(maxsize=None)
    def _continuation_value(self, state: State, observer: int) -> float:
        if state.terminal:
            return utility(state, observer)
        probabilities = self.action_model.probabilities(state)
        return sum(
            probability
            * self._continuation_value(apply_action(state, action), observer)
            for action, probability in probabilities.items()
        )

    def action_values(self, state: State) -> dict[str, float]:
        observer = state.actor
        possibilities = information_states(state, observer)
        return {
            action: sum(
                probability
                * self._continuation_value(
                    apply_action(concrete, action), observer
                )
                for concrete, probability in possibilities
            )
            for action in state.legal_actions()
        }

    def _response_measurements(self, state: State, action: str) -> tuple[float, ...]:
        observer = state.actor
        next_state = apply_action(state, action)
        if next_state.terminal or next_state.actor == observer:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        probabilities = self.action_model.probabilities(next_state)
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in probabilities.values()
        )
        maximum_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        normalized = entropy / maximum_entropy if maximum_entropy else 0.0
        compression = 1.0 - normalized if len(probabilities) > 1 else 1.0
        faces_wager = next_state.to_call > 0
        fold_probability = probabilities.get("fold", 0.0) if faces_wager else 0.0
        commitment = (
            next_state.to_call / max(next_state.pot + next_state.to_call, 1)
            if faces_wager else 0.0
        )
        pressure = (
            (compression + fold_probability + commitment) / 3
            if faces_wager else 0.0
        )
        return entropy, normalized, compression, fold_probability, commitment, pressure

    def measure(self, state: State, chosen_action: str) -> BehavioralMeasurement:
        if chosen_action not in state.legal_actions():
            raise ValueError("chosen action must be legal in the supplied state")
        values = self.action_values(state)
        chosen_value = values[chosen_action]
        best_value = max(values.values())
        regret = max(best_value - chosen_value, 0.0)
        payoff_scale = max(state.pot + state.bet_size, 1)
        control = math.exp(-regret / payoff_scale)

        public_information_probabilities, private_probabilities = (
            self.action_model.information_probabilities(state)
        )
        public_action_probability = public_information_probabilities[chosen_action]
        private_action_probability = private_probabilities[chosen_action]
        information_gain = math.log(
            max(private_action_probability, 1e-12)
            / max(public_action_probability, 1e-12)
        )
        predictive_control = 1.0 - math.exp(-max(information_gain, 0.0))

        action_probability = self.action_model.probabilities(state)[chosen_action]
        surprisal = -math.log(max(action_probability, 1e-12))
        effective_surprisal = surprisal * control

        entropy, normalized, compression, fold_probability, commitment, pressure = (
            self._response_measurements(state, chosen_action)
        )
        return BehavioralMeasurement(
            action_values=values,
            chosen_value=chosen_value,
            best_value=best_value,
            regret=regret,
            control_efficiency=control,
            public_action_probability=public_action_probability,
            private_conditioned_action_probability=private_action_probability,
            private_information_gain=information_gain,
            predictive_control=predictive_control,
            response_entropy=entropy,
            response_entropy_normalized=normalized,
            response_compression=compression,
            predicted_fold_probability=fold_probability,
            commitment_ratio=commitment,
            pressure_index=pressure,
            action_probability=action_probability,
            action_surprisal=surprisal,
            effective_surprisal=effective_surprisal,
        )
