"""Exploratory online PCC regime evidence for interactive Leduc play.

This module is deliberately conservative: it emits *-LIKE* behavioral regime
labels, not ground-truth latent PCC states.  Scores are transparent rolling
features computed at the human decision point and do not use the opponent's
private card, terminal payoff, or future actions.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Deque

from .engine import State, equity


REGIMES = ("PRESSURE-LIKE", "CHAOS-LIKE", "CONTROL-LIKE", "MIXED")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = [count / total for count in counts.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    maximum = math.log(len(counts))
    return entropy / maximum if maximum > 0 else 0.0


@dataclass(frozen=True)
class DecisionEvidence:
    pressure: float
    chaos: float
    control: float
    label: str
    faced_pressure: float
    initiative: float
    action_surprise: float
    action_diversity: float
    adequacy_proxy: float
    leverage: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "pressure": self.pressure,
            "chaos": self.chaos,
            "control": self.control,
            "label": self.label,
            "faced_pressure": self.faced_pressure,
            "initiative": self.initiative,
            "action_surprise": self.action_surprise,
            "action_diversity": self.action_diversity,
            "adequacy_proxy": self.adequacy_proxy,
            "leverage": self.leverage,
        }


@dataclass(frozen=True)
class RollingRegime:
    pressure: float
    chaos: float
    control: float
    label: str
    decisions: int
    confidence: float
    transitions: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "pressure": self.pressure,
            "chaos": self.chaos,
            "control": self.control,
            "label": self.label,
            "decisions": self.decisions,
            "confidence": self.confidence,
            "transitions": list(self.transitions),
        }


class LiveRegimeTracker:
    """Rolling, transparent PCC-like regime evidence for a human player.

    The detector intentionally avoids claiming latent psychological state.
    Pressure-like evidence emphasizes constraint faced at the decision point;
    Chaos-like evidence requires surprise/diversity *and* a value-adequacy
    proxy; Control-like evidence emphasizes initiative and causal leverage.
    """

    def __init__(self, window: int = 12, min_margin: float = 0.08) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if min_margin < 0:
            raise ValueError("min_margin must be non-negative")
        self.window = int(window)
        self.min_margin = float(min_margin)
        self._recent: Deque[DecisionEvidence] = deque(maxlen=self.window)
        self._action_counts: Counter[str] = Counter()
        self._label_history: list[str] = []
        self._transitions: list[str] = []

    def _label(self, pressure: float, chaos: float, control: float) -> tuple[str, float]:
        ranked = sorted(
            (("PRESSURE-LIKE", pressure), ("CHAOS-LIKE", chaos), ("CONTROL-LIKE", control)),
            key=lambda item: item[1],
            reverse=True,
        )
        margin = ranked[0][1] - ranked[1][1]
        if ranked[0][1] < 0.25 or margin < self.min_margin:
            return "MIXED", _clamp01(margin)
        return ranked[0][0], _clamp01(margin)

    def observe(self, state: State, action: str) -> DecisionEvidence:
        if action not in state.legal_actions():
            raise ValueError("action must be legal in supplied state")

        legal = state.legal_actions()
        wager_faced = state.to_call > 0
        faced_pressure = _clamp01(state.to_call / max(state.pot + state.to_call, 1)) if wager_faced else 0.0

        # Initiative is explicit action-taking, but calling while pressured is
        # not treated as initiative.  This is descriptive, not a PCC label.
        initiative = 1.0 if action in ("bet", "raise") else 0.0

        # Online action surprise is computed against *past human actions only*.
        # Laplace smoothing keeps unseen legal actions finite.
        denominator = sum(self._action_counts[a] + 1.0 for a in legal)
        p_action = (self._action_counts[action] + 1.0) / denominator
        max_surprisal = math.log(max(len(legal), 1))
        action_surprise = (
            min(1.0, -math.log(max(p_action, 1e-12)) / max_surprisal)
            if max_surprisal > 1e-12
            else 0.0
        )

        prospective = self._action_counts.copy()
        prospective[action] += 1
        action_diversity = _normalized_entropy(prospective)

        # Own-card equity is available to the human at decision time and is
        # therefore permitted as an adequacy proxy.  Folding receives a strong
        # adequacy penalty so random folds cannot masquerade as Chaos.
        own_equity = equity(state, state.actor)
        if action == "fold":
            adequacy_proxy = max(0.0, 1.0 - own_equity) * 0.35
        elif action in ("bet", "raise"):
            adequacy_proxy = _clamp01(0.35 + 0.65 * own_equity)
        else:
            adequacy_proxy = _clamp01(0.5 + 0.5 * (1.0 - abs(own_equity - 0.5) * 2.0))

        # Leverage approximates how strongly the chosen action changes the
        # opponent's immediate response set / pot commitment.
        if action in ("bet", "raise"):
            wager = state.bet_size if action == "bet" else state.to_call + state.bet_size
            leverage = _clamp01(wager / max(state.pot + wager, 1))
        elif action == "fold":
            leverage = 0.0
        else:
            leverage = 0.15 if wager_faced else 0.05

        # Transparent v0.9 exploratory scores.  Pressure emphasizes being
        # constrained, Chaos requires adequate surprising/diverse behavior,
        # and Control emphasizes initiative plus leverage, moderated by equity.
        pressure = _clamp01(
            0.65 * faced_pressure
            + 0.25 * (1.0 if action == "fold" and wager_faced else 0.0)
            + 0.10 * (1.0 if len(legal) <= 2 and wager_faced else 0.0)
        )
        chaos = _clamp01(
            adequacy_proxy * (0.55 * action_surprise + 0.45 * action_diversity)
        )
        control = _clamp01(
            (0.55 * initiative + 0.45 * leverage)
            * (0.45 + 0.55 * own_equity)
        )
        label, _ = self._label(pressure, chaos, control)

        evidence = DecisionEvidence(
            pressure=pressure,
            chaos=chaos,
            control=control,
            label=label,
            faced_pressure=faced_pressure,
            initiative=initiative,
            action_surprise=action_surprise,
            action_diversity=action_diversity,
            adequacy_proxy=adequacy_proxy,
            leverage=leverage,
        )
        self._recent.append(evidence)
        self._action_counts[action] += 1
        return evidence

    def rolling(self) -> RollingRegime:
        if not self._recent:
            return RollingRegime(0.0, 0.0, 0.0, "MIXED", 0, 0.0, tuple())
        n = len(self._recent)
        pressure = sum(item.pressure for item in self._recent) / n
        chaos = sum(item.chaos for item in self._recent) / n
        control = sum(item.control for item in self._recent) / n
        label, confidence = self._label(pressure, chaos, control)

        if not self._label_history or self._label_history[-1] != label:
            if self._label_history:
                self._transitions.append(f"{self._label_history[-1]} -> {label}")
            self._label_history.append(label)

        return RollingRegime(
            pressure=pressure,
            chaos=chaos,
            control=control,
            label=label,
            decisions=n,
            confidence=confidence,
            transitions=tuple(self._transitions[-8:]),
        )


def format_regime_panel(snapshot: RollingRegime) -> list[str]:
    """Compact text UI suitable for the existing terminal play loop."""
    width = 12

    def bar(value: float) -> str:
        filled = int(round(_clamp01(value) * width))
        return "#" * filled + "." * (width - filled)

    lines = [
        "--- Live PCC (exploratory rolling evidence) ---",
        f"Pressure [{bar(snapshot.pressure)}] {snapshot.pressure:0.2f}",
        f"Chaos    [{bar(snapshot.chaos)}] {snapshot.chaos:0.2f}",
        f"Control  [{bar(snapshot.control)}] {snapshot.control:0.2f}",
        f"Regime: {snapshot.label} | margin confidence {snapshot.confidence:0.2f} | window n={snapshot.decisions}",
    ]
    if snapshot.transitions:
        lines.append("Recent transition: " + snapshot.transitions[-1])
    lines.append("Note: -LIKE labels are exploratory observables, not latent-state ground truth.")
    return lines
