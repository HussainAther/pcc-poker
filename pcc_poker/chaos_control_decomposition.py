"""Fresh-seed decomposition of Chaos/Control entanglement in effective surprisal.

This experiment does not alter either PCC generator or the independent value
floor.  A public action model is calibrated twice: a static model that sees the
current public betting context and a temporal model that additionally sees the
public action history.  Evaluation surprise is then split into:

* history-explained surprise: surprise removed by conditioning on history;
* history-residual surprise: surprise that remains after conditioning on history.

Both pieces are multiplied by the same non-PCC performance adequacy from
``value_floor.py``.  Hidden PCC weights are consulted only after aggregation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .behavioral import PublicActionModel
from .policies import MODES
from .simulate import generate_family_dataset
from .value_floor import UniformContinuationValueModel, measure_synthetic_effective_chaos

DEFAULT_CALIBRATION_SEEDS = {"score": 2001, "independent": 2009}
DEFAULT_EVALUATION_SEEDS = {"score": 2201, "independent": 2209}
MIN_RESIDUAL_CHAOS_CORRELATION = 0.20
MIN_RESIDUAL_CHAOS_MARGIN = 0.03
MIN_HISTORY_CONTROL_MARGIN = 0.03
MIN_MARGIN_IMPROVEMENT = 0.03


def _correlation(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _static_record_context(record: dict) -> tuple:
    return (
        int(record["actor"]),
        int(record["round_index"]),
        bool(record["to_call"] > 0),
        int(record["pot"]),
        tuple(record["legal_actions"]),
    )


def _static_state_context(state) -> tuple:
    return (
        state.actor,
        state.round_index,
        bool(state.to_call > 0),
        state.pot,
        state.legal_actions(),
    )


def _history_signature(history) -> tuple:
    counts = Counter(history)
    last = history[-1] if history else "none"
    penultimate = history[-2] if len(history) > 1 else "none"
    aggressive = counts["bet"] + counts["raise"]
    passive = counts["check"] + counts["call"]
    return (
        min(aggressive, 3),
        min(passive, 3),
        min(counts["fold"], 2),
        last,
        penultimate,
    )


def _temporal_record_context(record: dict) -> tuple:
    return (_static_record_context(record), _history_signature(record["history"]))


def _temporal_state_context(state) -> tuple:
    return (_static_state_context(state), _history_signature(state.history))


class FrozenStaticTemporalActionModel:
    """Smoothed current-state and history-conditioned public action models."""

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.smoothing = float(smoothing)
        self.static_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.temporal_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.global_counts: Counter = Counter()

    @classmethod
    def from_records(cls, records: list[dict], smoothing: float = 1.0):
        model = cls(smoothing)
        for record in records:
            action = record["action"]
            model.static_counts[_static_record_context(record)][action] += 1
            model.temporal_counts[_temporal_record_context(record)][action] += 1
            model.global_counts[action] += 1
        return model

    def _probabilities(self, counts: Counter, legal: tuple[str, ...]) -> dict[str, float]:
        denominator = sum(counts[action] + self.smoothing for action in legal)
        return {
            action: (counts[action] + self.smoothing) / denominator
            for action in legal
        }

    def probabilities(self, state, *, temporal: bool) -> dict[str, float]:
        legal = state.legal_actions()
        if temporal:
            counts = self.temporal_counts.get(_temporal_state_context(state))
        else:
            counts = self.static_counts.get(_static_state_context(state))
        if not counts or not sum(counts.values()):
            counts = self.global_counts
        return self._probabilities(counts, legal)


@dataclass(frozen=True)
class ChaosControlDecompositionMeasurement:
    static_normalized_surprisal: float
    temporal_normalized_surprisal: float
    history_explained_surprisal: float
    history_residual_surprisal: float
    performance_adequacy: float
    static_effective_surprisal: float
    history_explained_effective_surprisal: float
    history_residual_effective_surprisal: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class ChaosControlDecompositionOracle:
    def __init__(self, model: FrozenStaticTemporalActionModel) -> None:
        self.model = model
        self.value_model = UniformContinuationValueModel()

    def measure(self, state, chosen_action: str) -> ChaosControlDecompositionMeasurement:
        legal_count = len(state.legal_actions())
        scale = math.log(legal_count) if legal_count > 1 else 1.0
        p_static = max(self.model.probabilities(state, temporal=False)[chosen_action], 1e-12)
        p_temporal = max(self.model.probabilities(state, temporal=True)[chosen_action], 1e-12)
        static = -math.log(p_static) / scale if legal_count > 1 else 0.0
        temporal = -math.log(p_temporal) / scale if legal_count > 1 else 0.0
        explained = max(0.0, static - temporal)
        residual = temporal
        floor = measure_synthetic_effective_chaos(
            state,
            chosen_action,
            static,
            value_model=self.value_model,
        )
        adequacy = floor.adequacy
        return ChaosControlDecompositionMeasurement(
            static_normalized_surprisal=static,
            temporal_normalized_surprisal=temporal,
            history_explained_surprisal=explained,
            history_residual_surprisal=residual,
            performance_adequacy=adequacy,
            static_effective_surprisal=static * adequacy,
            history_explained_effective_surprisal=explained * adequacy,
            history_residual_effective_surprisal=residual * adequacy,
        )


def aggregate_decomposition(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("is_focal_policy") and "behavioral_measurements" in record:
            grouped[(record["policy_family"], record["mixture_id"], record["focal_seat"])].append(record)
    rows = []
    fields = (
        "static_effective_surprisal",
        "history_explained_effective_surprisal",
        "history_residual_effective_surprisal",
        "performance_adequacy",
    )
    for (family, mixture_id, focal_seat), decisions in sorted(grouped.items()):
        measurements = [d["behavioral_measurements"] for d in decisions]
        row = {
            "policy_family": family,
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "weights": {mode: float(decisions[0]["target_pcc_weights"][mode]) for mode in MODES},
            "decisions": len(decisions),
        }
        for field in fields:
            row[field] = float(np.mean([measurement[field] for measurement in measurements]))
        rows.append(row)
    return rows


def _metric_correlations(rows: list[dict], metric: str) -> dict[str, float]:
    return {
        mode: _correlation(
            [row[metric] for row in rows],
            [row["weights"][mode] for row in rows],
        )
        for mode in MODES
    }


def _chaos_margin(correlations: dict[str, float]) -> float:
    return correlations["chaos"] - max(correlations["pressure"], correlations["control"])


def _control_margin(correlations: dict[str, float]) -> float:
    return correlations["control"] - max(correlations["pressure"], correlations["chaos"])


def summarize_chaos_control_decomposition(records: list[dict]) -> dict:
    rows = aggregate_decomposition(records)
    families = sorted({row["policy_family"] for row in rows})
    by_family = {}
    for family in families:
        subset = [row for row in rows if row["policy_family"] == family]
        static = _metric_correlations(subset, "static_effective_surprisal")
        explained = _metric_correlations(subset, "history_explained_effective_surprisal")
        residual = _metric_correlations(subset, "history_residual_effective_surprisal")
        by_family[family] = {
            "mixtures": len({row["mixture_id"] for row in subset}),
            "seat_level_examples": len(subset),
            "mean_performance_adequacy": float(np.mean([row["performance_adequacy"] for row in subset])),
            "static_effective_correlations": static,
            "history_explained_effective_correlations": explained,
            "history_residual_effective_correlations": residual,
            "static_chaos_margin": _chaos_margin(static),
            "history_explained_control_margin": _control_margin(explained),
            "history_residual_chaos_margin": _chaos_margin(residual),
            "residual_chaos_margin_gain_over_static": _chaos_margin(residual) - _chaos_margin(static),
        }

    independent = by_family.get("independent", {})
    residual_corr = independent.get("history_residual_effective_correlations", {})
    checks = {
        "independent_residual_chaos_correlation_at_least_0_20": residual_corr.get("chaos", 0.0) >= MIN_RESIDUAL_CHAOS_CORRELATION,
        "independent_residual_is_chaos_discriminant_by_0_03": independent.get("history_residual_chaos_margin", -1.0) >= MIN_RESIDUAL_CHAOS_MARGIN,
        "independent_history_explained_is_control_discriminant_by_0_03": independent.get("history_explained_control_margin", -1.0) >= MIN_HISTORY_CONTROL_MARGIN,
        "independent_residual_improves_chaos_margin_by_0_03": independent.get("residual_chaos_margin_gain_over_static", -1.0) >= MIN_MARGIN_IMPROVEMENT,
    }
    return {
        "chaos_control_entanglement_decomposition_supported": all(checks.values()),
        "prespecified_checks": checks,
        "families": by_family,
        "thresholds": {
            "minimum_residual_chaos_correlation": MIN_RESIDUAL_CHAOS_CORRELATION,
            "minimum_residual_chaos_margin": MIN_RESIDUAL_CHAOS_MARGIN,
            "minimum_history_explained_control_margin": MIN_HISTORY_CONTROL_MARGIN,
            "minimum_residual_margin_improvement": MIN_MARGIN_IMPROVEMENT,
        },
        "interpretation": (
            "Support would mean that conditioning on public action history removes a Control-linked part of "
            "effective surprisal while leaving a more Chaos-specific value-preserving residual in the independent family. "
            "Failure is retained without changing contexts, thresholds, seeds, generators, or the value floor."
        ),
    }


def run_chaos_control_decomposition(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 60,
    evaluation_hands_per_seat: int = 100,
    score_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["score"],
    independent_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["independent"],
    score_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["score"],
    independent_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["independent"],
) -> dict:
    calibration_records = []
    for family, seed in (("score", score_calibration_seed), ("independent", independent_calibration_seed)):
        records, _ = generate_family_dataset(
            family,
            calibration_mixtures,
            calibration_hands_per_seat,
            seed,
        )
        calibration_records.extend(records)

    model = FrozenStaticTemporalActionModel.from_records(calibration_records)
    oracle = ChaosControlDecompositionOracle(model)
    evaluation_records = []
    for family, seed in (("score", score_evaluation_seed), ("independent", independent_evaluation_seed)):
        records, _ = generate_family_dataset(
            family,
            evaluation_mixtures,
            evaluation_hands_per_seat,
            seed,
            measurement_oracle=oracle,
        )
        evaluation_records.extend(records)

    report = summarize_chaos_control_decomposition(evaluation_records)
    report["design"] = {
        "calibration_seeds": {
            "score": score_calibration_seed,
            "independent": independent_calibration_seed,
        },
        "evaluation_seeds": {
            "score": score_evaluation_seed,
            "independent": independent_evaluation_seed,
        },
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "history_boundary": "public prior actions only; no weights, policy labels, component scores, outcomes, or opponent private cards",
        "value_floor": "unchanged UniformContinuationValueModel",
    }
    return report


def write_chaos_control_decomposition(path: str | Path, **kwargs) -> dict:
    report = run_chaos_control_decomposition(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
