"""Fresh-seed test of whether public Pressure exposure suppresses effective surprise.

The pressure exposure is computed without PCC weights from the public response
model after the chosen action: response compression, predicted fold probability,
and commitment. Effective surprisal uses the unchanged independent value floor.
At aggregation time, effective surprisal is residualized on pressure exposure
within policy family using no hidden labels. Hidden PCC weights are consulted
only afterward for construct-validity correlations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .behavioral import PublicActionModel
from .engine import apply_action
from .policies import MODES
from .simulate import generate_family_dataset
from .value_floor import UniformContinuationValueModel, measure_synthetic_effective_chaos

DEFAULT_CALIBRATION_SEEDS = {"score": 2401, "independent": 2409}
DEFAULT_EVALUATION_SEEDS = {"score": 2601, "independent": 2609}
MIN_PRESSURE_CORRELATION_REDUCTION = 0.20
MIN_CHAOS_MARGIN_GAIN = 0.03
MIN_PRESSURE_EXPOSURE_CORRELATION = 0.20


def _correlation(left, right) -> float:
    x = np.asarray(left, dtype=float); y = np.asarray(right, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@dataclass(frozen=True)
class PressureSurpriseMeasurement:
    normalized_surprisal: float
    performance_adequacy: float
    effective_surprisal: float
    response_compression: float
    predicted_fold_probability: float
    commitment_ratio: float
    pressure_exposure: float

    def as_dict(self):
        return self.__dict__.copy()


class PressureSurpriseOracle:
    def __init__(self, action_model: PublicActionModel) -> None:
        self.action_model = action_model
        self.value_model = UniformContinuationValueModel()

    def measure(self, state, chosen_action: str) -> PressureSurpriseMeasurement:
        legal = state.legal_actions()
        probability = max(self.action_model.probabilities(state)[chosen_action], 1e-12)
        scale = math.log(len(legal)) if len(legal) > 1 else 1.0
        normalized = -math.log(probability) / scale if len(legal) > 1 else 0.0
        floor = measure_synthetic_effective_chaos(
            state, chosen_action, normalized, value_model=self.value_model
        )

        next_state = apply_action(state, chosen_action)
        compression = fold_probability = commitment = pressure = 0.0
        if not next_state.terminal and next_state.actor != state.actor and next_state.to_call > 0:
            probs = self.action_model.probabilities(next_state)
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 0.0
            compression = 1.0 - (entropy / max_entropy if max_entropy else 0.0)
            fold_probability = probs.get("fold", 0.0)
            commitment = next_state.to_call / max(next_state.pot + next_state.to_call, 1)
            pressure = (compression + fold_probability + commitment) / 3.0

        return PressureSurpriseMeasurement(
            normalized_surprisal=normalized,
            performance_adequacy=floor.adequacy,
            effective_surprisal=floor.effective_surprisal,
            response_compression=compression,
            predicted_fold_probability=fold_probability,
            commitment_ratio=commitment,
            pressure_exposure=pressure,
        )


def _aggregate(records):
    groups = defaultdict(list)
    for r in records:
        if r.get("is_focal_policy") and "behavioral_measurements" in r:
            groups[(r["policy_family"], r["mixture_id"], r["focal_seat"])].append(r)
    rows = []
    for (family, mixture_id, seat), decisions in sorted(groups.items()):
        ms = [d["behavioral_measurements"] for d in decisions]
        rows.append({
            "policy_family": family,
            "mixture_id": mixture_id,
            "focal_seat": seat,
            "weights": {m: float(decisions[0]["target_pcc_weights"][m]) for m in MODES},
            "effective_surprisal": float(np.mean([m["effective_surprisal"] for m in ms])),
            "pressure_exposure": float(np.mean([m["pressure_exposure"] for m in ms])),
            "decisions": len(decisions),
        })
    return rows


def _corrs(rows, field):
    return {m: _correlation([r[field] for r in rows], [r["weights"][m] for r in rows]) for m in MODES}


def _chaos_margin(c):
    return c["chaos"] - max(c["pressure"], c["control"])


def summarize_pressure_surprise(records) -> dict:
    rows = _aggregate(records)
    result = {}
    for family in sorted({r["policy_family"] for r in rows}):
        subset = [r.copy() for r in rows if r["policy_family"] == family]
        x = np.asarray([r["pressure_exposure"] for r in subset], dtype=float)
        y = np.asarray([r["effective_surprisal"] for r in subset], dtype=float)
        if x.std() < 1e-12:
            slope = 0.0
        else:
            slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        mean_x = float(x.mean()) if len(x) else 0.0
        for r in subset:
            r["pressure_adjusted_effective_surprisal"] = r["effective_surprisal"] - slope * (r["pressure_exposure"] - mean_x)
        raw = _corrs(subset, "effective_surprisal")
        adjusted = _corrs(subset, "pressure_adjusted_effective_surprisal")
        exposure = _corrs(subset, "pressure_exposure")
        result[family] = {
            "pressure_exposure_correlations": exposure,
            "effective_surprisal_correlations": raw,
            "pressure_adjusted_effective_surprisal_correlations": adjusted,
            "pressure_suppression_slope": slope,
            "raw_chaos_margin": _chaos_margin(raw),
            "adjusted_chaos_margin": _chaos_margin(adjusted),
            "chaos_margin_gain": _chaos_margin(adjusted) - _chaos_margin(raw),
            "pressure_correlation_reduction": abs(raw["pressure"]) - abs(adjusted["pressure"]),
            "seat_level_examples": len(subset),
        }

    checks = {}
    for family, data in result.items():
        checks[f"{family}_pressure_exposure_tracks_pressure"] = data["pressure_exposure_correlations"]["pressure"] >= MIN_PRESSURE_EXPOSURE_CORRELATION
        checks[f"{family}_pressure_correlation_reduced_by_0_20"] = data["pressure_correlation_reduction"] >= MIN_PRESSURE_CORRELATION_REDUCTION
        checks[f"{family}_chaos_margin_improves_by_0_03"] = data["chaos_margin_gain"] >= MIN_CHAOS_MARGIN_GAIN
    return {
        "pressure_suppression_mechanism_supported": all(checks.values()),
        "prespecified_checks": checks,
        "families": result,
        "thresholds": {
            "minimum_pressure_exposure_correlation": MIN_PRESSURE_EXPOSURE_CORRELATION,
            "minimum_pressure_correlation_reduction": MIN_PRESSURE_CORRELATION_REDUCTION,
            "minimum_chaos_margin_gain": MIN_CHAOS_MARGIN_GAIN,
        },
        "interpretation": "Support requires a label-free public Pressure exposure to track assigned Pressure, explain a substantial portion of the negative Pressure/effective-surprisal association, and improve Chaos discrimination in both policy families. Residualization never uses PCC weights.",
    }


def run_pressure_surprise_decomposition(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 60,
    evaluation_hands_per_seat: int = 100,
    score_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["score"],
    independent_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["independent"],
    score_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["score"],
    independent_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["independent"],
):
    calibration = []
    for family, seed in (("score", score_calibration_seed), ("independent", independent_calibration_seed)):
        records, _ = generate_family_dataset(family, calibration_mixtures, calibration_hands_per_seat, seed)
        calibration.extend(records)
    action_model = PublicActionModel.from_records(calibration)
    oracle = PressureSurpriseOracle(action_model)
    evaluation = []
    for family, seed in (("score", score_evaluation_seed), ("independent", independent_evaluation_seed)):
        records, _ = generate_family_dataset(family, evaluation_mixtures, evaluation_hands_per_seat, seed, measurement_oracle=oracle)
        evaluation.extend(records)
    report = summarize_pressure_surprise(evaluation)
    report["design"] = {
        "calibration_seeds": {"score": score_calibration_seed, "independent": independent_calibration_seed},
        "evaluation_seeds": {"score": score_evaluation_seed, "independent": independent_evaluation_seed},
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "weight_boundary": "PCC weights are used only after aggregation for correlations; pressure adjustment is fit from pressure exposure and effective surprisal only.",
    }
    return report


def write_pressure_surprise_decomposition(path: str | Path, **kwargs):
    report = run_pressure_surprise_decomposition(**kwargs)
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
