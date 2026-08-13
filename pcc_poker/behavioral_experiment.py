"""Validation harness for label-free behavioral PCC measurements."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .behavioral import CounterfactualOracle, PublicActionModel
from .policies import MODES
from .simulate import generate_family_dataset

MEASURES = ("pressure_index", "control_efficiency", "effective_surprisal")
DESCRIPTIVE_EFFECT_THRESHOLD = 0.20


def _correlation(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_measured_mixtures(records: list[dict]) -> list[dict]:
    """Aggregate focal decisions; target weights remain validation outcomes only."""
    grouped = defaultdict(list)
    for record in records:
        if record.get("is_focal_policy") and "behavioral_measurements" in record:
            grouped[(record["policy_family"], record["mixture_id"], record["focal_seat"])].append(record)

    rows = []
    for (family, mixture_id, focal_seat), decisions in sorted(grouped.items()):
        weights = decisions[0]["target_pcc_weights"]
        action_counts = Counter(record["action"] for record in decisions)
        rows.append({
            "policy_family": family,
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "weights": {mode: float(weights[mode]) for mode in MODES},
            "measurements": {
                measure: float(np.mean([
                    record["behavioral_measurements"][measure]
                    for record in decisions
                ]))
                for measure in MEASURES
            },
            "mean_payoff": float(np.mean([
                record["terminal_payoff"] for record in decisions
            ])),
            "action_rates": {
                action: action_counts[action] / len(decisions)
                for action in ("check", "bet", "fold", "call", "raise")
            },
            "decisions": len(decisions),
        })
    return rows


def summarize_behavioral_validation(records: list[dict]) -> dict:
    rows = aggregate_measured_mixtures(records)
    families = sorted({row["policy_family"] for row in rows})
    measurement_to_mode = {
        "pressure_index": "pressure",
        "control_efficiency": "control",
        "effective_surprisal": "chaos",
    }
    by_family = {}
    for family in families:
        subset = [row for row in rows if row["policy_family"] == family]
        matrix = {
            measure: {
                mode: _correlation(
                    [row["measurements"][measure] for row in subset],
                    [row["weights"][mode] for row in subset],
                )
                for mode in MODES
            }
            for measure in MEASURES
        }
        by_family[family] = {
            "mixtures": len({row["mixture_id"] for row in subset}),
            "seat_level_examples": len(subset),
            "measurement_weight_correlations": matrix,
            "matching_axis_correlations": {
                measurement_to_mode[measure]: matrix[measure][measurement_to_mode[measure]]
                for measure in MEASURES
            },
            "measurement_payoff_correlations": {
                measure: _correlation(
                    [row["measurements"][measure] for row in subset],
                    [row["mean_payoff"] for row in subset],
                )
                for measure in MEASURES
            },
        }

    matching_sign_consistency = {
        mode: all(
            by_family[family]["matching_axis_correlations"][mode] > 0
            for family in families
        )
        for mode in MODES
    }
    cross_family_result = {}
    for mode in MODES:
        correlations = [
            by_family[family]["matching_axis_correlations"][mode]
            for family in families
        ]
        if all(value >= DESCRIPTIVE_EFFECT_THRESHOLD for value in correlations):
            result = "positive_in_both_families"
        elif any(value <= -DESCRIPTIVE_EFFECT_THRESHOLD for value in correlations):
            result = "contradicted_in_at_least_one_family"
        else:
            result = "inconclusive"
        cross_family_result[mode] = {
            "result": result,
            "correlations": dict(zip(families, correlations)),
        }
    return {
        "status": "completed",
        "measurement_inputs": "public betting context plus acting player's private rank for information-set value",
        "generator_weights_used_as_predictors": False,
        "families": by_family,
        "matching_axis_positive_in_every_family": matching_sign_consistency,
        "cross_family_construct_result": cross_family_result,
        "descriptive_effect_threshold": DESCRIPTIVE_EFFECT_THRESHOLD,
        "threshold_note": (
            "The 0.20 threshold is a transparent descriptive convention added "
            "for this engineering validation, not a preregistered significance test."
        ),
        "warning": (
            "Assigned weights are used only for construct-validation correlations. "
            "The measurements themselves are computed without generator labels, "
            "hidden opponent cards, or component scores."
        ),
    }


def run_behavioral_validation(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 30,
    evaluation_hands_per_seat: int = 50,
    score_calibration_seed: int = 101,
    independent_calibration_seed: int = 103,
    score_evaluation_seed: int = 211,
    independent_evaluation_seed: int = 223,
) -> dict:
    """Calibrate publicly, then evaluate both families on wholly separate hands."""
    calibration_records = []
    for family, seed in (
        ("score", score_calibration_seed),
        ("independent", independent_calibration_seed),
    ):
        records, _ = generate_family_dataset(
            family,
            calibration_mixtures,
            calibration_hands_per_seat,
            seed,
        )
        calibration_records.extend(records)

    public_model = PublicActionModel.from_records(calibration_records)
    oracle = CounterfactualOracle(public_model)
    evaluation_records = []
    for family, seed in (
        ("score", score_evaluation_seed),
        ("independent", independent_evaluation_seed),
    ):
        records, _ = generate_family_dataset(
            family,
            evaluation_mixtures,
            evaluation_hands_per_seat,
            seed,
            measurement_oracle=oracle,
        )
        evaluation_records.extend(records)

    report = summarize_behavioral_validation(evaluation_records)
    report["design"] = {
        "calibration_mixtures_per_family": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures_per_family": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "calibration_seeds": {
            "score": score_calibration_seed,
            "independent": independent_calibration_seed,
        },
        "evaluation_seeds": {
            "score": score_evaluation_seed,
            "independent": independent_evaluation_seed,
        },
        "calibration_evaluation_overlap": False,
        "unit_of_analysis": "mixture by focal seat",
        "confirmatory_status": "synthetic engineering validation only",
    }
    return report


def write_behavioral_validation(output_path: str | Path, **kwargs) -> dict:
    report = run_behavioral_validation(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
