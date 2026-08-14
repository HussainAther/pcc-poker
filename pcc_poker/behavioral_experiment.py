"""Validation harness for label-free behavioral PCC measurements."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .behavioral import CounterfactualOracle, PublicActionModel
from .policies import MODES
from .simulate import generate_family_dataset

MEASURES = (
    "pressure_index",
    "control_efficiency",
    "predictive_control",
    "opponent_adaptation_control",
    "effective_surprisal",
)
DESCRIPTIVE_EFFECT_THRESHOLD = 0.20


def _correlation(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fisher_interval(correlation: float, sample_size: int) -> list[float]:
    """Approximate two-sided 95% Fisher-z interval for a Pearson correlation."""
    if sample_size <= 3:
        return [-1.0, 1.0]
    clipped = min(max(correlation, -0.999999), 0.999999)
    center = np.arctanh(clipped)
    margin = 1.96 / np.sqrt(sample_size - 3)
    return [float(np.tanh(center - margin)), float(np.tanh(center + margin))]


def aggregate_measured_mixtures(records: list[dict]) -> list[dict]:
    """Aggregate focal decisions; target weights remain validation outcomes only."""
    adaptation = {}
    complete_groups = defaultdict(list)
    for record in records:
        complete_groups[
            (record["policy_family"], record["mixture_id"], record["focal_seat"])
        ].append(record)
    for key, group_records in complete_groups.items():
        focal_seat = key[2]
        folds = [0, 0]
        faced_wagers = [0, 0]
        predicted_fold_rates = []
        aggressive_actions = []
        for record in group_records:
            round_index = int(record["round_index"])
            if record["actor"] == focal_seat:
                predicted_fold_rates.append(
                    (folds[round_index] + 1) / (faced_wagers[round_index] + 3)
                )
                aggressive_actions.append(
                    float(record["action"] in {"bet", "raise"})
                )
            elif record["to_call"] > 0:
                faced_wagers[round_index] += 1
                folds[round_index] += int(record["action"] == "fold")
        x = np.asarray(predicted_fold_rates, dtype=float)
        y = np.asarray(aggressive_actions, dtype=float)
        adaptation[key] = (
            float(np.mean((x - x.mean()) * (y - y.mean())))
            if len(x) > 1 else 0.0
        )

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
                if measure != "opponent_adaptation_control"
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
        rows[-1]["measurements"]["opponent_adaptation_control"] = adaptation[
            (family, mixture_id, focal_seat)
        ]
    return rows


def summarize_behavioral_validation(
    records: list[dict], control_measure: str = "control_efficiency"
) -> dict:
    rows = aggregate_measured_mixtures(records)
    families = sorted({row["policy_family"] for row in rows})
    measurement_to_mode = {
        "pressure_index": "pressure",
        control_measure: "control",
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
                for measure in measurement_to_mode
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
        "measurement_inputs": (
            "public betting context, prior public opponent actions, and the acting "
            "player's private rank for information-set value"
        ),
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


def run_predictive_control_confirmation(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 30,
    evaluation_hands_per_seat: int = 50,
    score_calibration_seed: int = 307,
    independent_calibration_seed: int = 311,
    score_evaluation_seed: int = 401,
    independent_evaluation_seed: int = 409,
) -> dict:
    """Prospective confirmation of the frozen private-information Control metric."""
    calibration_records = []
    for family, seed in (
        ("score", score_calibration_seed),
        ("independent", independent_calibration_seed),
    ):
        records, _ = generate_family_dataset(
            family, calibration_mixtures, calibration_hands_per_seat, seed
        )
        calibration_records.extend(records)

    oracle = CounterfactualOracle(PublicActionModel.from_records(calibration_records))
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

    report = summarize_behavioral_validation(
        evaluation_records, control_measure="predictive_control"
    )
    report["prospective_test"] = {
        "candidate_frozen_before_seed_results": True,
        "control_definition": (
            "positive pointwise information gain for the chosen action when the "
            "acting player's own private rank is added to full public context"
        ),
        "old_validation_seeds_reused": False,
        "calibration_seeds": {
            "score": score_calibration_seed,
            "independent": independent_calibration_seed,
        },
        "evaluation_seeds": {
            "score": score_evaluation_seed,
            "independent": independent_evaluation_seed,
        },
        "calibration_evaluation_overlap": False,
        "primary_control_measure": "predictive_control",
        "legacy_control_efficiency_reported_as_secondary": True,
    }
    return report


def write_predictive_control_confirmation(output_path: str | Path, **kwargs) -> dict:
    report = run_predictive_control_confirmation(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_opponent_adaptation_confirmation(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 60,
    evaluation_hands_per_seat: int = 100,
    score_calibration_seed: int = 503,
    independent_calibration_seed: int = 509,
    score_evaluation_seed: int = 601,
    independent_evaluation_seed: int = 607,
) -> dict:
    """Confirm whether response-contingent aggression uniquely tracks Control."""
    calibration_records = []
    for family, seed in (
        ("score", score_calibration_seed),
        ("independent", independent_calibration_seed),
    ):
        records, _ = generate_family_dataset(
            family, calibration_mixtures, calibration_hands_per_seat, seed
        )
        calibration_records.extend(records)
    oracle = CounterfactualOracle(PublicActionModel.from_records(calibration_records))

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

    report = summarize_behavioral_validation(
        evaluation_records, control_measure="opponent_adaptation_control"
    )
    discriminant = {}
    for family, result in report["families"].items():
        correlations = result["measurement_weight_correlations"][
            "opponent_adaptation_control"
        ]
        discriminant[family] = {
            "control_is_largest_correlation": correlations["control"] > max(
                correlations["pressure"], correlations["chaos"]
            ),
            "correlations": correlations,
            "control_correlation_approximate_95pct_ci": _fisher_interval(
                correlations["control"], result["seat_level_examples"]
            ),
        }
    report["prospective_test"] = {
        "candidate_frozen_before_confirmation_seeds": True,
        "development_evaluation_seeds": {"score": 401, "independent": 409},
        "confirmation_evaluation_seeds": {
            "score": score_evaluation_seed,
            "independent": independent_evaluation_seed,
        },
        "old_validation_or_development_seeds_reused": False,
        "control_definition": (
            "within-seat covariance between aggression and the round-specific "
            "opponent fold rate estimated only from earlier observed actions"
        ),
        "primary_control_measure": "opponent_adaptation_control",
        "evaluation_mixtures_per_family": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "discriminant_check": discriminant,
    }
    return report


def write_opponent_adaptation_confirmation(output_path: str | Path, **kwargs) -> dict:
    report = run_opponent_adaptation_confirmation(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_adaptive_family_validation(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 60,
    evaluation_hands_per_seat: int = 100,
    calibration_seed: int = 809,
    evaluation_seed: int = 811,
) -> dict:
    """Test whether the deliberately adaptive family expresses all PCC axes."""
    calibration_records, _ = generate_family_dataset(
        "adaptive",
        calibration_mixtures,
        calibration_hands_per_seat,
        calibration_seed,
    )
    oracle = CounterfactualOracle(PublicActionModel.from_records(calibration_records))
    evaluation_records, _ = generate_family_dataset(
        "adaptive",
        evaluation_mixtures,
        evaluation_hands_per_seat,
        evaluation_seed,
        measurement_oracle=oracle,
    )
    report = summarize_behavioral_validation(
        evaluation_records, control_measure="opponent_adaptation_control"
    )
    # This is a single-family construct check; the generic cross-family field
    # produced by the shared summarizer does not apply.
    report.pop("cross_family_construct_result", None)
    report["adaptive_family_construct_result"] = {
        mode: {
            "result": (
                "positive_in_adaptive_family"
                if report["families"]["adaptive"]["matching_axis_correlations"][mode]
                >= DESCRIPTIVE_EFFECT_THRESHOLD
                else "inconclusive_in_adaptive_family"
            ),
            "correlation": report["families"]["adaptive"][
                "matching_axis_correlations"
            ][mode],
        }
        for mode in MODES
    }
    report["design"] = {
        "purpose": "game-mechanic construct validation, not independent PCC evidence",
        "policy_family": "adaptive",
        "control_mechanism": (
            "card-value-constrained aggression timed to an online estimate of "
            "the opponent's round-specific fold probability"
        ),
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "calibration_evaluation_overlap": False,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "circularity_warning": (
            "The generator and measurement both operationalize opponent-response "
            "adaptation; success validates implementation, not natural occurrence."
        ),
    }
    return report


def write_adaptive_family_validation(output_path: str | Path, **kwargs) -> dict:
    report = run_adaptive_family_validation(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
