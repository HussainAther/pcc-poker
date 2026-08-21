"""Post-intervention decomposition of Score-family value-sensitive Control.

This diagnostic changes no policy and no frozen v0.8 artifact. It asks why the
prospective Score-Control contextual-gain intervention recovers information
uptake and context alignment but still fails the final value-sensitive
intervention stage.

The decomposition separates:
  * positive aligned-vs-yoked context gain,
  * counterfactual action efficiency,
  * regret,
  * their value-weighted product, and
  * the frequency of positive-context decisions that are low-efficiency.

Synthetic PCC weights remain validation labels used only after trajectory
aggregation.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .behavioral import CounterfactualOracle, PublicActionModel
from .contextual_control_observable import FrozenAlignedYokedHistoryModel
from .control_structural_recovery import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    DEFAULT_YOKE_SEED,
)
from .policies import MODES
from .score_control_intervention import _generate_dataset

MAX_EFFICIENCY_CONTROL_CORRELATION = -0.20
MIN_REGRET_CONTROL_CORRELATION = 0.20
MIN_PRODUCT_ATTENUATION = 0.05
MIN_LOW_EFFICIENCY_CONTEXT_CONTROL_CORRELATION = 0.20
LOW_EFFICIENCY_THRESHOLD = 0.80


def _corr(a, b) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _aggregate(records: list[dict], model: FrozenAlignedYokedHistoryModel) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        if not record.get("is_focal_policy"):
            continue
        pa = max(model.probability(record, condition="aligned"), 1e-12)
        py = max(model.probability(record, condition="yoked"), 1e-12)
        context_gain = math.log(pa) - math.log(py)
        positive_gain = max(context_gain, 0.0)
        measurements = record["behavioral_measurements"]
        efficiency = float(measurements["control_efficiency"])
        regret = float(measurements["regret"])
        grouped[(record["policy_family"], record["mixture_id"], record["focal_seat"])].append(
            (context_gain, positive_gain, efficiency, regret, record)
        )

    rows = []
    for (family, mixture_id, focal_seat), values in sorted(grouped.items()):
        first = values[0][4]
        rows.append({
            "policy_family": family,
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "decisions": len(values),
            "context_alignment": float(np.mean([v[0] for v in values])),
            "positive_context_gain": float(np.mean([v[1] for v in values])),
            "control_efficiency": float(np.mean([v[2] for v in values])),
            "regret": float(np.mean([v[3] for v in values])),
            "value_sensitive_intervention": float(np.mean([v[1] * v[2] for v in values])),
            "positive_context_rate": float(np.mean([v[0] > 0 for v in values])),
            "low_efficiency_positive_context_rate": float(np.mean([
                v[0] > 0 and v[2] < LOW_EFFICIENCY_THRESHOLD for v in values
            ])),
            "weights": {mode: float(first["target_pcc_weights"][mode]) for mode in MODES},
        })
    return rows


def _family_summary(rows: list[dict], family: str) -> dict:
    subset = [row for row in rows if row["policy_family"] == family]
    if not subset:
        raise ValueError(f"missing family {family!r}")

    correlations = {}
    for measure in (
        "context_alignment",
        "positive_context_gain",
        "control_efficiency",
        "regret",
        "value_sensitive_intervention",
        "positive_context_rate",
        "low_efficiency_positive_context_rate",
    ):
        correlations[measure] = {
            mode: _corr(
                [row[measure] for row in subset],
                [row["weights"][mode] for row in subset],
            )
            for mode in MODES
        }

    positive_control = correlations["positive_context_gain"]["control"]
    product_control = correlations["value_sensitive_intervention"]["control"]
    attenuation = positive_control - product_control
    return {
        "groups": len(subset),
        "means": {
            measure: float(np.mean([row[measure] for row in subset]))
            for measure in (
                "context_alignment",
                "positive_context_gain",
                "control_efficiency",
                "regret",
                "value_sensitive_intervention",
                "positive_context_rate",
                "low_efficiency_positive_context_rate",
            )
        },
        "weight_correlations": correlations,
        "positive_gain_to_value_product_control_attenuation": attenuation,
    }


def summarize_score_control_value_decomposition(
    records: list[dict], model: FrozenAlignedYokedHistoryModel
) -> dict:
    rows = _aggregate(records, model)
    score = _family_summary(rows, "score")
    adaptive = _family_summary(rows, "adaptive")

    checks = {
        "score_efficiency_is_control_anticorrelated": (
            score["weight_correlations"]["control_efficiency"]["control"]
            <= MAX_EFFICIENCY_CONTROL_CORRELATION
        ),
        "score_regret_is_control_correlated": (
            score["weight_correlations"]["regret"]["control"]
            >= MIN_REGRET_CONTROL_CORRELATION
        ),
        "value_guardrail_attenuates_score_control_signal_by_0_05": (
            score["positive_gain_to_value_product_control_attenuation"]
            >= MIN_PRODUCT_ATTENUATION
        ),
        "low_efficiency_positive_contexts_concentrate_with_control": (
            score["weight_correlations"]["low_efficiency_positive_context_rate"]["control"]
            >= MIN_LOW_EFFICIENCY_CONTEXT_CONTROL_CORRELATION
        ),
    }
    return {
        "status": "confirmed" if all(checks.values()) else "partial",
        "value_guardrail_bottleneck_supported": all(checks.values()),
        "hypothesis": (
            "After contextual response gain is strengthened, Score-Control reads context but Control-heavy "
            "trajectories increasingly select actions with lower counterfactual efficiency. The value guardrail "
            "therefore attenuates the final context-by-value signal rather than revealing a missing context signal."
        ),
        "families": {"score": score, "adaptive": adaptive},
        "prespecified_checks": checks,
        "thresholds": {
            "maximum_efficiency_control_correlation": MAX_EFFICIENCY_CONTROL_CORRELATION,
            "minimum_regret_control_correlation": MIN_REGRET_CONTROL_CORRELATION,
            "minimum_product_attenuation": MIN_PRODUCT_ATTENUATION,
            "minimum_low_efficiency_context_control_correlation": MIN_LOW_EFFICIENCY_CONTEXT_CONTROL_CORRELATION,
            "low_efficiency_threshold": LOW_EFFICIENCY_THRESHOLD,
        },
        "trajectory_groups": len(rows),
        "policy_modified": False,
        "human_data_accessed": False,
        "frozen_v0.8_human_panel_modified": False,
        "interpretation": (
            "This is a post-intervention mechanism decomposition. It does not resolve Poker Control and does not "
            "authorize retuning. A future intervention, if any, should target value-sensitive action selection rather "
            "than increasing contextual gain again."
        ),
    }


def run_score_control_value_decomposition(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 30,
    evaluation_mixtures: int = 40,
    evaluation_hands_per_seat: int = 60,
    score_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["score"],
    adaptive_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["adaptive"],
    score_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["score"],
    adaptive_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["adaptive"],
    yoke_seed: int = DEFAULT_YOKE_SEED,
) -> dict:
    calibration = []
    for family, seed in (("score", score_calibration_seed), ("adaptive", adaptive_calibration_seed)):
        batch, _ = _generate_dataset(family, calibration_mixtures, calibration_hands_per_seat, seed)
        calibration.extend(batch)

    history_model = FrozenAlignedYokedHistoryModel.from_records(calibration, seed=yoke_seed)
    value_oracle = CounterfactualOracle(PublicActionModel.from_records(calibration))

    evaluation = []
    for family, seed in (("score", score_evaluation_seed), ("adaptive", adaptive_evaluation_seed)):
        batch, _ = _generate_dataset(
            family,
            evaluation_mixtures,
            evaluation_hands_per_seat,
            seed,
            measurement_oracle=value_oracle,
        )
        evaluation.extend(batch)

    report = summarize_score_control_value_decomposition(evaluation, history_model)
    report["design"] = {
        "status": "post_v0.8_score_control_value_decomposition",
        "calibration_seeds": {"score": score_calibration_seed, "adaptive": adaptive_calibration_seed},
        "evaluation_seeds": {"score": score_evaluation_seed, "adaptive": adaptive_evaluation_seed},
        "yoke_seed": yoke_seed,
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "weight_boundary": "Synthetic PCC weights are used only after trajectory aggregation for diagnostic correlations.",
    }
    return report


def write_score_control_value_decomposition(path: str | Path, **kwargs) -> dict:
    report = run_score_control_value_decomposition(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
