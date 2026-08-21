"""Post-freeze structural recovery test for synthetic Poker Control.

This experiment does not modify the v0.8.0 frozen human-facing panel. It asks
whether Control is recoverable as a three-stage structure across two synthetic
implementation families:

    information uptake -> context alignment -> value-sensitive intervention

Every candidate measurement is label-free at decision time. Synthetic PCC
weights are consulted only after trajectory aggregation for construct-validity
correlations.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .behavioral import CounterfactualOracle, PublicActionModel
from .contextual_control_observable import FrozenAlignedYokedHistoryModel
from .policies import MODES
from .simulate import generate_family_dataset

FAMILIES = ("score", "adaptive")
DEFAULT_CALIBRATION_SEEDS = {"score": 5101, "adaptive": 5109}
DEFAULT_EVALUATION_SEEDS = {"score": 5301, "adaptive": 5309}
DEFAULT_YOKE_SEED = 51999
MIN_CONTROL_CORRELATION = 0.20
MIN_DISCRIMINANT_MARGIN = 0.05

STAGES = {
    "information_uptake": "aligned public-history log likelihood minus static-context log likelihood",
    "context_alignment": "aligned public-history log likelihood minus matched context-yoked log likelihood",
    "value_sensitive_intervention": "positive context-alignment gain weighted by label-free counterfactual action efficiency",
}


def _corr(a, b) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_control_structural_recovery(
    records: list[dict], model: FrozenAlignedYokedHistoryModel
) -> dict:
    grouped = defaultdict(list)
    for record in records:
        if not record.get("is_focal_policy"):
            continue
        pa = max(model.probability(record, condition="aligned"), 1e-12)
        py = max(model.probability(record, condition="yoked"), 1e-12)
        ps = max(model.probability(record, condition="static"), 1e-12)
        aligned_over_static = math.log(pa) - math.log(ps)
        aligned_over_yoked = math.log(pa) - math.log(py)
        efficiency = float(record["behavioral_measurements"]["control_efficiency"])
        value_sensitive = max(aligned_over_yoked, 0.0) * efficiency
        grouped[(record["policy_family"], record["mixture_id"], record["focal_seat"])].append(
            (aligned_over_static, aligned_over_yoked, value_sensitive, record)
        )

    rows = []
    for (family, mixture_id, focal_seat), values in sorted(grouped.items()):
        first = values[0][3]
        rows.append({
            "policy_family": family,
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "decisions": len(values),
            "information_uptake": float(np.mean([value[0] for value in values])),
            "context_alignment": float(np.mean([value[1] for value in values])),
            "value_sensitive_intervention": float(np.mean([value[2] for value in values])),
            "weights": {mode: float(first["target_pcc_weights"][mode]) for mode in MODES},
        })

    observed_families = tuple(sorted({row["policy_family"] for row in rows}))
    if observed_families != tuple(sorted(FAMILIES)):
        raise ValueError(f"structural recovery requires families {FAMILIES!r}")

    family_results = {}
    for family in FAMILIES:
        subset = [row for row in rows if row["policy_family"] == family]
        stage_results = {}
        for stage in STAGES:
            correlations = {
                mode: _corr(
                    [row[stage] for row in subset],
                    [row["weights"][mode] for row in subset],
                )
                for mode in MODES
            }
            margin = correlations["control"] - max(
                correlations["pressure"], correlations["chaos"]
            )
            checks = {
                "control_correlation_at_least_0_20": correlations["control"] >= MIN_CONTROL_CORRELATION,
                "control_discriminant_margin_at_least_0_05": margin >= MIN_DISCRIMINANT_MARGIN,
            }
            stage_results[stage] = {
                "weight_correlations": correlations,
                "control_correlation": correlations["control"],
                "discriminant_margin": margin,
                "checks": checks,
                "stage_recovered": all(checks.values()),
            }
        family_results[family] = {
            "groups": len(subset),
            "stages": stage_results,
            "all_three_stages_recovered": all(
                stage_results[stage]["stage_recovered"] for stage in STAGES
            ),
        }

    stage_replication = {
        stage: all(
            family_results[family]["stages"][stage]["stage_recovered"]
            for family in FAMILIES
        )
        for stage in STAGES
    }
    checks = {
        "information_uptake_replicates_across_families": stage_replication["information_uptake"],
        "context_alignment_replicates_across_families": stage_replication["context_alignment"],
        "value_sensitive_intervention_replicates_across_families": stage_replication["value_sensitive_intervention"],
        "matched_yoke_margins_preserved": all(model.margin_checks().values()),
    }
    return {
        "control_structural_recovery_confirmed": all(checks.values()),
        "status": (
            "confirmed"
            if all(checks.values())
            else "partial" if any(stage_replication.values()) or any(
                family_results[family]["all_three_stages_recovered"] for family in FAMILIES
            ) else "failed"
        ),
        "structural_hypothesis": "information uptake -> context alignment -> value-sensitive intervention",
        "stages": STAGES,
        "families": family_results,
        "stage_replication": stage_replication,
        "prespecified_checks": checks,
        "thresholds": {
            "minimum_control_correlation": MIN_CONTROL_CORRELATION,
            "minimum_discriminant_margin": MIN_DISCRIMINANT_MARGIN,
        },
        "margin_checks": model.margin_checks(),
        "trajectory_groups": len(rows),
        "interpretation_rule": (
            "Control is structurally resolved only if all three stages are positive and discriminant in both "
            "synthetic implementation families. A one-family success is retained as partial mechanism evidence, "
            "not promoted to the frozen human-facing panel."
        ),
    }


def run_control_structural_recovery(
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
    if calibration_mixtures < 2 or evaluation_mixtures < 2:
        raise ValueError("at least two mixtures are required")
    if calibration_hands_per_seat < 1 or evaluation_hands_per_seat < 1:
        raise ValueError("hand counts must be positive")

    calibration = []
    for family, seed in (
        ("score", score_calibration_seed),
        ("adaptive", adaptive_calibration_seed),
    ):
        records, _ = generate_family_dataset(
            family, calibration_mixtures, calibration_hands_per_seat, seed
        )
        calibration.extend(records)

    history_model = FrozenAlignedYokedHistoryModel.from_records(
        calibration, seed=yoke_seed
    )
    value_oracle = CounterfactualOracle(PublicActionModel.from_records(calibration))

    evaluation = []
    for family, seed in (
        ("score", score_evaluation_seed),
        ("adaptive", adaptive_evaluation_seed),
    ):
        records, _ = generate_family_dataset(
            family,
            evaluation_mixtures,
            evaluation_hands_per_seat,
            seed,
            measurement_oracle=value_oracle,
        )
        evaluation.extend(records)

    report = summarize_control_structural_recovery(evaluation, history_model)
    report["design"] = {
        "status": "post_v0.8_synthetic_control_structural_recovery",
        "families": list(FAMILIES),
        "calibration_seeds": {
            "score": score_calibration_seed,
            "adaptive": adaptive_calibration_seed,
        },
        "evaluation_seeds": {
            "score": score_evaluation_seed,
            "adaptive": adaptive_evaluation_seed,
        },
        "yoke_seed": yoke_seed,
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "human_data_accessed": False,
        "frozen_v0.8_human_panel_modified": False,
        "weight_boundary": "Synthetic PCC weights are used only after trajectory aggregation for construct-validity correlations.",
    }
    return report


def write_control_structural_recovery(path: str | Path, **kwargs) -> dict:
    report = run_control_structural_recovery(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
