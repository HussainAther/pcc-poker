"""Fresh-seed construct validation for the independent effective-Chaos candidate.

The observable combines action surprisal from a separately calibrated public
behavior model with the fixed uniform-continuation value floor in value_floor.py.
Hidden PCC weights are used only after scoring, as construct-validity targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from collections import defaultdict

import numpy as np

from .behavioral import PublicActionModel
from .policies import MODES
from .simulate import generate_family_dataset
from .value_floor import UniformContinuationValueModel, measure_synthetic_effective_chaos

DEFAULT_CALIBRATION_SEEDS = {"score": 1401, "independent": 1409}
DEFAULT_EVALUATION_SEEDS = {"score": 1601, "independent": 1609}
MIN_CHAOS_CORRELATION = 0.20
MIN_DISCRIMINANT_MARGIN = 0.05
RAW_MARGIN_NONINFERIORITY = 0.02
MIN_ANY_FAMILY_MARGIN_GAIN = 0.02


def _correlation(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@dataclass(frozen=True)
class ChaosMeasurement:
    raw_normalized_surprisal: float
    performance_adequacy: float
    independent_effective_surprisal: float
    value_regret: float

    def as_dict(self) -> dict[str, float]:
        return {
            "raw_normalized_surprisal": self.raw_normalized_surprisal,
            "performance_adequacy": self.performance_adequacy,
            "independent_effective_surprisal": self.independent_effective_surprisal,
            "value_regret": self.value_regret,
        }


class IndependentChaosOracle:
    """Frozen public-surprisal model plus non-PCC information-set value floor."""

    def __init__(self, action_model: PublicActionModel) -> None:
        self.action_model = action_model
        self.value_model = UniformContinuationValueModel()

    def measure(self, state, chosen_action: str) -> ChaosMeasurement:
        probability = self.action_model.probabilities(state)[chosen_action]
        surprisal = -math.log(max(probability, 1e-12))
        legal_count = len(state.legal_actions())
        normalized = surprisal / math.log(legal_count) if legal_count > 1 else 0.0
        scored = measure_synthetic_effective_chaos(
            state,
            chosen_action,
            normalized,
            value_model=self.value_model,
        )
        return ChaosMeasurement(
            raw_normalized_surprisal=normalized,
            performance_adequacy=scored.adequacy,
            independent_effective_surprisal=scored.effective_surprisal,
            value_regret=scored.regret,
        )


def aggregate_effective_chaos(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("is_focal_policy") and "behavioral_measurements" in record:
            grouped[(record["policy_family"], record["mixture_id"], record["focal_seat"])].append(record)
    rows = []
    for (family, mixture_id, focal_seat), decisions in sorted(grouped.items()):
        weights = decisions[0]["target_pcc_weights"]
        rows.append({
            "policy_family": family,
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "weights": {mode: float(weights[mode]) for mode in MODES},
            "raw_normalized_surprisal": float(np.mean([
                d["behavioral_measurements"]["raw_normalized_surprisal"] for d in decisions
            ])),
            "performance_adequacy": float(np.mean([
                d["behavioral_measurements"]["performance_adequacy"] for d in decisions
            ])),
            "independent_effective_surprisal": float(np.mean([
                d["behavioral_measurements"]["independent_effective_surprisal"] for d in decisions
            ])),
            "decisions": len(decisions),
        })
    return rows


def _metric_correlations(rows: list[dict], metric: str) -> dict[str, float]:
    return {
        mode: _correlation(
            [row[metric] for row in rows],
            [row["weights"][mode] for row in rows],
        )
        for mode in MODES
    }


def _margin(correlations: dict[str, float]) -> float:
    return correlations["chaos"] - max(correlations["pressure"], correlations["control"])


def _shuffled_chaos_correlation(rows: list[dict], seed: int) -> float:
    mixture_ids = sorted({row["mixture_id"] for row in rows})
    source = mixture_ids.copy()
    random.Random(seed).shuffle(source)
    weight_by_id = {}
    first_by_id = {mid: next(row for row in rows if row["mixture_id"] == mid) for mid in mixture_ids}
    for target_id, source_id in zip(mixture_ids, source):
        weight_by_id[target_id] = first_by_id[source_id]["weights"]["chaos"]
    return _correlation(
        [row["independent_effective_surprisal"] for row in rows],
        [weight_by_id[row["mixture_id"]] for row in rows],
    )


def summarize_effective_chaos_validation(records: list[dict], *, shuffle_seed: int = 1901) -> dict:
    rows = aggregate_effective_chaos(records)
    families = sorted({row["policy_family"] for row in rows})
    by_family = {}
    margin_gains = []
    for index, family in enumerate(families):
        subset = [row for row in rows if row["policy_family"] == family]
        raw = _metric_correlations(subset, "raw_normalized_surprisal")
        effective = _metric_correlations(subset, "independent_effective_surprisal")
        raw_margin = _margin(raw)
        effective_margin = _margin(effective)
        margin_gain = effective_margin - raw_margin
        margin_gains.append(margin_gain)
        shuffled = _shuffled_chaos_correlation(subset, shuffle_seed + index)
        checks = {
            "chaos_correlation_at_least_0_20": effective["chaos"] >= MIN_CHAOS_CORRELATION,
            "chaos_exceeds_off_axes_by_at_least_0_05": effective_margin >= MIN_DISCRIMINANT_MARGIN,
            "value_floor_not_materially_worse_than_raw_margin": effective_margin >= raw_margin - RAW_MARGIN_NONINFERIORITY,
        }
        by_family[family] = {
            "mixtures": len({row["mixture_id"] for row in subset}),
            "seat_level_examples": len(subset),
            "mean_decisions_per_example": float(np.mean([row["decisions"] for row in subset])),
            "mean_performance_adequacy": float(np.mean([row["performance_adequacy"] for row in subset])),
            "raw_surprisal_weight_correlations": raw,
            "effective_surprisal_weight_correlations": effective,
            "raw_discriminant_margin": raw_margin,
            "effective_discriminant_margin": effective_margin,
            "value_floor_margin_gain": margin_gain,
            "shuffled_chaos_weight_correlation": shuffled,
            "prespecified_checks": checks,
        }
    cross_family_checks = {
        "all_families_positive_and_descriptively_meaningful": all(
            result["prespecified_checks"]["chaos_correlation_at_least_0_20"]
            for result in by_family.values()
        ),
        "all_families_discriminant": all(
            result["prespecified_checks"]["chaos_exceeds_off_axes_by_at_least_0_05"]
            for result in by_family.values()
        ),
        "value_floor_noninferior_in_all_families": all(
            result["prespecified_checks"]["value_floor_not_materially_worse_than_raw_margin"]
            for result in by_family.values()
        ),
        "value_floor_improves_margin_in_at_least_one_family": any(
            gain >= MIN_ANY_FAMILY_MARGIN_GAIN for gain in margin_gains
        ),
    }
    confirmed = all(cross_family_checks.values())
    return {
        "effective_chaos_construct_confirmed": confirmed,
        "families": by_family,
        "prespecified_checks": cross_family_checks,
        "thresholds": {
            "minimum_chaos_correlation": MIN_CHAOS_CORRELATION,
            "minimum_discriminant_margin": MIN_DISCRIMINANT_MARGIN,
            "raw_margin_noninferiority_tolerance": RAW_MARGIN_NONINFERIORITY,
            "minimum_margin_gain_in_at_least_one_family": MIN_ANY_FAMILY_MARGIN_GAIN,
        },
        "interpretation": (
            "Confirmation supports this exact synthetic effective-surprisal operationalization "
            "as a Chaos candidate across both engineered policy families. Failure is retained "
            "without retuning thresholds or the value floor."
        ),
        "shuffled_label_baseline_is_diagnostic_not_acceptance_criterion": True,
    }


def run_effective_chaos_validation(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 25,
    evaluation_mixtures: int = 60,
    evaluation_hands_per_seat: int = 100,
    score_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["score"],
    independent_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["independent"],
    score_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["score"],
    independent_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["independent"],
    shuffle_seed: int = 1901,
) -> dict:
    calibration_records = []
    for family, seed in (("score", score_calibration_seed), ("independent", independent_calibration_seed)):
        records, _ = generate_family_dataset(
            family, calibration_mixtures, calibration_hands_per_seat, seed
        )
        calibration_records.extend(records)
    oracle = IndependentChaosOracle(PublicActionModel.from_records(calibration_records))

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

    report = summarize_effective_chaos_validation(evaluation_records, shuffle_seed=shuffle_seed)
    report["design"] = {
        "candidate_frozen_before_evaluation": True,
        "calibration_seeds": {"score": score_calibration_seed, "independent": independent_calibration_seed},
        "evaluation_seeds": {"score": score_evaluation_seed, "independent": independent_evaluation_seed},
        "calibration_evaluation_overlap": False,
        "calibration_mixtures_per_family": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures_per_family": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "surprisal_model": "smoothed public-action frequencies from disjoint calibration hands",
        "value_model": "exact Leduc information-set Q-values under uniform legal-action continuation",
        "constant_floor_baseline": "raw normalized surprisal (adequacy fixed to one)",
        "human_data_used": False,
    }
    return report


def write_effective_chaos_validation(output_path: str | Path, **kwargs) -> dict:
    report = run_effective_chaos_validation(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
