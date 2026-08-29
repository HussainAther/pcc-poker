"""Out-of-distribution evaluation of continuous PCC mixture recovery."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np

from .analyze import load_jsonl
from .mixed import (
    _metrics,
    _ridge_predict,
    mixture_examples,
)
from .policies import MODES
from .simulate import simulate_match, write_jsonl


REGIONS = (
    "pressure_heavy",
    "control_heavy",
    "chaos_heavy",
    "balanced",
    "boundary",
)


def sample_ood_weights(
    region: str,
    rng: random.Random,
) -> tuple[float, float, float]:
    """Sample a PCC weight vector from a prespecified OOD simplex region."""

    if region == "pressure_heavy":
        pressure = rng.uniform(0.80, 0.95)
        remainder = 1.0 - pressure
        control = rng.uniform(0.0, remainder)
        chaos = remainder - control
        return pressure, control, chaos

    if region == "control_heavy":
        control = rng.uniform(0.80, 0.95)
        remainder = 1.0 - control
        pressure = rng.uniform(0.0, remainder)
        chaos = remainder - pressure
        return pressure, control, chaos

    if region == "chaos_heavy":
        chaos = rng.uniform(0.80, 0.95)
        remainder = 1.0 - chaos
        pressure = rng.uniform(0.0, remainder)
        control = remainder - pressure
        return pressure, control, chaos

    if region == "balanced":
        # Deliberately concentrated near the center of the simplex.
        while True:
            values = np.asarray(
                [
                    rng.gammavariate(20.0, 1.0),
                    rng.gammavariate(20.0, 1.0),
                    rng.gammavariate(20.0, 1.0),
                ],
                dtype=float,
            )
            values /= values.sum()

            if np.max(np.abs(values - (1.0 / 3.0))) <= 0.08:
                return tuple(float(value) for value in values)

    if region == "boundary":
        # Force exactly one objective close to zero.
        small_index = rng.randrange(3)
        small = rng.uniform(0.0, 0.05)

        remainder = 1.0 - small
        split = rng.uniform(0.15, 0.85)

        values = [0.0, 0.0, 0.0]
        values[small_index] = small

        other_indices = [
            index for index in range(3)
            if index != small_index
        ]

        values[other_indices[0]] = remainder * split
        values[other_indices[1]] = remainder * (1.0 - split)

        return tuple(values)

    raise ValueError(
        f"unknown OOD region {region!r}; choices={REGIONS}"
    )


def generate_mixed_ood_dataset(
    mixtures_per_region: int = 20,
    hands_per_seat: int = 100,
    seed: int = 51,
    focal_temperature: float = 0.35,
    reference_temperature: float = 0.35,
) -> tuple[list[dict], dict]:
    """Generate prespecified OOD PCC mixtures in both seats."""

    if mixtures_per_region < 1:
        raise ValueError("mixtures_per_region must be positive")

    if hands_per_seat < 1:
        raise ValueError("hands_per_seat must be positive")

    rng = random.Random(seed)
    balanced_reference = (1 / 3, 1 / 3, 1 / 3)

    records: list[dict] = []
    groups: list[dict] = []

    global_index = 0

    for region in REGIONS:
        for region_index in range(mixtures_per_region):
            mixture_id = f"ood-{region}-{region_index:04d}"
            weights = sample_ood_weights(region, rng)

            group_seeds = []

            for focal_seat in (0, 1):
                simulation_seed = (
                    seed * 100_000
                    + global_index * 2
                    + focal_seat
                )
                group_seeds.append(simulation_seed)

                policy_weights = [
                    balanced_reference,
                    balanced_reference,
                ]
                policy_labels = [
                    "balanced_reference",
                    "balanced_reference",
                ]

                policy_weights[focal_seat] = weights
                policy_labels[focal_seat] = mixture_id

                temperatures = [
                    reference_temperature,
                    reference_temperature,
                ]
                temperatures[focal_seat] = focal_temperature

                batch, _ = simulate_match(
                    hands_per_seat,
                    policy_weights[0],
                    policy_weights[1],
                    simulation_seed,
                    policy_labels[0],
                    policy_labels[1],
                    temperatures[0],
                    temperatures[1],
                )

                for record in batch:
                    record["mixture_id"] = mixture_id
                    record["ood_region"] = region
                    record["simulation_seed"] = simulation_seed
                    record["focal_seat"] = focal_seat
                    record["is_focal_policy"] = (
                        record["actor"] == focal_seat
                    )
                    record["target_pcc_weights"] = dict(
                        zip(MODES, weights)
                    )

                records.extend(batch)

            groups.append(
                {
                    "mixture_id": mixture_id,
                    "ood_region": region,
                    "weights": dict(zip(MODES, weights)),
                    "simulation_seeds": group_seeds,
                }
            )

            global_index += 1

    return records, {
        "status": "completed",
        "design": "prespecified_ood_simplex_regions",
        "regions": list(REGIONS),
        "mixtures_per_region": mixtures_per_region,
        "total_mixtures": mixtures_per_region * len(REGIONS),
        "hands_per_seat": hands_per_seat,
        "total_hands": (
            mixtures_per_region
            * len(REGIONS)
            * hands_per_seat
            * 2
        ),
        "seed": seed,
        "focal_temperature": focal_temperature,
        "reference_temperature": reference_temperature,
        "groups": groups,
    }


def write_mixed_ood_dataset(
    output_path: str | Path,
    **kwargs,
) -> dict:
    records, summary = generate_mixed_ood_dataset(**kwargs)

    write_jsonl(output_path, records)

    summary_path = Path(output_path).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    return summary


def analyze_mixed_ood(
    training_records: list[dict],
    ood_records: list[dict],
) -> dict:
    """Train on ordinary mixtures and evaluate only on prespecified OOD mixtures."""

    train = mixture_examples(training_records)
    test = mixture_examples(ood_records)

    if not train or not test:
        return {
            "status": "insufficient_examples",
            "train_examples": len(train),
            "test_examples": len(test),
        }

    truth = np.stack([row["target"] for row in test])

    action_prediction = _ridge_predict(
        train,
        test,
        "action_features",
    )

    contextual_prediction = _ridge_predict(
        train,
        test,
        "contextual_features",
    )

    action_report = _metrics(truth, action_prediction)
    contextual_report = _metrics(truth, contextual_prediction)

    # Map each OOD mixture back to its prespecified simplex region.
    region_by_mixture = {}

    for record in ood_records:
        mixture_id = record.get("mixture_id")
        region = record.get("ood_region")

        if mixture_id is not None and region is not None:
            region_by_mixture[mixture_id] = region

    regional_indices = defaultdict(list)

    for index, example in enumerate(test):
        region = region_by_mixture.get(example["mixture_id"])

        if region is not None:
            regional_indices[region].append(index)

    regional_reports = {}

    for region in REGIONS:
        indices = regional_indices.get(region, [])

        if not indices:
            regional_reports[region] = {
                "examples": 0,
            }
            continue

        region_truth = truth[indices]
        region_action = action_prediction[indices]
        region_contextual = contextual_prediction[indices]

        regional_reports[region] = {
            "examples": len(indices),
            "action_frequency_baseline": _metrics(
                region_truth,
                region_action,
            ),
            "contextual_history_model": _metrics(
                region_truth,
                region_contextual,
            ),
        }

        action_mae = regional_reports[region][
            "action_frequency_baseline"
        ]["mae"]

        contextual_mae = regional_reports[region][
            "contextual_history_model"
        ]["mae"]

        regional_reports[region][
            "contextual_below_action_frequency"
        ] = contextual_mae < action_mae

    return {
        "status": "completed",
        "prediction_target": (
            "continuous_pressure_control_chaos_weights"
        ),
        "training_distribution": (
            "ordinary_dirichlet_mixtures"
        ),
        "evaluation_distribution": (
            "prespecified_ood_simplex_regions"
        ),
        "observable_features_only": True,
        "train_examples": len(train),
        "ood_test_examples": len(test),
        "overall": {
            "action_frequency_baseline": action_report,
            "contextual_history_model": contextual_report,
            "relative_mae_improvement_over_action_frequency": (
                float(
                    (
                        action_report["mae"]
                        - contextual_report["mae"]
                    )
                    / action_report["mae"]
                )
            ),
            "contextual_mae_below_action_frequency": (
                contextual_report["mae"]
                < action_report["mae"]
            ),
        },
        "by_region": regional_reports,
        "prespecified_checks": {
            "contextual_beats_action_frequency_overall": (
                contextual_report["mae"]
                < action_report["mae"]
            ),
            "contextual_beats_action_frequency_in_majority_of_regions": (
                sum(
                    bool(
                        regional_reports[region].get(
                            "contextual_below_action_frequency",
                            False,
                        )
                    )
                    for region in REGIONS
                )
                >= 3
            ),
        },
        "warning": (
            "OOD recovery concerns synthetic policies within the "
            "engineered PCC family. It does not establish that "
            "human behavior follows PCC."
        ),
    }


def analyze_mixed_ood_files(
    training_path: str | Path,
    ood_path: str | Path,
    output_path: str | Path,
) -> dict:
    report = analyze_mixed_ood(
        load_jsonl(training_path),
        load_jsonl(ood_path),
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    return report
