"""Prospective seat-yoked validation of the facing-bet bet-call Control signal.

Motivation
----------
Fresh-seed context-matched validation replicated the facing-bet bet-call
transition effect, but context matching did not attenuate the seat-specific
ablation-effect gap.

This experiment asks whether that gap persists when the exact same mixture
weights are evaluated in both focal seats.

No human data are accessed and no previously frozen artifact is modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from .analyze import load_jsonl
from .context_matched_transition_validation import (
    mixed_pair_block,
    make_examples,
)
from .history_structure_probe import MIXED_PAIR_CATEGORIES
from .mixed import _metrics, _ridge_predict
from .mixed_ood import sample_ood_weights
from .mixed_ood_ablation import _nested_features
from .policies import MODES
from .simulate import simulate_match


DEFAULT_SEEDS = (1201, 1202, 1203, 1204, 1205)

TARGET_PAIR = ("bet", "call")
TARGET_FACING_BET = True

DEFAULT_MIN_POSITIVE_SEED_FRACTION = 0.80


def generate_seat_yoked_evaluation(
    seed: int,
    mixtures: int = 20,
    hands_per_seat: int = 100,
    focal_temperature: float = 0.35,
    reference_temperature: float = 0.35,
) -> tuple[list[dict], dict]:
    """Evaluate each sampled mixture in both focal seats.

    A single mixture draw is reused for seat 0 and seat 1. This yokes the
    latent PCC target across seats while keeping independent simulations.
    """

    if mixtures < 1:
        raise ValueError("mixtures must be positive")

    if hands_per_seat < 1:
        raise ValueError("hands_per_seat must be positive")

    rng = random.Random(seed)

    balanced = (1 / 3, 1 / 3, 1 / 3)

    records: list[dict] = []
    groups = []

    for mixture_index in range(mixtures):
        mixture_id = f"seat-yoked-{seed}-{mixture_index:04d}"

        # IMPORTANT:
        # one draw here, outside the focal-seat loop
        weights = sample_ood_weights("control_heavy", rng)

        simulation_seeds = []

        for focal_seat in (0, 1):
            simulation_seed = (
                seed * 100_000
                + mixture_index * 2
                + focal_seat
            )

            simulation_seeds.append(simulation_seed)

            policy_weights = [balanced, balanced]
            policy_weights[focal_seat] = weights

            labels = ["balanced_reference", "balanced_reference"]
            labels[focal_seat] = mixture_id

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
                labels[0],
                labels[1],
                temperatures[0],
                temperatures[1],
            )

            for row in batch:
                row["mixture_id"] = mixture_id
                row["ood_region"] = "control_heavy"
                row["simulation_seed"] = simulation_seed
                row["focal_seat"] = focal_seat
                row["is_focal_policy"] = row["actor"] == focal_seat
                row["target_pcc_weights"] = dict(
                    zip(MODES, weights)
                )

                # Makes the yoking explicit in the artifact.
                row["seat_yoke_id"] = mixture_id

            records.extend(batch)

        groups.append(
            {
                "mixture_id": mixture_id,
                "seat_yoke_id": mixture_id,
                "weights": dict(zip(MODES, weights)),
                "simulation_seeds": simulation_seeds,
            }
        )

    return records, {
        "seed": seed,
        "region": "control_heavy",
        "design": "same mixture weights evaluated in both focal seats",
        "mixtures": mixtures,
        "hands_per_seat": hands_per_seat,
        "total_hands": mixtures * hands_per_seat * 2,
        "groups": groups,
    }


def attach_bet_call_features(examples: list[dict]) -> None:
    """Attach the prespecified facing-bet bet-call block and its ablation.

    The transition is assigned to the public context of the second action,
    exactly as in the preceding context-matched validation.  Only the
    bet-call coordinate is zeroed for the ablated representation.
    """

    pair_index = MIXED_PAIR_CATEGORIES.index(TARGET_PAIR)

    for example in examples:
        block = mixed_pair_block(
            example["rows"],
            round_index=None,
            facing_bet=TARGET_FACING_BET,
        )
        full = np.concatenate([example["M2_features"], block])
        ablated = full.copy()
        ablated[len(example["M2_features"]) + pair_index] = 0.0

        example["bet_call_features"] = full
        example["bet_call_ablated"] = ablated


def evaluate_seat_effect(
    training_examples: list[dict],
    evaluation_examples: list[dict],
) -> dict:
    """Measure facing-bet bet-call ablation independently by focal seat."""

    truth = np.stack([
        example["target"]
        for example in evaluation_examples
    ])

    full_prediction = _ridge_predict(
        training_examples,
        evaluation_examples,
        "bet_call_features",
    )

    ablated_prediction = _ridge_predict(
        training_examples,
        evaluation_examples,
        "bet_call_ablated",
    )

    overall_full = _metrics(
        truth,
        full_prediction,
    )

    overall_ablated = _metrics(
        truth,
        ablated_prediction,
    )

    by_seat = {}

    for seat in (0, 1):
        indices = [
            index
            for index, example
            in enumerate(evaluation_examples)
            if example["focal_seat"] == seat
        ]

        seat_truth = truth[indices]

        seat_full = _metrics(
            seat_truth,
            full_prediction[indices],
        )

        seat_ablated = _metrics(
            seat_truth,
            ablated_prediction[indices],
        )

        by_seat[str(seat)] = {
            "examples": len(indices),
            "full_mae": seat_full["mae"],
            "ablated_mae": seat_ablated["mae"],
            "delta_mae": (
                seat_ablated["mae"]
                - seat_full["mae"]
            ),
        }

    delta_seat_0 = by_seat["0"]["delta_mae"]
    delta_seat_1 = by_seat["1"]["delta_mae"]

    return {
        "overall": {
            "full": overall_full,
            "ablated": overall_ablated,
            "delta_mae": (
                overall_ablated["mae"]
                - overall_full["mae"]
            ),
        },
        "by_seat": by_seat,
        "seat_effect_difference": (
            delta_seat_0 - delta_seat_1
        ),
        "absolute_seat_effect_gap": abs(
            delta_seat_0 - delta_seat_1
        ),
    }

def analyze_seed(
    training_examples: list[dict],
    evaluation_records: list[dict],
) -> dict:

    test = make_examples(evaluation_records)

    attach_bet_call_features(training_examples)
    attach_bet_call_features(test)

    result = evaluate_seat_effect(
        training_examples,
        test,
    )

    return {
        "examples": len(test),
        "transition": {
            "pair": list(TARGET_PAIR),
            "facing_bet": TARGET_FACING_BET,
        },
        "effect": result,
    }

def aggregate_seed_results(
    seed_reports: list[dict],
) -> dict:

    seat_0_effects = np.asarray([
        report["analysis"]["effect"]["by_seat"]["0"]["delta_mae"]
        for report in seed_reports
    ])

    seat_1_effects = np.asarray([
        report["analysis"]["effect"]["by_seat"]["1"]["delta_mae"]
        for report in seed_reports
    ])

    differences = seat_0_effects - seat_1_effects
    gaps = np.abs(differences)

    return {
        "seat_0": {
            "mean_delta_mae": float(
                seat_0_effects.mean()
            ),
            "positive_seed_fraction": float(
                np.mean(seat_0_effects > 0)
            ),
        },
        "seat_1": {
            "mean_delta_mae": float(
                seat_1_effects.mean()
            ),
            "positive_seed_fraction": float(
                np.mean(seat_1_effects > 0)
            ),
        },
        "seat_difference": {
            "mean_signed_difference": float(
                differences.mean()
            ),
            "mean_absolute_gap": float(
                gaps.mean()
            ),
            "same_direction_fraction": float(
                np.mean(
                    np.sign(seat_0_effects)
                    == np.sign(seat_1_effects)
                )
            ),
        },
    }

def run_seat_yoked_transition_validation(
    training_records: list[dict],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    mixtures_per_seed: int = 20,
    hands_per_seat: int = 100,
) -> dict:

    seeds = tuple(int(seed) for seed in seeds)

    if not seeds:
        raise ValueError("at least one seed is required")

    train = make_examples(training_records)

    if not train:
        raise ValueError(
            "training_records contain no focal-policy examples"
        )

    seed_reports = []

    for seed in seeds:
        records, design = generate_seat_yoked_evaluation(
            seed=seed,
            mixtures=mixtures_per_seed,
            hands_per_seat=hands_per_seat,
        )

        seed_reports.append(
            {
                "seed": seed,
                "design": design,
                "analysis": analyze_seed(
                    train,
                    records,
                ),
            }
        )

    return {
        "status": "completed",
        "analysis_status": (
            "prospective_seat_yoked_transition_validation"
        ),
        "target_transition": {
            "pair": list(TARGET_PAIR),
            "facing_bet": True,
            "round_index": None,
        },
        "seeds": list(seeds),
        "mixtures_per_seed": mixtures_per_seed,
        "hands_per_seat": hands_per_seat,
        "seed_results": seed_reports,
        "aggregate": aggregate_seed_results(
            seed_reports
        ),
        "provenance": {
            "human_data_accessed": False,
            "previous_frozen_artifacts_modified": False,
            "threshold_tuned_after_results": False,
        },
        "warning": (
            "Synthetic construct validation only. "
            "Seat effects in this simulation do not establish "
            "human poker behavior."
        ),
    }

def write_seat_yoked_transition_validation(
    output_path: str | Path,
    training_path: str | Path = (
        "outputs/mixed-recovery-data.jsonl"
    ),
    **kwargs,
) -> dict:

    training_records = load_jsonl(
        training_path
    )

    report = run_seat_yoked_transition_validation(
        training_records,
        **kwargs,
    )

    target = Path(output_path)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-seed seat-yoked validation "
            "of the facing-bet bet-call signal"
        )
    )

    parser.add_argument(
        "--training",
        default="outputs/mixed-recovery-data.jsonl",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )

    parser.add_argument(
        "--mixtures-per-seed",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--hands-per-seat",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        default=(
            "validation/"
            "seat-yoked-transition-validation.json"
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:

    args = _build_parser().parse_args(argv)

    report = (
        write_seat_yoked_transition_validation(
            args.output,
            training_path=args.training,
            seeds=args.seeds,
            mixtures_per_seed=args.mixtures_per_seed,
            hands_per_seat=args.hands_per_seat,
        )
    )

    print(
        json.dumps(
            {
                "status": report["status"],
                "seeds": report["seeds"],
                "aggregate": report["aggregate"],
                "output": args.output,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
