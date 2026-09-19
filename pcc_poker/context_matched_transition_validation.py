"""Prospective fresh-seed validation of context-localized Control transitions.

This post-v0.8 analysis freezes two hypotheses discovered by the exploratory
history-structure probe:

1. unordered bet-call adjacency contributes positively to Control-heavy mixture
   recovery when the second action occurs while facing a bet (either round);
2. unordered bet-check adjacency contributes positively in round 1 while not
   facing a bet.

The existing ordinary mixed-recovery dataset is used as a fixed training set.
Every requested seed generates a fresh Control-heavy OOD evaluation set.  The
analysis also asks whether seat-specific ablation-effect asymmetry is smaller
under the matched context than under the corresponding all-context feature.

No human data are accessed and no frozen v0.8 artifact is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from .analyze import load_jsonl
from .history_structure_probe import MIXED_PAIR_CATEGORIES
from .mixed import _metrics, _ridge_predict
from .mixed_ood import sample_ood_weights
from .mixed_ood_ablation import _nested_features
from .policies import MODES
from .simulate import simulate_match

PRIMARY_HYPOTHESES = (
    {
        "name": "facing_bet_bet_call",
        "pair": ("bet", "call"),
        "round_index": None,
        "facing_bet": True,
        "description": "bet-call adjacency while facing a bet in either round",
    },
    {
        "name": "round1_open_bet_check",
        "pair": ("bet", "check"),
        "round_index": 1,
        "facing_bet": False,
        "description": "bet-check adjacency in round 1 while not facing a bet",
    },
)

DEFAULT_SEEDS = (1101, 1102, 1103, 1104, 1105)
DEFAULT_MIN_POSITIVE_SEED_FRACTION = 0.80


def _context_matches(row: dict, round_index: int | None, facing_bet: bool | None) -> bool:
    if round_index is not None and row["round_index"] != round_index:
        return False
    if facing_bet is not None and (row["to_call"] > 0) != facing_bet:
        return False
    return True


def mixed_pair_block(
    rows: list[dict],
    round_index: int | None = None,
    facing_bet: bool | None = None,
) -> np.ndarray:
    """Return fixed-normalization unordered mixed-pair frequencies.

    A transition is assigned to the public context of its second action, matching
    the exploratory localization probe.  Frequencies are normalized by the
    number of mixed transitions in the selected context; an empty context yields
    an all-zero block.
    """
    ordered_by_hand: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        ordered_by_hand[row["hand_id"]].append(row)

    counts: Counter[tuple[str, str]] = Counter()
    total = 0
    for hand_rows in ordered_by_hand.values():
        hand_rows = sorted(hand_rows, key=lambda row: row["decision_index"])
        for left, right in zip(hand_rows, hand_rows[1:]):
            if not _context_matches(right, round_index, facing_bet):
                continue
            pair = tuple(sorted((left["action"], right["action"])))
            if pair in MIXED_PAIR_CATEGORIES:
                counts[pair] += 1
                total += 1

    if total == 0:
        return np.zeros(len(MIXED_PAIR_CATEGORIES), dtype=float)
    return np.asarray(
        [counts[pair] / total for pair in MIXED_PAIR_CATEGORIES],
        dtype=float,
    )


def make_examples(records: list[dict]) -> list[dict]:
    """Aggregate focal-policy trajectories into one example per mixture and seat."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        if row.get("is_focal_policy"):
            grouped[(row["mixture_id"], row["focal_seat"])].append(row)

    examples = []
    for (mixture_id, focal_seat), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (row["hand_id"], row["decision_index"]))
        m2 = _nested_features(rows)["M2"]
        target = np.asarray(
            [rows[0]["target_pcc_weights"][mode] for mode in MODES],
            dtype=float,
        )
        examples.append(
            {
                "mixture_id": mixture_id,
                "focal_seat": focal_seat,
                "target": target,
                "M2_features": m2,
                "global_mixed_features": np.concatenate([m2, mixed_pair_block(rows)]),
                "rows": rows,
            }
        )
    return examples


def _attach_hypothesis_features(examples: list[dict], hypothesis: dict) -> None:
    feature_name = hypothesis["name"] + "_features"
    pair_index = MIXED_PAIR_CATEGORIES.index(tuple(hypothesis["pair"]))
    for example in examples:
        block = mixed_pair_block(
            example["rows"],
            round_index=hypothesis["round_index"],
            facing_bet=hypothesis["facing_bet"],
        )
        full = np.concatenate([example["M2_features"], block])
        ablated = full.copy()
        ablated[len(example["M2_features"]) + pair_index] = 0.0
        example[feature_name] = full
        example[feature_name + "_ablated"] = ablated

        global_ablated = example["global_mixed_features"].copy()
        global_ablated[len(example["M2_features"]) + pair_index] = 0.0
        example[hypothesis["name"] + "_global_ablated"] = global_ablated


def generate_control_heavy_evaluation(
    seed: int,
    mixtures: int = 20,
    hands_per_seat: int = 100,
    focal_temperature: float = 0.35,
    reference_temperature: float = 0.35,
) -> tuple[list[dict], dict]:
    """Generate only the prespecified Control-heavy OOD region for one fresh seed."""
    if mixtures < 1:
        raise ValueError("mixtures must be positive")
    if hands_per_seat < 1:
        raise ValueError("hands_per_seat must be positive")

    rng = random.Random(seed)
    balanced = (1 / 3, 1 / 3, 1 / 3)
    records: list[dict] = []
    groups = []

    for mixture_index in range(mixtures):
        mixture_id = f"fresh-{seed}-control_heavy-{mixture_index:04d}"
        weights = sample_ood_weights("control_heavy", rng)
        simulation_seeds = []
        for focal_seat in (0, 1):
            simulation_seed = seed * 100_000 + mixture_index * 2 + focal_seat
            simulation_seeds.append(simulation_seed)
            policy_weights = [balanced, balanced]
            policy_weights[focal_seat] = weights
            labels = ["balanced_reference", "balanced_reference"]
            labels[focal_seat] = mixture_id
            temperatures = [reference_temperature, reference_temperature]
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
                row["target_pcc_weights"] = dict(zip(MODES, weights))
            records.extend(batch)
        groups.append(
            {
                "mixture_id": mixture_id,
                "weights": dict(zip(MODES, weights)),
                "simulation_seeds": simulation_seeds,
            }
        )

    return records, {
        "seed": seed,
        "region": "control_heavy",
        "mixtures": mixtures,
        "hands_per_seat": hands_per_seat,
        "total_hands": mixtures * hands_per_seat * 2,
        "focal_temperature": focal_temperature,
        "reference_temperature": reference_temperature,
        "groups": groups,
    }


def _effect_report(
    train: list[dict],
    test: list[dict],
    full_feature: str,
    ablated_feature: str,
) -> dict:
    truth = np.stack([example["target"] for example in test])
    full_prediction = _ridge_predict(train, test, full_feature)
    ablated_prediction = _ridge_predict(train, test, ablated_feature)
    full_metrics = _metrics(truth, full_prediction)
    ablated_metrics = _metrics(truth, ablated_prediction)

    by_seat = {}
    for seat in (0, 1):
        indices = [index for index, example in enumerate(test) if example["focal_seat"] == seat]
        seat_truth = truth[indices]
        seat_full = _metrics(seat_truth, full_prediction[indices])
        seat_ablated = _metrics(seat_truth, ablated_prediction[indices])
        by_seat[str(seat)] = {
            "examples": len(indices),
            "full_mae": seat_full["mae"],
            "ablated_mae": seat_ablated["mae"],
            "delta_mae": seat_ablated["mae"] - seat_full["mae"],
        }

    seat_gap = abs(by_seat["0"]["delta_mae"] - by_seat["1"]["delta_mae"])
    return {
        "examples": len(test),
        "full": full_metrics,
        "ablated": ablated_metrics,
        "delta_mae": ablated_metrics["mae"] - full_metrics["mae"],
        "positive_degradation": ablated_metrics["mae"] > full_metrics["mae"],
        "by_seat": by_seat,
        "absolute_seat_effect_gap": seat_gap,
    }


def analyze_seed(training_examples: list[dict], evaluation_records: list[dict]) -> dict:
    test = make_examples(evaluation_records)
    for hypothesis in PRIMARY_HYPOTHESES:
        _attach_hypothesis_features(training_examples, hypothesis)
        _attach_hypothesis_features(test, hypothesis)

    hypotheses = {}
    for hypothesis in PRIMARY_HYPOTHESES:
        name = hypothesis["name"]
        context = _effect_report(
            training_examples,
            test,
            name + "_features",
            name + "_features_ablated",
        )
        global_effect = _effect_report(
            training_examples,
            test,
            "global_mixed_features",
            name + "_global_ablated",
        )
        global_gap = global_effect["absolute_seat_effect_gap"]
        context_gap = context["absolute_seat_effect_gap"]
        hypotheses[name] = {
            "pair": list(hypothesis["pair"]),
            "round_index": hypothesis["round_index"],
            "facing_bet": hypothesis["facing_bet"],
            "description": hypothesis["description"],
            "context_matched": context,
            "all_context_comparator": global_effect,
            "seat_gap_attenuated": context_gap < global_gap,
            "seat_gap_ratio": (context_gap / global_gap) if global_gap > 0 else None,
        }
    return {"examples": len(test), "hypotheses": hypotheses}


def _aggregate_seed_results(seed_reports: list[dict], min_positive_seed_fraction: float) -> dict:
    aggregate = {}
    context_gaps = []
    global_gaps = []
    for hypothesis in PRIMARY_HYPOTHESES:
        name = hypothesis["name"]
        deltas = np.asarray(
            [report["analysis"]["hypotheses"][name]["context_matched"]["delta_mae"] for report in seed_reports],
            dtype=float,
        )
        positive_rate = float(np.mean(deltas > 0))
        pooled_context_gap = float(np.mean([
            report["analysis"]["hypotheses"][name]["context_matched"]["absolute_seat_effect_gap"]
            for report in seed_reports
        ]))
        pooled_global_gap = float(np.mean([
            report["analysis"]["hypotheses"][name]["all_context_comparator"]["absolute_seat_effect_gap"]
            for report in seed_reports
        ]))
        context_gaps.append(pooled_context_gap)
        global_gaps.append(pooled_global_gap)
        aggregate[name] = {
            "delta_mae_mean": float(deltas.mean()),
            "delta_mae_std": float(deltas.std()),
            "positive_seed_fraction": positive_rate,
            "minimum_positive_seed_fraction": min_positive_seed_fraction,
            "positive_mean_effect": float(deltas.mean()) > 0,
            "replicates_in_required_fraction": positive_rate >= min_positive_seed_fraction,
            "primary_replication_passed": (
                float(deltas.mean()) > 0
                and positive_rate >= min_positive_seed_fraction
            ),
            "mean_context_seat_effect_gap": pooled_context_gap,
            "mean_all_context_seat_effect_gap": pooled_global_gap,
            "seat_gap_attenuated": pooled_context_gap < pooled_global_gap,
        }

    mean_context_gap = float(np.mean(context_gaps))
    mean_global_gap = float(np.mean(global_gaps))
    return {
        "hypotheses": aggregate,
        "seat_asymmetry": {
            "mean_context_seat_effect_gap": mean_context_gap,
            "mean_all_context_seat_effect_gap": mean_global_gap,
            "attenuation_ratio": (mean_context_gap / mean_global_gap) if mean_global_gap > 0 else None,
            "attenuated_after_context_matching": mean_context_gap < mean_global_gap,
        },
        "all_primary_replications_passed": all(
            item["primary_replication_passed"] for item in aggregate.values()
        ),
    }


def run_context_matched_transition_validation(
    training_records: list[dict],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    mixtures_per_seed: int = 20,
    hands_per_seat: int = 100,
    focal_temperature: float = 0.35,
    reference_temperature: float = 0.35,
    min_positive_seed_fraction: float = DEFAULT_MIN_POSITIVE_SEED_FRACTION,
) -> dict:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if not 0 < min_positive_seed_fraction <= 1:
        raise ValueError("min_positive_seed_fraction must be in (0, 1]")

    train = make_examples(training_records)
    if not train:
        raise ValueError("training_records contain no focal-policy examples")

    seed_reports = []
    for seed in seeds:
        evaluation_records, design = generate_control_heavy_evaluation(
            seed=seed,
            mixtures=mixtures_per_seed,
            hands_per_seat=hands_per_seat,
            focal_temperature=focal_temperature,
            reference_temperature=reference_temperature,
        )
        seed_reports.append(
            {
                "seed": seed,
                "design": design,
                "analysis": analyze_seed(train, evaluation_records),
            }
        )

    return {
        "status": "completed",
        "analysis_status": "prospective_post_v0.8_fresh_seed_validation",
        "training_distribution": "fixed ordinary Dirichlet mixed-recovery dataset",
        "evaluation_distribution": "fresh Control-heavy OOD mixtures",
        "seeds": list(seeds),
        "mixtures_per_seed": mixtures_per_seed,
        "hands_per_seat": hands_per_seat,
        "focal_temperature": focal_temperature,
        "reference_temperature": reference_temperature,
        "prespecified_hypotheses": [
            {
                **hypothesis,
                "pair": list(hypothesis["pair"]),
            }
            for hypothesis in PRIMARY_HYPOTHESES
        ],
        "falsification_rule": {
            "effect": "ablating the prespecified pair must increase MAE",
            "replication": "mean delta MAE > 0 and the required fraction of fresh seeds has delta MAE > 0",
            "minimum_positive_seed_fraction": min_positive_seed_fraction,
            "seat_asymmetry": "absolute seat-specific ablation-effect gap must be smaller after context matching than in the all-context comparator",
        },
        "seed_results": seed_reports,
        "aggregate": _aggregate_seed_results(seed_reports, min_positive_seed_fraction),
        "provenance": {
            "human_data_accessed": False,
            "frozen_v0.8_artifacts_modified": False,
            "threshold_tuned_after_results": False,
        },
        "warning": (
            "This is a synthetic construct-validation experiment. Positive fresh-seed "
            "replication would not establish that human poker behavior follows PCC."
        ),
    }


def write_context_matched_transition_validation(
    output_path: str | Path,
    training_path: str | Path = "outputs/mixed-recovery-data.jsonl",
    **kwargs,
) -> dict:
    training_records = load_jsonl(training_path)
    report = run_context_matched_transition_validation(training_records, **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh-seed matched-context validation of Control-heavy transition signals"
    )
    parser.add_argument(
        "--training",
        default="outputs/mixed-recovery-data.jsonl",
        help="fixed ordinary-mixture training JSONL",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--mixtures-per-seed", type=int, default=20)
    parser.add_argument("--hands-per-seat", type=int, default=100)
    parser.add_argument("--focal-temperature", type=float, default=0.35)
    parser.add_argument("--reference-temperature", type=float, default=0.35)
    parser.add_argument(
        "--min-positive-seed-fraction",
        type=float,
        default=DEFAULT_MIN_POSITIVE_SEED_FRACTION,
    )
    parser.add_argument(
        "--output",
        default="validation/context-matched-transition-validation.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = write_context_matched_transition_validation(
        args.output,
        training_path=args.training,
        seeds=args.seeds,
        mixtures_per_seed=args.mixtures_per_seed,
        hands_per_seat=args.hands_per_seat,
        focal_temperature=args.focal_temperature,
        reference_temperature=args.reference_temperature,
        min_positive_seed_fraction=args.min_positive_seed_fraction,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "seeds": report["seeds"],
                "all_primary_replications_passed": report["aggregate"]["all_primary_replications_passed"],
                "seat_asymmetry": report["aggregate"]["seat_asymmetry"],
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
