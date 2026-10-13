"""Prospective paired matched-opportunity validation for the seat effect."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .analyze import load_jsonl
from .context_matched_transition_validation import make_examples, mixed_pair_block
from .history_structure_probe import MIXED_PAIR_CATEGORIES
from .mixed import _metrics, _ridge_predict
from .seat_opportunity_decomposition import _eligible_transitions, _stable_transition_key
from .seat_yoked_transition_validation import generate_seat_yoked_evaluation

DEFAULT_SEEDS = (1401, 1402, 1403, 1404, 1405)
TARGET_PAIR = ("bet", "call")
PAIR_INDEX = MIXED_PAIR_CATEGORIES.index(TARGET_PAIR)


def _pair_examples(examples: list[dict]) -> list[tuple[dict, dict]]:
    """Return seat-0/seat-1 example pairs sharing the same mixture id."""
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for example in examples:
        mixture_id = str(example["mixture_id"])
        seat = int(example["focal_seat"])
        if seat in grouped[mixture_id]:
            raise ValueError(f"duplicate seat {seat} for mixture {mixture_id}")
        grouped[mixture_id][seat] = example

    pairs = []
    for mixture_id in sorted(grouped):
        seats = grouped[mixture_id]
        if set(seats) != {0, 1}:
            raise ValueError(f"mixture {mixture_id} is not paired across seats")
        pairs.append((seats[0], seats[1]))
    return pairs


def _selected_block(transitions: list[dict], budgets: dict[int, int]) -> tuple[np.ndarray, dict]:
    selected = []
    eligible_by_round = {}
    selected_by_round = {}

    for round_index in (0, 1):
        candidates = [
            item for item in transitions
            if int(item["round_index"]) == round_index
        ]
        candidates.sort(key=_stable_transition_key)
        budget = int(budgets[round_index])
        eligible_by_round[str(round_index)] = len(candidates)
        chosen = candidates[:budget]
        selected_by_round[str(round_index)] = len(chosen)
        selected.extend(chosen)

    counts = Counter(item["pair"] for item in selected)
    total = len(selected)
    if total == 0:
        block = np.zeros(len(MIXED_PAIR_CATEGORIES), dtype=float)
    else:
        block = np.asarray(
            [counts[pair] / total for pair in MIXED_PAIR_CATEGORIES],
            dtype=float,
        )

    return block, {
        "eligible_total": len(transitions),
        "eligible_by_round": eligible_by_round,
        "selected_total": total,
        "selected_by_round": selected_by_round,
        "matched_budgets": {str(k): int(v) for k, v in budgets.items()},
        "matched_bet_call_share": float(block[PAIR_INDEX]),
    }


def attach_paired_features(examples: list[dict]) -> None:
    """Attach ordinary and exactly opportunity-matched features to paired examples."""
    for seat0, seat1 in _pair_examples(examples):
        transitions = {
            0: _eligible_transitions(seat0["rows"]),
            1: _eligible_transitions(seat1["rows"]),
        }
        budgets = {}
        for round_index in (0, 1):
            counts = [
                sum(int(item["round_index"]) == round_index for item in transitions[seat])
                for seat in (0, 1)
            ]
            budgets[round_index] = min(counts)

        for seat, example in ((0, seat0), (1, seat1)):
            ordinary = mixed_pair_block(example["rows"], facing_bet=True)
            matched, diagnostics = _selected_block(transitions[seat], budgets)
            diagnostics["ordinary_bet_call_share"] = float(ordinary[PAIR_INDEX])

            offset = len(example["M2_features"])
            ordinary_full = np.concatenate([example["M2_features"], ordinary])
            ordinary_ablated = ordinary_full.copy()
            ordinary_ablated[offset + PAIR_INDEX] = 0.0

            matched_full = np.concatenate([example["M2_features"], matched])
            matched_ablated = matched_full.copy()
            matched_ablated[offset + PAIR_INDEX] = 0.0

            example["ordinary_features"] = ordinary_full
            example["ordinary_ablated"] = ordinary_ablated
            example["paired_matched_features"] = matched_full
            example["paired_matched_ablated"] = matched_ablated
            example["paired_opportunity_diagnostics"] = diagnostics


def _evaluate_representation(train: list[dict], test: list[dict], prefix: str) -> dict:
    truth = np.stack([example["target"] for example in test])
    full = _ridge_predict(train, test, f"{prefix}_features")
    ablated = _ridge_predict(train, test, f"{prefix}_ablated")

    by_seat = {}
    for seat in (0, 1):
        indices = [i for i, example in enumerate(test) if int(example["focal_seat"]) == seat]
        seat_truth = truth[indices]
        full_metrics = _metrics(seat_truth, full[indices])
        ablated_metrics = _metrics(seat_truth, ablated[indices])
        by_seat[str(seat)] = {
            "examples": len(indices),
            "full_mae": full_metrics["mae"],
            "ablated_mae": ablated_metrics["mae"],
            "delta_mae": ablated_metrics["mae"] - full_metrics["mae"],
        }

    diff = by_seat["0"]["delta_mae"] - by_seat["1"]["delta_mae"]
    return {
        "by_seat": by_seat,
        "signed_seat_difference": float(diff),
        "absolute_seat_gap": float(abs(diff)),
    }


def _diagnostic_summary(test: list[dict]) -> dict:
    by_seat = {}
    for seat in (0, 1):
        rows = [
            example["paired_opportunity_diagnostics"]
            for example in test
            if int(example["focal_seat"]) == seat
        ]
        by_seat[str(seat)] = {
            "examples": len(rows),
            "mean_eligible_total": float(np.mean([r["eligible_total"] for r in rows])),
            "mean_eligible_round_0": float(np.mean([r["eligible_by_round"]["0"] for r in rows])),
            "mean_eligible_round_1": float(np.mean([r["eligible_by_round"]["1"] for r in rows])),
            "mean_matched_budget_round_0": float(np.mean([r["matched_budgets"]["0"] for r in rows])),
            "mean_matched_budget_round_1": float(np.mean([r["matched_budgets"]["1"] for r in rows])),
            "mean_selected_round_0": float(np.mean([r["selected_by_round"]["0"] for r in rows])),
            "mean_selected_round_1": float(np.mean([r["selected_by_round"]["1"] for r in rows])),
            "mean_ordinary_bet_call_share": float(np.mean([r["ordinary_bet_call_share"] for r in rows])),
            "mean_matched_bet_call_share": float(np.mean([r["matched_bet_call_share"] for r in rows])),
        }

    exact_matches = []
    for seat0, seat1 in _pair_examples(test):
        d0 = seat0["paired_opportunity_diagnostics"]
        d1 = seat1["paired_opportunity_diagnostics"]
        exact_matches.append(
            d0["selected_by_round"] == d1["selected_by_round"]
            and d0["selected_by_round"] == d0["matched_budgets"]
            and d1["selected_by_round"] == d1["matched_budgets"]
        )

    return {
        "by_seat": by_seat,
        "paired_yokes": len(exact_matches),
        "exact_selected_count_match_fraction": float(np.mean(exact_matches)) if exact_matches else None,
    }


def analyze_seed(training_examples: list[dict], evaluation_records: list[dict]) -> dict:
    test = make_examples(evaluation_records)
    attach_paired_features(test)

    ordinary = _evaluate_representation(training_examples, test, "ordinary")
    matched = _evaluate_representation(training_examples, test, "paired_matched")
    ordinary_gap = ordinary["absolute_seat_gap"]
    matched_gap = matched["absolute_seat_gap"]
    attenuation = None if np.isclose(ordinary_gap, 0.0) else 1.0 - matched_gap / ordinary_gap

    return {
        "examples": len(test),
        "ordinary": ordinary,
        "paired_matched": matched,
        "gap_attenuation": None if attenuation is None else float(attenuation),
        "opportunity_diagnostics": _diagnostic_summary(test),
    }


def aggregate_seed_results(seed_reports: list[dict]) -> dict:
    result = {}
    for label in ("ordinary", "paired_matched"):
        seat0 = np.asarray([
            report["analysis"][label]["by_seat"]["0"]["delta_mae"]
            for report in seed_reports
        ])
        seat1 = np.asarray([
            report["analysis"][label]["by_seat"]["1"]["delta_mae"]
            for report in seed_reports
        ])
        gaps = np.abs(seat0 - seat1)
        result[label] = {
            "seat_0_mean_delta_mae": float(seat0.mean()),
            "seat_1_mean_delta_mae": float(seat1.mean()),
            "mean_absolute_seat_gap": float(gaps.mean()),
            "median_absolute_seat_gap": float(np.median(gaps)),
            "seat_0_positive_seed_fraction": float(np.mean(seat0 > 0)),
            "seat_1_positive_seed_fraction": float(np.mean(seat1 > 0)),
        }

    attenuations = [
        report["analysis"]["gap_attenuation"]
        for report in seed_reports
        if report["analysis"]["gap_attenuation"] is not None
    ]
    result["gap_attenuation"] = {
        "mean": float(np.mean(attenuations)) if attenuations else None,
        "median": float(np.median(attenuations)) if attenuations else None,
        "positive_seed_fraction": float(np.mean(np.asarray(attenuations) > 0)) if attenuations else None,
    }
    return result


def run_paired_matched_opportunity_validation(
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
        raise ValueError("training_records contain no focal-policy examples")
    attach_paired_features(train)

    seed_reports = []
    for seed in seeds:
        records, design = generate_seat_yoked_evaluation(
            seed=seed,
            mixtures=mixtures_per_seed,
            hands_per_seat=hands_per_seat,
        )
        seed_reports.append({
            "seed": seed,
            "design": design,
            "analysis": analyze_seed(train, records),
        })

    return {
        "status": "completed",
        "analysis_status": "prospective_paired_matched_opportunity_validation",
        "seeds": list(seeds),
        "mixtures_per_seed": mixtures_per_seed,
        "hands_per_seat": hands_per_seat,
        "matching_rule": "per-yoke per-round minimum eligible transition count",
        "seed_results": seed_reports,
        "aggregate": aggregate_seed_results(seed_reports),
        "provenance": {
            "human_data_accessed": False,
            "previous_frozen_artifacts_modified": False,
            "matching_rule_tuned_after_results": False,
            "threshold_tuned_after_results": False,
        },
        "warning": "Synthetic construct validation only; this does not establish human poker behavior.",
    }


def write_paired_matched_opportunity_validation(
    output_path: str | Path,
    training_path: str | Path = "outputs/mixed-recovery-data.jsonl",
    **kwargs,
) -> dict:
    report = run_paired_matched_opportunity_validation(load_jsonl(training_path), **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prospective paired matched-opportunity validation")
    parser.add_argument("--training", default="outputs/mixed-recovery-data.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--mixtures-per-seed", type=int, default=20)
    parser.add_argument("--hands-per-seat", type=int, default=100)
    parser.add_argument("--output", default="validation/paired-matched-opportunity-validation.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = write_paired_matched_opportunity_validation(
        args.output,
        training_path=args.training,
        seeds=args.seeds,
        mixtures_per_seed=args.mixtures_per_seed,
        hands_per_seat=args.hands_per_seat,
    )
    print(json.dumps({
        "status": report["status"],
        "seeds": report["seeds"],
        "aggregate": report["aggregate"],
        "output": args.output,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
