"""Prospective decomposition of seat asymmetry into opportunity vs response effects."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .analyze import load_jsonl
from .context_matched_transition_validation import make_examples, mixed_pair_block
from .history_structure_probe import MIXED_PAIR_CATEGORIES
from .mixed import _metrics, _ridge_predict
from .seat_yoked_transition_validation import generate_seat_yoked_evaluation

DEFAULT_SEEDS = (1301, 1302, 1303, 1304, 1305)
ROUND_BUDGETS = {0: 14, 1: 41}
TARGET_PAIR = ("bet", "call")
PAIR_INDEX = MIXED_PAIR_CATEGORIES.index(TARGET_PAIR)


def _eligible_transitions(rows: list[dict]) -> list[dict]:
    """Return facing-bet mixed transitions with stable public identifiers."""
    by_hand: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        by_hand[row["hand_id"]].append(row)

    transitions = []
    for hand_id, hand_rows in by_hand.items():
        ordered = sorted(hand_rows, key=lambda row: row["decision_index"])
        for left, right in zip(ordered, ordered[1:]):
            if right["to_call"] <= 0:
                continue
            pair = tuple(sorted((left["action"], right["action"])))
            if pair not in MIXED_PAIR_CATEGORIES:
                continue
            transitions.append(
                {
                    "hand_id": hand_id,
                    "left_decision_index": left["decision_index"],
                    "right_decision_index": right["decision_index"],
                    "round_index": int(right["round_index"]),
                    "pair": pair,
                }
            )
    return transitions


def _stable_transition_key(transition: dict) -> str:
    payload = "|".join(
        str(transition[key])
        for key in (
            "hand_id",
            "left_decision_index",
            "right_decision_index",
            "round_index",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def opportunity_standardized_block(
    rows: list[dict],
    round_budgets: dict[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a fixed-budget facing-bet mixed-pair block.

    Transition selection depends only on stable trajectory identifiers, never on
    the action-pair label. Each round is sampled up to its frozen budget.
    """
    budgets = dict(ROUND_BUDGETS if round_budgets is None else round_budgets)
    transitions = _eligible_transitions(rows)
    selected = []
    counts_by_round = {}
    shortfall_by_round = {}

    for round_index in (0, 1):
        candidates = [
            item for item in transitions
            if item["round_index"] == round_index
        ]
        candidates.sort(key=_stable_transition_key)
        budget = int(budgets[round_index])
        counts_by_round[str(round_index)] = len(candidates)
        shortfall_by_round[str(round_index)] = len(candidates) < budget
        selected.extend(candidates[:budget])

    counts = Counter(item["pair"] for item in selected)
    total = len(selected)
    if total == 0:
        block = np.zeros(len(MIXED_PAIR_CATEGORIES), dtype=float)
    else:
        block = np.asarray(
            [counts[pair] / total for pair in MIXED_PAIR_CATEGORIES],
            dtype=float,
        )

    ordinary = mixed_pair_block(rows, facing_bet=True)
    diagnostics = {
        "eligible_total": len(transitions),
        "eligible_by_round": counts_by_round,
        "selected_total": total,
        "selected_by_round": {
            str(round_index): min(counts_by_round[str(round_index)], int(budgets[round_index]))
            for round_index in (0, 1)
        },
        "budget_shortfall_by_round": shortfall_by_round,
        "any_budget_shortfall": any(shortfall_by_round.values()),
        "ordinary_bet_call_share": float(ordinary[PAIR_INDEX]),
        "standardized_bet_call_share": float(block[PAIR_INDEX]),
    }
    return block, diagnostics


def attach_decomposition_features(examples: list[dict]) -> None:
    for example in examples:
        ordinary = mixed_pair_block(example["rows"], facing_bet=True)
        standardized, diagnostics = opportunity_standardized_block(example["rows"])

        offset = len(example["M2_features"])
        ordinary_full = np.concatenate([example["M2_features"], ordinary])
        ordinary_ablated = ordinary_full.copy()
        ordinary_ablated[offset + PAIR_INDEX] = 0.0

        standardized_full = np.concatenate([example["M2_features"], standardized])
        standardized_ablated = standardized_full.copy()
        standardized_ablated[offset + PAIR_INDEX] = 0.0

        example["ordinary_features"] = ordinary_full
        example["ordinary_ablated"] = ordinary_ablated
        example["standardized_features"] = standardized_full
        example["standardized_ablated"] = standardized_ablated
        example["opportunity_diagnostics"] = diagnostics


def _evaluate_representation(train: list[dict], test: list[dict], prefix: str) -> dict:
    truth = np.stack([example["target"] for example in test])
    full = _ridge_predict(train, test, f"{prefix}_features")
    ablated = _ridge_predict(train, test, f"{prefix}_ablated")

    by_seat = {}
    for seat in (0, 1):
        indices = [i for i, example in enumerate(test) if example["focal_seat"] == seat]
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
    report = {}
    for seat in (0, 1):
        rows = [e["opportunity_diagnostics"] for e in test if e["focal_seat"] == seat]
        report[str(seat)] = {
            "examples": len(rows),
            "mean_eligible_total": float(np.mean([r["eligible_total"] for r in rows])),
            "mean_eligible_round_0": float(np.mean([r["eligible_by_round"]["0"] for r in rows])),
            "mean_eligible_round_1": float(np.mean([r["eligible_by_round"]["1"] for r in rows])),
            "budget_shortfall_fraction": float(np.mean([r["any_budget_shortfall"] for r in rows])),
            "mean_ordinary_bet_call_share": float(np.mean([r["ordinary_bet_call_share"] for r in rows])),
            "mean_standardized_bet_call_share": float(np.mean([r["standardized_bet_call_share"] for r in rows])),
        }
    return report


def analyze_seed(training_examples: list[dict], evaluation_records: list[dict]) -> dict:
    test = make_examples(evaluation_records)
    attach_decomposition_features(test)
    ordinary = _evaluate_representation(training_examples, test, "ordinary")
    standardized = _evaluate_representation(training_examples, test, "standardized")
    ordinary_gap = ordinary["absolute_seat_gap"]
    standardized_gap = standardized["absolute_seat_gap"]
    attenuation = None if np.isclose(ordinary_gap, 0.0) else 1.0 - standardized_gap / ordinary_gap
    return {
        "examples": len(test),
        "ordinary": ordinary,
        "opportunity_standardized": standardized,
        "gap_attenuation": None if attenuation is None else float(attenuation),
        "opportunity_diagnostics": _diagnostic_summary(test),
    }


def aggregate_seed_results(seed_reports: list[dict]) -> dict:
    attenuations = [
        report["analysis"]["gap_attenuation"]
        for report in seed_reports
        if report["analysis"]["gap_attenuation"] is not None
    ]
    result = {}
    for label in ("ordinary", "opportunity_standardized"):
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
            "seat_0_positive_seed_fraction": float(np.mean(seat0 > 0)),
            "seat_1_positive_seed_fraction": float(np.mean(seat1 > 0)),
        }
    result["gap_attenuation"] = {
        "mean": float(np.mean(attenuations)) if attenuations else None,
        "median": float(np.median(attenuations)) if attenuations else None,
        "positive_seed_fraction": float(np.mean(np.asarray(attenuations) > 0)) if attenuations else None,
    }
    return result


def run_seat_opportunity_decomposition(
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
    attach_decomposition_features(train)

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
        "analysis_status": "prospective_seat_opportunity_decomposition",
        "seeds": list(seeds),
        "mixtures_per_seed": mixtures_per_seed,
        "hands_per_seat": hands_per_seat,
        "round_budgets": {str(k): v for k, v in ROUND_BUDGETS.items()},
        "seed_results": seed_reports,
        "aggregate": aggregate_seed_results(seed_reports),
        "provenance": {
            "human_data_accessed": False,
            "previous_frozen_artifacts_modified": False,
            "budget_tuned_after_results": False,
            "threshold_tuned_after_results": False,
        },
        "warning": "Synthetic construct validation only; this does not establish human poker behavior.",
    }


def write_seat_opportunity_decomposition(
    output_path: str | Path,
    training_path: str | Path = "outputs/mixed-recovery-data.jsonl",
    **kwargs,
) -> dict:
    report = run_seat_opportunity_decomposition(load_jsonl(training_path), **kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prospective seat-opportunity decomposition")
    parser.add_argument("--training", default="outputs/mixed-recovery-data.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--mixtures-per-seed", type=int, default=20)
    parser.add_argument("--hands-per-seat", type=int, default=100)
    parser.add_argument("--output", default="validation/seat-opportunity-decomposition.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = write_seat_opportunity_decomposition(
        args.output,
        training_path=args.training,
        seeds=args.seeds,
        mixtures_per_seed=args.mixtures_per_seed,
        hands_per_seat=args.hands_per_seat,
    )
    print(json.dumps({"status": report["status"], "seeds": report["seeds"], "aggregate": report["aggregate"], "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
