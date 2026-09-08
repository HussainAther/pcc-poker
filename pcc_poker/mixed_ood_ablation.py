"""Nested observable-feature ablation for mixed-weight OOD recovery."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .analyze import load_jsonl
from .mixed import ACTIONS, _metrics, _ridge_predict
from .mixed_ood import REGIONS
from .policies import MODES

FORBIDDEN_PREDICTOR_FIELDS = {
    "private_rank",
    "showdown_equity",
    "component_scores",
    "action_probabilities",
    "hidden_pcc_weights",
    "target_pcc_weights",
    "terminal_payoff",
    "simulation_seed",
}

MODEL_ORDER = ("M0", "M1", "M2", "M3")
MODEL_LABELS = {
    "M0": "action_frequencies",
    "M1": "public_betting_context",
    "M2": "public_state_intensity",
    "M3": "sequential_history",
}


def _rates(values: list[str], categories: tuple[str, ...]) -> list[float]:
    counts = Counter(values)
    total = max(len(values), 1)
    return [counts[category] / total for category in categories]


def _nested_features(records: list[dict]) -> dict[str, np.ndarray]:
    """Build nested feature sets using only observable public behavior/state."""
    ordered = sorted(records, key=lambda row: (row["hand_id"], row["decision_index"]))
    actions = [row["action"] for row in ordered]

    m0 = _rates(actions, ACTIONS)

    m1 = list(m0)
    for round_index in (0, 1):
        for facing_bet in (False, True):
            context_actions = [
                row["action"]
                for row in ordered
                if row["round_index"] == round_index
                and (row["to_call"] > 0) == facing_bet
            ]
            m1.extend(_rates(context_actions, ACTIONS))

    by_hand = defaultdict(list)
    for row in ordered:
        by_hand[row["hand_id"]].append(row)

    # Public state/value-intensity summaries. These deliberately exclude private
    # cards, equity, terminal outcomes, hidden PCC internals, and policy scores.
    m2 = list(m1)
    m2.extend([
        float(np.mean([row["pot"] for row in ordered])),
        float(np.mean([row["to_call"] for row in ordered])),
        float(np.mean([len(row["legal_actions"]) for row in ordered])),
        float(np.mean([row["round_index"] for row in ordered])),
        len(ordered) / max(len(by_hand), 1),
        float(ordered[0]["focal_seat"]),
    ])

    transitions = []
    for hand_rows in by_hand.values():
        hand_actions = [row["action"] for row in hand_rows]
        transitions.extend(zip(hand_actions, hand_actions[1:]))

    transition_categories = tuple(
        (left, right) for left in ACTIONS for right in ACTIONS
    )
    transition_counts = Counter(transitions)
    transition_total = max(len(transitions), 1)

    m3 = list(m2)
    m3.extend(
        transition_counts[pair] / transition_total
        for pair in transition_categories
    )

    return {
        "M0": np.asarray(m0, dtype=float),
        "M1": np.asarray(m1, dtype=float),
        "M2": np.asarray(m2, dtype=float),
        "M3": np.asarray(m3, dtype=float),
    }


def nested_mixture_examples(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("is_focal_policy"):
            grouped[(record["mixture_id"], record["focal_seat"])].append(record)

    examples = []
    for (mixture_id, focal_seat), rows in sorted(grouped.items()):
        features = _nested_features(rows)
        target = np.asarray(
            [rows[0]["target_pcc_weights"][mode] for mode in MODES],
            dtype=float,
        )
        example = {
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "target": target,
        }
        for model in MODEL_ORDER:
            example[f"{model}_features"] = features[model]
        examples.append(example)
    return examples


def analyze_mixed_ood_ablation(
    training_records: list[dict],
    ood_records: list[dict],
) -> dict:
    train = nested_mixture_examples(training_records)
    test = nested_mixture_examples(ood_records)

    if not train or not test:
        return {
            "status": "insufficient_examples",
            "train_examples": len(train),
            "test_examples": len(test),
        }

    truth = np.stack([row["target"] for row in test])

    predictions = {}
    overall = {}
    for model in MODEL_ORDER:
        prediction = _ridge_predict(train, test, f"{model}_features")
        predictions[model] = prediction
        overall[model] = {
            "label": MODEL_LABELS[model],
            "feature_count": int(train[0][f"{model}_features"].shape[0]),
            **_metrics(truth, prediction),
        }

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

    by_region = {}
    for region in REGIONS:
        indices = regional_indices.get(region, [])
        if not indices:
            by_region[region] = {"examples": 0}
            continue

        region_truth = truth[indices]
        report = {"examples": len(indices), "models": {}}
        for model in MODEL_ORDER:
            report["models"][model] = _metrics(
                region_truth,
                predictions[model][indices],
            )
        report["best_model_by_mae"] = min(
            MODEL_ORDER,
            key=lambda model: report["models"][model]["mae"],
        )
        by_region[region] = report

    increments = {}
    for previous, current in zip(MODEL_ORDER, MODEL_ORDER[1:]):
        previous_mae = overall[previous]["mae"]
        current_mae = overall[current]["mae"]
        increments[f"{previous}_to_{current}"] = {
            "mae_change": float(current_mae - previous_mae),
            "relative_improvement": float(
                (previous_mae - current_mae) / previous_mae
            ),
            "improves_mae": current_mae < previous_mae,
        }

    best_overall = min(MODEL_ORDER, key=lambda model: overall[model]["mae"])

    return {
        "status": "completed",
        "design": "nested_observable_feature_ablation",
        "prediction_target": "continuous_pressure_control_chaos_weights",
        "training_distribution": "ordinary_dirichlet_mixtures",
        "evaluation_distribution": "prespecified_ood_simplex_regions",
        "observable_features_only": True,
        "forbidden_predictor_fields": sorted(FORBIDDEN_PREDICTOR_FIELDS),
        "models": {
            "M0": "action frequencies only",
            "M1": "M0 + public betting context (round x facing-bet action rates)",
            "M2": "M1 + public state/value-intensity summaries (pot, to-call, legal actions, round, decisions/hand, seat)",
            "M3": "M2 + within-hand action-transition history",
        },
        "train_examples": len(train),
        "ood_test_examples": len(test),
        "overall": overall,
        "incremental_effects": increments,
        "best_overall_model_by_mae": best_overall,
        "by_region": by_region,
        "prespecified_checks": {
            "at_least_one_context_layer_improves_over_M0": any(
                overall[model]["mae"] < overall["M0"]["mae"]
                for model in MODEL_ORDER[1:]
            ),
            "full_M3_beats_M0_overall": (
                overall["M3"]["mae"] < overall["M0"]["mae"]
            ),
            "balanced_region_retained_without_tuning": (
                "balanced" in by_region and by_region["balanced"].get("examples", 0) > 0
            ),
        },
        "warning": (
            "This is a synthetic identifiability decomposition. It does not "
            "establish human PCC constructs, and it intentionally excludes "
            "private cards, equity, terminal outcomes, and hidden policy internals."
        ),
    }


def analyze_mixed_ood_ablation_files(
    training_path: str | Path,
    ood_path: str | Path,
    output_path: str | Path,
) -> dict:
    report = analyze_mixed_ood_ablation(
        load_jsonl(training_path),
        load_jsonl(ood_path),
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
