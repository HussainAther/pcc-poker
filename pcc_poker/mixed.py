"""Leakage-resistant recovery of continuous PCC mixture weights."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .policies import MODES
from .analyze import load_jsonl
from .simulate import generate_mixed_dataset

ACTIONS = ("check", "bet", "fold", "call", "raise")


def _group_is_test(mixture_id: str) -> bool:
    """Assign a complete mixture group to the deterministic 20% test split."""
    digest = hashlib.sha256(mixture_id.encode()).hexdigest()
    return int(digest[:8], 16) % 5 == 0


def _rates(values: list[str], categories: tuple[str, ...]) -> list[float]:
    counts = Counter(values)
    total = max(len(values), 1)
    return [counts[category] / total for category in categories]


def _features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return action-frequency and richer observable-history feature vectors."""
    ordered = sorted(records, key=lambda row: (row["hand_id"], row["decision_index"]))
    actions = [row["action"] for row in ordered]
    action_features = _rates(actions, ACTIONS)

    contextual = list(action_features)
    for round_index in (0, 1):
        for facing_bet in (False, True):
            context_actions = [
                row["action"] for row in ordered
                if row["round_index"] == round_index
                and (row["to_call"] > 0) == facing_bet
            ]
            contextual.extend(_rates(context_actions, ACTIONS))

    transitions = []
    by_hand = defaultdict(list)
    for row in ordered:
        by_hand[row["hand_id"]].append(row)
    for hand_rows in by_hand.values():
        hand_actions = [row["action"] for row in hand_rows]
        transitions.extend(zip(hand_actions, hand_actions[1:]))
    transition_categories = tuple((left, right) for left in ACTIONS for right in ACTIONS)
    transition_counts = Counter(transitions)
    transition_total = max(len(transitions), 1)
    contextual.extend(
        transition_counts[pair] / transition_total for pair in transition_categories
    )

    contextual.extend([
        float(np.mean([row["pot"] for row in ordered])),
        float(np.mean([row["to_call"] for row in ordered])),
        float(np.mean([len(row["legal_actions"]) for row in ordered])),
        float(np.mean([row["round_index"] for row in ordered])),
        len(ordered) / max(len(by_hand), 1),
        float(ordered[0]["focal_seat"]),
    ])
    return np.asarray(action_features, dtype=float), np.asarray(contextual, dtype=float)


def mixture_examples(records: list[dict]) -> list[dict]:
    """Aggregate focal decisions by mixture and seat without using hidden features."""
    grouped = defaultdict(list)
    for record in records:
        if record.get("is_focal_policy"):
            grouped[(record["mixture_id"], record["focal_seat"])].append(record)

    examples = []
    for (mixture_id, focal_seat), rows in sorted(grouped.items()):
        action_features, contextual_features = _features(rows)
        target = np.asarray(
            [rows[0]["target_pcc_weights"][mode] for mode in MODES], dtype=float
        )
        examples.append({
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "action_features": action_features,
            "contextual_features": contextual_features,
            "target": target,
        })
    return examples


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Project approximate weights to nonnegative rows summing to one."""
    clipped = np.clip(values, 0.0, None)
    totals = clipped.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 1e-12
    clipped[empty] = 1 / len(MODES)
    totals[empty] = 1.0
    return clipped / totals


def _ridge_predict(
    train: list[dict],
    test: list[dict],
    feature_key: str,
    targets: np.ndarray | None = None,
    penalty: float = 3.0,
) -> np.ndarray:
    x_train = np.stack([row[feature_key] for row in train])
    x_test = np.stack([row[feature_key] for row in test])
    y_train = targets if targets is not None else np.stack([row["target"] for row in train])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z_train = np.column_stack([np.ones(len(train)), (x_train - mean) / scale])
    z_test = np.column_stack([np.ones(len(test)), (x_test - mean) / scale])
    regularizer = np.eye(z_train.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        z_train.T @ z_train + regularizer,
        z_train.T @ y_train,
    )
    return _project_simplex(z_test @ coefficients)


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    absolute_error = np.abs(truth - prediction)
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "per_mode_mae": {
            mode: float(absolute_error[:, index].mean())
            for index, mode in enumerate(MODES)
        },
        "dominant_mode_accuracy": float(
            np.mean(np.argmax(truth, axis=1) == np.argmax(prediction, axis=1))
        ),
    }


def analyze_mixed_recovery(
    records: list[dict],
    shuffle_repetitions: int = 25,
    seed: int = 101,
) -> dict:
    """Recover unseen continuous mixtures and compare prespecified baselines."""
    examples = mixture_examples(records)
    train = [row for row in examples if not _group_is_test(row["mixture_id"])]
    test = [row for row in examples if _group_is_test(row["mixture_id"])]
    train_groups = sorted({row["mixture_id"] for row in train})
    test_groups = sorted({row["mixture_id"] for row in test})
    if not train or not test or set(train_groups) & set(test_groups):
        return {
            "status": "insufficient_groups",
            "train_mixtures": len(train_groups),
            "test_mixtures": len(test_groups),
        }

    truth = np.stack([row["target"] for row in test])
    action_prediction = _ridge_predict(train, test, "action_features")
    contextual_prediction = _ridge_predict(train, test, "contextual_features")
    action_report = _metrics(truth, action_prediction)
    contextual_report = _metrics(truth, contextual_prediction)

    rng = np.random.default_rng(seed)
    train_targets = np.stack([row["target"] for row in train])
    shuffled_metrics = []
    for _ in range(shuffle_repetitions):
        shuffled = train_targets[rng.permutation(len(train_targets))]
        prediction = _ridge_predict(
            train, test, "contextual_features", targets=shuffled
        )
        shuffled_metrics.append(_metrics(truth, prediction))

    shuffled_mae = np.asarray([report["mae"] for report in shuffled_metrics])
    shuffled_rmse = np.asarray([report["rmse"] for report in shuffled_metrics])
    shuffled_accuracy = np.asarray([
        report["dominant_mode_accuracy"] for report in shuffled_metrics
    ])
    shuffled_report = {
        "repetitions": shuffle_repetitions,
        "mae_mean": float(shuffled_mae.mean()),
        "mae_std": float(shuffled_mae.std()),
        "rmse_mean": float(shuffled_rmse.mean()),
        "dominant_mode_accuracy_mean": float(shuffled_accuracy.mean()),
    }

    return {
        "status": "completed",
        "prediction_target": "continuous_pressure_control_chaos_weights",
        "split_unit": "mixture_id; both seats and simulation seeds remain grouped",
        "observable_features_only": True,
        "train_mixtures": len(train_groups),
        "test_mixtures": len(test_groups),
        "train_mixture_ids": train_groups,
        "test_mixture_ids": test_groups,
        "train_examples": len(train),
        "test_examples": len(test),
        "action_frequency_baseline": action_report,
        "contextual_history_model": contextual_report,
        "shuffled_target_baseline": shuffled_report,
        "relative_mae_improvement_over_action_frequency": float(
            (action_report["mae"] - contextual_report["mae"])
            / action_report["mae"]
        ),
        "prespecified_checks": {
            "contextual_mae_below_action_frequency": (
                contextual_report["mae"] < action_report["mae"]
            ),
            "contextual_mae_below_shuffled_mean": (
                contextual_report["mae"] < shuffled_report["mae_mean"]
            ),
        },
        "warning": (
            "Recovery of synthetic objective weights establishes identifiability "
            "only; it is not evidence that human behavior follows PCC."
        ),
    }


def analyze_mixed_file(input_path: str | Path, output_path: str | Path) -> dict:
    report = analyze_mixed_recovery(load_jsonl(input_path))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_mixed_grid(
    seeds: tuple[int, ...] = (41, 42, 43, 44, 45),
    temperatures: tuple[float, ...] = (0.25, 0.35, 0.5),
    mixtures: int = 60,
    hands_per_seat: int = 100,
    alpha: float = 0.7,
    shuffle_repetitions: int = 25,
) -> dict:
    """Replicate continuous recovery across independent seeds and temperatures."""
    runs = []
    for temperature in temperatures:
        for seed in seeds:
            records, _ = generate_mixed_dataset(
                mixtures=mixtures,
                hands_per_seat=hands_per_seat,
                seed=seed,
                alpha=alpha,
                focal_temperature=temperature,
            )
            report = analyze_mixed_recovery(
                records,
                shuffle_repetitions=shuffle_repetitions,
                seed=seed + 100_000,
            )
            runs.append({
                "seed": seed,
                "focal_temperature": temperature,
                "contextual_mae": report["contextual_history_model"]["mae"],
                "action_frequency_mae": report["action_frequency_baseline"]["mae"],
                "shuffled_mae_mean": report["shuffled_target_baseline"]["mae_mean"],
                "contextual_dominant_accuracy": report["contextual_history_model"]["dominant_mode_accuracy"],
                "action_frequency_dominant_accuracy": report["action_frequency_baseline"]["dominant_mode_accuracy"],
                "checks": report["prespecified_checks"],
            })

    contextual_mae = np.asarray([run["contextual_mae"] for run in runs])
    action_mae = np.asarray([run["action_frequency_mae"] for run in runs])
    shuffled_mae = np.asarray([run["shuffled_mae_mean"] for run in runs])
    return {
        "status": "completed",
        "seeds": list(seeds),
        "focal_temperatures": list(temperatures),
        "reference_temperature": 0.35,
        "mixtures_per_run": mixtures,
        "hands_per_seat": hands_per_seat,
        "dirichlet_alpha": alpha,
        "runs": runs,
        "aggregate": {
            "runs": len(runs),
            "contextual_mae_mean": float(contextual_mae.mean()),
            "contextual_mae_std": float(contextual_mae.std()),
            "action_frequency_mae_mean": float(action_mae.mean()),
            "shuffled_mae_mean": float(shuffled_mae.mean()),
            "contextual_below_action_frequency_rate": float(
                np.mean(contextual_mae < action_mae)
            ),
            "contextual_below_shuffled_rate": float(
                np.mean(contextual_mae < shuffled_mae)
            ),
        },
        "warning": (
            "This grid tests robustness within one hand-authored policy family, "
            "not generalization to human behavior or other policy mechanisms."
        ),
    }


def write_mixed_grid(output_path: str | Path, **kwargs) -> dict:
    report = run_mixed_grid(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
