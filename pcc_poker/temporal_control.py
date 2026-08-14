"""Cross-fitted, label-free temporal detection of opponent adaptation."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .policies import MODES
from .simulate import generate_family_dataset


ACTIONS = ("check", "bet", "fold", "call", "raise")
CONTEXTS = tuple((round_index, facing) for round_index in (0, 1) for facing in (0, 1))
FROZEN_POLICY_SHA256 = "ec6020ea7903365c5437ab10bf813cd1a77ab7f62613e118048d613870c0f962"


def _policy_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).with_name("families.py").read_bytes()).hexdigest()


def _rates(counts: Counter, categories: tuple[str, ...], smoothing: float = 1.0) -> list[float]:
    denominator = sum(counts[item] + smoothing for item in categories)
    return [(counts[item] + smoothing) / denominator for item in categories]


def _entropy(probabilities: list[float]) -> float:
    maximum = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    value = -sum(probability * math.log(max(probability, 1e-12)) for probability in probabilities)
    return value / maximum


def _static_features(record: dict) -> np.ndarray:
    private = [float(record["private_rank"] == rank) for rank in (0, 1, 2)]
    public = [
        float(record.get("public_rank") == rank)
        for rank in (None, 0, 1, 2)
    ]
    legal = [float(action in record["legal_actions"]) for action in ACTIONS]
    history_counts = Counter(record["history"])
    hand_history = [history_counts[action] / 4.0 for action in ACTIONS]
    return np.asarray(
        private
        + public
        + legal
        + hand_history
        + [
            float(record["round_index"]),
            float(record["to_call"] > 0),
            float(record["pot"]) / 20.0,
            float(record["to_call"]) / 8.0,
            float(record["focal_seat"]),
        ],
        dtype=float,
    )


def temporal_examples(records: list[dict]) -> list[dict]:
    """Create prior-only decision features for each focal-policy action."""
    by_simulation = defaultdict(list)
    for record in records:
        by_simulation[int(record["simulation_seed"])].append(record)

    examples = []
    for simulation_seed, sequence in by_simulation.items():
        context_counts = {context: Counter() for context in CONTEXTS}
        recent_actions: deque[str] = deque(maxlen=8)
        opponent_decisions = 0
        focal_seat = int(sequence[0]["focal_seat"])
        for record in sequence:
            round_index = int(record["round_index"])
            facing = int(record["to_call"] > 0)
            current_context = (round_index, facing)
            if record["actor"] == focal_seat:
                temporal = []
                for context in CONTEXTS:
                    probabilities = _rates(context_counts[context], ACTIONS)
                    temporal.extend(probabilities)
                    temporal.append(_entropy(probabilities))
                    temporal.append(math.log1p(sum(context_counts[context].values())) / 6.0)
                recent_counts = Counter(recent_actions)
                temporal.extend(_rates(recent_counts, ACTIONS))
                temporal.append(math.log1p(opponent_decisions) / 6.0)

                facing_counts = context_counts[(round_index, 1)]
                facing_total = sum(facing_counts.values())
                predicted_fold = (facing_counts["fold"] + 1) / (facing_total + 3)
                predicted_resistance = (
                    facing_counts["call"] + facing_counts["raise"] + 2
                ) / (facing_total + 3)
                open_counts = context_counts[(round_index, 0)]
                open_total = sum(open_counts.values())
                predicted_open_aggression = (
                    open_counts["bet"] + 1
                ) / (open_total + 2)
                temporal.extend((
                    predicted_fold,
                    predicted_resistance,
                    predicted_open_aggression,
                    predicted_fold * float(facing == 0),
                    predicted_resistance * float(facing == 1),
                ))
                examples.append({
                    "mixture_id": record["mixture_id"],
                    "focal_seat": focal_seat,
                    "simulation_seed": simulation_seed,
                    "static": _static_features(record),
                    "temporal": np.asarray(temporal, dtype=float),
                    "target": float(record["action"] in {"bet", "raise"}),
                    "action": record["action"],
                    "shuffle_context": (
                        round_index,
                        facing,
                        tuple(record["legal_actions"]),
                    ),
                    "weights": {
                        mode: float(record["target_pcc_weights"][mode])
                        for mode in MODES
                    },
                })
            else:
                action = record["action"]
                context_counts[current_context][action] += 1
                recent_actions.append(action)
                opponent_decisions += 1
    return examples


def _standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (matrix - mean) / scale, mean, scale


def _design(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (matrix - mean) / scale
    return np.column_stack((np.ones(len(matrix)), standardized))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _fit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: float = 2.0,
    iterations: int = 60,
) -> dict:
    standardized, mean, scale = _standardize_fit(matrix)
    design = np.column_stack((np.ones(len(matrix)), standardized))
    coefficients = np.zeros(design.shape[1], dtype=float)
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    for _ in range(iterations):
        probability = _sigmoid(design @ coefficients)
        weights = np.clip(probability * (1.0 - probability), 1e-5, None)
        gradient = design.T @ (probability - target) + regularizer @ coefficients
        hessian = design.T @ (design * weights[:, None]) + regularizer
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {"coefficients": coefficients, "mean": mean, "scale": scale}


def _predict(model: dict, matrix: np.ndarray) -> np.ndarray:
    design = _design(matrix, model["mean"], model["scale"])
    return _sigmoid(design @ model["coefficients"])


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    return float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    positives = int(target.sum())
    negatives = len(target) - positives
    if not positives or not negatives:
        return 0.5
    order = np.argsort(probability, kind="mergesort")
    sorted_values = probability[order]
    ranks = np.empty(len(probability), dtype=float)
    start = 0
    while start < len(probability):
        end = start + 1
        while end < len(probability) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[target == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _correlation(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fisher_interval(correlation: float, sample_size: int) -> list[float]:
    if sample_size <= 3:
        return [-1.0, 1.0]
    clipped = min(max(correlation, -0.999999), 0.999999)
    center = np.arctanh(clipped)
    margin = 1.96 / np.sqrt(sample_size - 3)
    return [float(np.tanh(center - margin)), float(np.tanh(center + margin))]


def validate_temporal_control(
    training_records: list[dict],
    evaluation_records: list[dict],
    shuffle_repetitions: int = 25,
    shuffle_seed: int = 63001,
) -> dict:
    """Fit on one set of mixtures and evaluate on wholly separate groups."""
    training = temporal_examples(training_records)
    evaluation = temporal_examples(evaluation_records)
    training_ids = {example["mixture_id"] for example in training}
    evaluation_ids = {example["mixture_id"] for example in evaluation}
    overlap = sorted(training_ids & evaluation_ids)
    if not training or not evaluation or overlap:
        return {
            "status": "invalid_split",
            "mixture_id_overlap": overlap,
            "training_examples": len(training),
            "evaluation_examples": len(evaluation),
        }

    x_static_train = np.stack([example["static"] for example in training])
    x_temporal_train = np.stack([example["temporal"] for example in training])
    x_static_eval = np.stack([example["static"] for example in evaluation])
    x_temporal_eval = np.stack([example["temporal"] for example in evaluation])
    y_train = np.asarray([example["target"] for example in training])
    y_eval = np.asarray([example["target"] for example in evaluation])

    static_model = _fit_logistic(x_static_train, y_train)
    adaptive_model = _fit_logistic(
        np.column_stack((x_static_train, x_temporal_train)), y_train
    )
    static_probability = _predict(static_model, x_static_eval)
    adaptive_probability = _predict(
        adaptive_model, np.column_stack((x_static_eval, x_temporal_eval))
    )
    static_loss = _log_loss(y_eval, static_probability)
    adaptive_loss = _log_loss(y_eval, adaptive_probability)

    rng = np.random.default_rng(shuffle_seed)
    context_indices = defaultdict(list)
    for index, example in enumerate(evaluation):
        context_indices[example["shuffle_context"]].append(index)
    shuffled_losses = []
    for _ in range(shuffle_repetitions):
        shuffled_temporal = x_temporal_eval.copy()
        for indices in context_indices.values():
            source = np.asarray(indices, dtype=int)
            shuffled_temporal[source] = x_temporal_eval[rng.permutation(source)]
        shuffled_probability = _predict(
            adaptive_model,
            np.column_stack((x_static_eval, shuffled_temporal)),
        )
        shuffled_losses.append(_log_loss(y_eval, shuffled_probability))

    clipped_static = np.clip(static_probability, 1e-9, 1 - 1e-9)
    clipped_adaptive = np.clip(adaptive_probability, 1e-9, 1 - 1e-9)
    static_log_probability = np.where(
        y_eval == 1, np.log(clipped_static), np.log(1 - clipped_static)
    )
    adaptive_log_probability = np.where(
        y_eval == 1, np.log(clipped_adaptive), np.log(1 - clipped_adaptive)
    )
    gain = adaptive_log_probability - static_log_probability

    grouped_indices = defaultdict(list)
    for index, example in enumerate(evaluation):
        grouped_indices[(example["mixture_id"], example["focal_seat"])].append(index)
    trajectory_scores = []
    for (mixture_id, focal_seat), indices in sorted(grouped_indices.items()):
        first = evaluation[indices[0]]
        trajectory_scores.append({
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "decisions": len(indices),
            "temporal_log_likelihood_gain": float(np.mean(gain[indices])),
            "weights": first["weights"],
        })

    correlations = {
        mode: _correlation(
            [row["temporal_log_likelihood_gain"] for row in trajectory_scores],
            [row["weights"][mode] for row in trajectory_scores],
        )
        for mode in MODES
    }
    control_interval = _fisher_interval(correlations["control"], len(trajectory_scores))
    shuffled_mean = float(np.mean(shuffled_losses))
    checks = {
        "adaptive_log_loss_below_static": adaptive_loss < static_loss,
        "adaptive_log_loss_below_shuffled_history": adaptive_loss < shuffled_mean,
        "control_correlation_at_least_0_20": correlations["control"] >= 0.20,
        "control_correlation_discriminant": (
            correlations["control"] > correlations["pressure"]
            and correlations["control"] > correlations["chaos"]
        ),
    }
    return {
        "status": "completed",
        "prediction_target": "aggressive action (bet or raise)",
        "training_mixtures": len(training_ids),
        "evaluation_mixtures": len(evaluation_ids),
        "training_examples": len(training),
        "evaluation_examples": len(evaluation),
        "mixture_id_overlap": overlap,
        "predictor_boundary": {
            "static": "own card, public card, legal actions, current hand history, pot, amount to call, round, and seat",
            "temporal_extension": "prior opponent actions only; cumulative context rates, entropy, recency, and smoothed response estimates",
            "excluded": "hidden PCC weights, component scores, action probabilities, outcomes, future cards, and actual opponent private cards",
        },
        "static_model": {
            "log_loss": static_loss,
            "auc": _auc(y_eval, static_probability),
        },
        "temporal_model": {
            "log_loss": adaptive_loss,
            "auc": _auc(y_eval, adaptive_probability),
            "relative_log_loss_improvement": (static_loss - adaptive_loss) / static_loss,
        },
        "shuffled_history_baseline": {
            "repetitions": shuffle_repetitions,
            "mean_log_loss": shuffled_mean,
            "standard_deviation": float(np.std(shuffled_losses)),
        },
        "trajectory_control_score": {
            "definition": "mean held-out log p_temporal(action) minus log p_static(action)",
            "trajectory_examples": len(trajectory_scores),
            "weight_correlations": correlations,
            "control_fisher_95_interval": control_interval,
        },
        "prespecified_checks": checks,
        "temporal_control_confirmed": all(checks.values()),
        "trajectory_scores": trajectory_scores,
        "warning": (
            "Synthetic identifiability does not establish that human players "
            "possess a PCC Control state."
        ),
    }


def run_temporal_control_validation(
    training_mixtures: int = 80,
    evaluation_mixtures: int = 80,
    hands_per_seat: int = 100,
    training_seed: int = 61001,
    evaluation_seed: int = 62001,
    shuffle_repetitions: int = 25,
) -> dict:
    policy_sha256 = _policy_source_sha256()
    if policy_sha256 != FROZEN_POLICY_SHA256:
        raise RuntimeError(
            "Adaptive policy source differs from the frozen v0.3 mechanism"
        )
    training_records, _ = generate_family_dataset(
        "adaptive",
        mixtures=training_mixtures,
        hands_per_seat=hands_per_seat,
        seed=training_seed,
    )
    evaluation_records, _ = generate_family_dataset(
        "adaptive",
        mixtures=evaluation_mixtures,
        hands_per_seat=hands_per_seat,
        seed=evaluation_seed,
    )
    report = validate_temporal_control(
        training_records,
        evaluation_records,
        shuffle_repetitions=shuffle_repetitions,
        shuffle_seed=evaluation_seed + 1000,
    )
    report["design"] = {
        "status": "frozen_temporal_control_validation",
        "policy_version": "0.3.0",
        "frozen_policy_sha256": FROZEN_POLICY_SHA256,
        "observed_policy_sha256": policy_sha256,
        "policies_modified": False,
        "training_mixtures": training_mixtures,
        "evaluation_mixtures": evaluation_mixtures,
        "hands_per_seat": hands_per_seat,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "shuffle_repetitions": shuffle_repetitions,
        "split_unit": "complete mixture_id; both seats remain in one partition",
    }
    return report


def write_temporal_control_validation(path: str | Path, **kwargs) -> dict:
    report = run_temporal_control_validation(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
