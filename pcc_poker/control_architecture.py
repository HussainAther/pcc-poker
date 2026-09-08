"""Poker architecture falsification: additive PCC vs context modulation.

This experiment is intentionally architecture-level rather than another latent-weight
recovery benchmark.  Agent identities are generated from fixed synthetic PCC mixtures,
but the regression predictors are measured behavioral signatures from disjoint seeds.
Control uses the already frozen aligned-vs-yoked public-history observable.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

import numpy as np

from .contextual_control_observable import FrozenAlignedYokedHistoryModel
from .policies import MODES, PURE_MIXTURES
from .simulate import generate_mixed_dataset, sample_simplex, simulate_match

CONTEXTS = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "pressure": PURE_MIXTURES["pressure"],
    "control": PURE_MIXTURES["control"],
    "chaos": PURE_MIXTURES["chaos"],
}
TARGETS = ("mean_payoff", "aggression_rate", "fold_rate", "action_entropy")
MIN_RELATIVE_IMPROVEMENT = 0.05
MIN_TARGETS_IMPROVED = 2


def _entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    probs = np.asarray([count / len(actions) for count in counts.values()], dtype=float)
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def _control_observable(records: list[dict], model: FrozenAlignedYokedHistoryModel) -> float:
    values = []
    for row in records:
        pa = max(model.probability(row, condition="aligned"), 1e-12)
        py = max(model.probability(row, condition="yoked"), 1e-12)
        values.append(math.log(pa) - math.log(py))
    return float(np.mean(values)) if values else 0.0


def _summary(records: list[dict], model: FrozenAlignedYokedHistoryModel) -> dict:
    actions = [row["action"] for row in records]
    n = max(len(actions), 1)
    aggression = sum(action in {"bet", "raise"} for action in actions) / n
    fold = sum(action == "fold" for action in actions) / n
    payoff_by_hand = {}
    for row in records:
        payoff_by_hand[row["hand_id"]] = float(row["terminal_payoff"])
    return {
        "pressure_signature": float(aggression),
        "control_signature": _control_observable(records, model),
        "chaos_signature": _entropy(actions),
        "mean_payoff": float(np.mean(list(payoff_by_hand.values()))) if payoff_by_hand else 0.0,
        "aggression_rate": float(aggression),
        "fold_rate": float(fold),
        "action_entropy": _entropy(actions),
        "decisions": len(records),
    }


def _paired_context_records(weights, opponent, *, seed: int, hands_per_seat: int) -> list[dict]:
    focal = []
    for focal_seat in (0, 1):
        mixtures = [opponent, opponent]
        mixtures[focal_seat] = weights
        labels = ["context_opponent", "context_opponent"]
        labels[focal_seat] = "focal_agent"
        records, _ = simulate_match(
            hands_per_seat,
            mixtures[0], mixtures[1],
            seed + focal_seat,
            labels[0], labels[1],
        )
        focal.extend(row for row in records if row["actor"] == focal_seat)
    return focal


def _standardize_design(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train[:, 1:].mean(axis=0)
    scale = train[:, 1:].std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (
        np.column_stack([np.ones(len(train)), (train[:, 1:] - mean) / scale]),
        np.column_stack([np.ones(len(test)), (test[:, 1:] - mean) / scale]),
    )


def _design(row: dict, interaction: str | None) -> np.ndarray:
    p = row["pressure_signature"]
    c = row["control_signature"]
    h = row["chaos_signature"]
    context_names = tuple(CONTEXTS)
    ctx = [1.0 if row["context"] == name else 0.0 for name in context_names[1:]]
    values = [1.0, p, c, h, *ctx]
    if interaction is not None:
        z = {"pressure": p, "control": c, "chaos": h}[interaction]
        values.extend(z * flag for flag in ctx)
    return np.asarray(values, dtype=float)


def _ridge_predict(train_rows: list[dict], test_rows: list[dict], target: str, interaction: str | None) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.stack([_design(row, interaction) for row in train_rows])
    xt = np.stack([_design(row, interaction) for row in test_rows])
    z, zt = _standardize_design(x, xt)
    y = np.asarray([row[target] for row in train_rows], dtype=float)
    yt = np.asarray([row[target] for row in test_rows], dtype=float)
    reg = np.eye(z.shape[1]) * 1.0
    reg[0, 0] = 0.0
    beta = np.linalg.solve(z.T @ z + reg, z.T @ y)
    prediction = zt @ beta
    scale = float(np.std(y))
    if scale < 1e-9:
        scale = 1.0
    return yt, prediction, scale


def _loao(rows: list[dict], agents: int, interaction: str | None) -> dict:
    errors = {target: [] for target in TARGETS}
    for held in range(agents):
        train = [row for row in rows if row["agent_index"] != held]
        test = [row for row in rows if row["agent_index"] == held]
        for target in TARGETS:
            truth, prediction, scale = _ridge_predict(train, test, target, interaction)
            errors[target].extend((np.abs(prediction - truth) / scale).tolist())
    per_target = {target: float(np.mean(values)) for target, values in errors.items()}
    return {"standardized_mae": float(np.mean(list(per_target.values()))), "per_target": per_target}


def run_control_architecture_export(
    *,
    agents: int = 12,
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 30,
    signature_replicates: int = 3,
    outcome_replicates: int = 3,
    hands_per_seat: int = 60,
    seed: int = 9101,
) -> dict:
    if agents < 8:
        raise ValueError("at least eight agents are required")
    calibration, _ = generate_mixed_dataset(
        mixtures=calibration_mixtures,
        hands_per_seat=calibration_hands_per_seat,
        seed=seed + 1,
    )
    public_calibration = [row for row in calibration if row.get("is_focal_policy")]
    control_model = FrozenAlignedYokedHistoryModel.from_records(public_calibration, seed=seed + 2)

    rng = random.Random(seed + 3)
    agent_weights = [sample_simplex(rng, 0.8) for _ in range(agents)]
    signatures = []
    for index, weights in enumerate(agent_weights):
        reps = []
        for rep in range(signature_replicates):
            records = _paired_context_records(
                weights, CONTEXTS["balanced"],
                seed=seed * 1000 + index * 100 + rep * 2,
                hands_per_seat=hands_per_seat,
            )
            reps.append(_summary(records, control_model))
        signatures.append({
            "pressure_signature": float(np.mean([r["pressure_signature"] for r in reps])),
            "control_signature": float(np.mean([r["control_signature"] for r in reps])),
            "chaos_signature": float(np.mean([r["chaos_signature"] for r in reps])),
        })

    rows = []
    for index, weights in enumerate(agent_weights):
        for context_index, (context, opponent) in enumerate(CONTEXTS.items()):
            reps = []
            for rep in range(outcome_replicates):
                records = _paired_context_records(
                    weights, opponent,
                    seed=seed * 2000 + index * 1000 + context_index * 100 + rep * 2,
                    hands_per_seat=hands_per_seat,
                )
                reps.append(_summary(records, control_model))
            row = {
                "agent_index": index,
                "context": context,
                **signatures[index],
            }
            for target in TARGETS:
                row[target] = float(np.mean([r[target] for r in reps]))
            rows.append(row)

    additive = _loao(rows, agents, None)
    interactions = {axis: _loao(rows, agents, axis) for axis in MODES}
    control = interactions["control"]
    relative = (additive["standardized_mae"] - control["standardized_mae"]) / additive["standardized_mae"]
    improved_targets = [
        target for target in TARGETS
        if control["per_target"][target] < additive["per_target"][target]
    ]
    interaction_improvements = {
        axis: float((additive["standardized_mae"] - result["standardized_mae"]) / additive["standardized_mae"])
        for axis, result in interactions.items()
    }
    primary_pass = relative >= MIN_RELATIVE_IMPROVEMENT and len(improved_targets) >= MIN_TARGETS_IMPROVED
    disproportionate = interaction_improvements["control"] > max(interaction_improvements["pressure"], interaction_improvements["chaos"])

    return {
        "schema_version": 1,
        "game": "poker",
        "additive_standardized_mae": additive["standardized_mae"],
        "control_context_standardized_mae": control["standardized_mae"],
        "relative_improvement": float(relative),
        "primary_pass": bool(primary_pass),
        "targets_improved": improved_targets,
        "targets_total": len(TARGETS),
        "interaction_improvements": interaction_improvements,
        "control_interaction_disproportionate": bool(disproportionate),
        "per_target": {"additive": additive["per_target"], "control_interaction": control["per_target"]},
        "design": {
            "agents": agents,
            "contexts": list(CONTEXTS),
            "targets": list(TARGETS),
            "signature_replicates": signature_replicates,
            "outcome_replicates": outcome_replicates,
            "hands_per_seat": hands_per_seat,
            "cross_validation": "leave-one-agent-out",
            "signature_outcome_seeds_disjoint": True,
            "latent_pcc_weights_in_generator": True,
            "latent_pcc_weights_used_as_regression_predictors": False,
            "control_signature": "frozen aligned-vs-context-yoked public-history log-likelihood advantage",
            "pressure_signature": "observed aggressive-action rate",
            "chaos_signature": "observed action entropy",
        },
        "thresholds": {"minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT, "minimum_targets_improved": MIN_TARGETS_IMPROVED},
        "guardrail": "This evaluates architectural organization in engineered synthetic PCC-mixture agents; unlike Blotto v1.1 it is not an emergence-without-latent-PCC experiment.",
        "rows": rows,
    }


def write_control_architecture_export(path: str | Path, **kwargs) -> dict:
    report = run_control_architecture_export(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
