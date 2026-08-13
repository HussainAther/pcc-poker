"""Cross-family tests of PCC mixture recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mixed import _metrics, _project_simplex, _ridge_predict, mixture_examples
from .analyze import load_jsonl
from .simulate import generate_family_dataset


def analyze_family_transfer(
    training_records: list[dict],
    transfer_records: list[dict],
    shuffle_repetitions: int = 25,
    seed: int = 303,
) -> dict:
    """Fit on one mechanism family and evaluate only on another family."""
    train = mixture_examples(training_records)
    test = mixture_examples(transfer_records)
    train_ids = sorted({row["mixture_id"] for row in train})
    test_ids = sorted({row["mixture_id"] for row in test})
    overlap = sorted(set(train_ids) & set(test_ids))
    if not train or not test or overlap:
        return {
            "status": "invalid_family_split",
            "train_mixtures": len(train_ids),
            "test_mixtures": len(test_ids),
            "overlapping_mixture_ids": overlap,
        }

    truth = np.stack([row["target"] for row in test])
    action_prediction = _ridge_predict(train, test, "action_features")
    contextual_prediction = _ridge_predict(train, test, "contextual_features")
    mean_weights = np.stack([row["target"] for row in train]).mean(axis=0)
    constant_prediction = _project_simplex(
        np.repeat(mean_weights[None, :], len(test), axis=0)
    )

    action_report = _metrics(truth, action_prediction)
    contextual_report = _metrics(truth, contextual_prediction)
    constant_report = _metrics(truth, constant_prediction)

    rng = np.random.default_rng(seed)
    train_targets = np.stack([row["target"] for row in train])
    shuffled = []
    for _ in range(shuffle_repetitions):
        permuted = train_targets[rng.permutation(len(train_targets))]
        prediction = _ridge_predict(
            train, test, "contextual_features", targets=permuted
        )
        shuffled.append(_metrics(truth, prediction)["mae"])

    training_families = sorted({
        row.get("policy_family", "unspecified")
        for row in training_records if row.get("is_focal_policy")
    })
    transfer_families = sorted({
        row.get("policy_family", "unspecified")
        for row in transfer_records if row.get("is_focal_policy")
    })
    shuffled_mean = float(np.mean(shuffled))
    return {
        "status": "completed",
        "training_policy_families": training_families,
        "transfer_policy_families": transfer_families,
        "training_mixtures": len(train_ids),
        "transfer_mixtures": len(test_ids),
        "training_examples": len(train),
        "transfer_examples": len(test),
        "mixture_id_overlap": False,
        "observable_features_only": True,
        "constant_mean_baseline": constant_report,
        "action_frequency_model": action_report,
        "contextual_history_model": contextual_report,
        "shuffled_target_baseline": {
            "repetitions": shuffle_repetitions,
            "mae_mean": shuffled_mean,
            "mae_std": float(np.std(shuffled)),
        },
        "prespecified_checks": {
            "contextual_below_constant_mean": (
                contextual_report["mae"] < constant_report["mae"]
            ),
            "contextual_below_action_frequency": (
                contextual_report["mae"] < action_report["mae"]
            ),
            "contextual_below_shuffled_mean": (
                contextual_report["mae"] < shuffled_mean
            ),
        },
        "warning": (
            "The target weights were assigned by construction across both policy "
            "families. Transfer is an anti-circularity test of behavioral signatures, "
            "not evidence of human PCC."
        ),
    }


def analyze_family_transfer_files(
    training_path: str | Path,
    transfer_path: str | Path,
    output_path: str | Path,
) -> dict:
    report = analyze_family_transfer(
        load_jsonl(training_path), load_jsonl(transfer_path)
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_family_transfer_grid(
    seed_pairs: tuple[tuple[int, int], ...] = (
        (61, 71), (62, 72), (63, 73), (64, 74), (65, 75)
    ),
    mixtures: int = 40,
    hands_per_seat: int = 75,
    alpha: float = 0.7,
    shuffle_repetitions: int = 10,
) -> dict:
    """Replicate transfer in both directions across independently generated sets."""
    runs = []
    for score_seed, independent_seed in seed_pairs:
        score_records, _ = generate_family_dataset(
            "score", mixtures, hands_per_seat, score_seed, alpha
        )
        independent_records, _ = generate_family_dataset(
            "independent", mixtures, hands_per_seat, independent_seed, alpha
        )
        directions = (
            ("score_to_independent", score_records, independent_records),
            ("independent_to_score", independent_records, score_records),
        )
        for direction, training, transfer in directions:
            report = analyze_family_transfer(
                training,
                transfer,
                shuffle_repetitions=shuffle_repetitions,
                seed=score_seed * 1000 + independent_seed,
            )
            runs.append({
                "direction": direction,
                "score_seed": score_seed,
                "independent_seed": independent_seed,
                "contextual_mae": report["contextual_history_model"]["mae"],
                "action_frequency_mae": report["action_frequency_model"]["mae"],
                "constant_mean_mae": report["constant_mean_baseline"]["mae"],
                "shuffled_mae_mean": report["shuffled_target_baseline"]["mae_mean"],
                "checks": report["prespecified_checks"],
            })

    summaries = {}
    for direction in ("score_to_independent", "independent_to_score"):
        subset = [run for run in runs if run["direction"] == direction]
        contextual = np.asarray([run["contextual_mae"] for run in subset])
        action = np.asarray([run["action_frequency_mae"] for run in subset])
        constant = np.asarray([run["constant_mean_mae"] for run in subset])
        shuffled = np.asarray([run["shuffled_mae_mean"] for run in subset])
        summaries[direction] = {
            "runs": len(subset),
            "contextual_mae_mean": float(contextual.mean()),
            "action_frequency_mae_mean": float(action.mean()),
            "constant_mean_mae_mean": float(constant.mean()),
            "shuffled_mae_mean": float(shuffled.mean()),
            "contextual_below_action_rate": float(np.mean(contextual < action)),
            "contextual_below_constant_rate": float(np.mean(contextual < constant)),
            "contextual_below_shuffled_rate": float(np.mean(contextual < shuffled)),
        }
    return {
        "status": "completed",
        "seed_pairs": [list(pair) for pair in seed_pairs],
        "mixtures_per_family": mixtures,
        "hands_per_seat": hands_per_seat,
        "runs": runs,
        "aggregate_by_direction": summaries,
        "warning": (
            "Both families still use researcher-assigned PCC mixture coordinates. "
            "Failure indicates mechanism-specific behavioral mappings; success "
            "would not independently establish naturally occurring PCC."
        ),
    }


def write_family_transfer_grid(output_path: str | Path, **kwargs) -> dict:
    report = run_family_transfer_grid(**kwargs)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
