"""Yoked within-hand sequence-destruction control for mixed OOD recovery."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .analyze import load_jsonl
from .mixed import _metrics, _ridge_predict
from .mixed_ood import REGIONS
from .mixed_ood_ablation import _nested_features
from .policies import MODES

DEFAULT_PERMUTATION_SEEDS = tuple(range(88001, 88026))


def _group_focal_records(records: list[dict]) -> list[tuple[tuple[str, int], list[dict]]]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("is_focal_policy"):
            grouped[(record["mixture_id"], record["focal_seat"])].append(record)
    return sorted(grouped.items())


def _prepare_examples(records: list[dict]) -> list[dict]:
    prepared = []
    for (mixture_id, focal_seat), rows in _group_focal_records(records):
        ordered = sorted(rows, key=lambda row: (row["hand_id"], row["decision_index"]))
        original = _nested_features(ordered)
        by_hand = defaultdict(list)
        for row in ordered:
            by_hand[row["hand_id"]].append(row["action"])
        target = np.asarray(
            [ordered[0]["target_pcc_weights"][mode] for mode in MODES],
            dtype=float,
        )
        prepared.append({
            "mixture_id": mixture_id,
            "focal_seat": focal_seat,
            "target": target,
            "M2_features": original["M2"],
            "M3_features": original["M3"],
            "hand_actions": tuple(tuple(actions) for actions in by_hand.values()),
        })
    return prepared


def _shuffled_m3_from_prepared(example: dict, rng: np.random.Generator) -> np.ndarray:
    from .mixed import ACTIONS

    transitions = []
    for actions in example["hand_actions"]:
        permuted = [str(action) for action in rng.permutation(actions)]
        assert Counter(actions) == Counter(permuted)
        transitions.extend(zip(permuted, permuted[1:]))

    transition_categories = tuple((left, right) for left in ACTIONS for right in ACTIONS)
    counts = Counter(transitions)
    total = max(len(transitions), 1)
    transition_block = np.asarray([counts[pair] / total for pair in transition_categories], dtype=float)
    return np.concatenate([example["M2_features"], transition_block])


def _shuffled_m3_features(rows: list[dict], rng: np.random.Generator) -> np.ndarray:
    prepared = _prepare_examples([dict(row, is_focal_policy=True) for row in rows])
    if len(prepared) != 1:
        raise ValueError("rows must describe exactly one mixture/seat example")
    return _shuffled_m3_from_prepared(prepared[0], rng)


def _shuffled_examples_from_prepared(prepared: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    examples = []
    for example in prepared:
        shuffled_m3 = _shuffled_m3_from_prepared(example, rng)
        np.testing.assert_allclose(
            example["M2_features"],
            shuffled_m3[: len(example["M2_features"])],
            rtol=0.0, atol=0.0,
        )
        examples.append({
            "mixture_id": example["mixture_id"],
            "focal_seat": example["focal_seat"],
            "target": example["target"],
            "M2_features": example["M2_features"],
            "M3_features": shuffled_m3,
        })
    return examples

def _region_indices(test_examples: list[dict], ood_records: list[dict]) -> dict[str, list[int]]:
    region_by_mixture = {}
    for record in ood_records:
        mixture_id = record.get("mixture_id")
        region = record.get("ood_region")
        if mixture_id is not None and region is not None:
            region_by_mixture[mixture_id] = region

    result = defaultdict(list)
    for index, example in enumerate(test_examples):
        region = region_by_mixture.get(example["mixture_id"])
        if region is not None:
            result[region].append(index)
    return result


def _summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _attenuation(m2_mae: float, original_m3_mae: float, shuffled_m3_mean_mae: float) -> dict:
    original_gain = m2_mae - original_m3_mae
    shuffled_gain = m2_mae - shuffled_m3_mean_mae
    attenuation = None
    if original_gain > 0:
        attenuation = (original_gain - shuffled_gain) / original_gain
    return {
        "original_M2_to_M3_gain": float(original_gain),
        "mean_shuffled_M2_to_M3_gain": float(shuffled_gain),
        "gain_attenuation_fraction": None if attenuation is None else float(attenuation),
    }


def analyze_mixed_ood_history_destruction(
    training_records: list[dict],
    ood_records: list[dict],
    permutation_seeds: tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS,
) -> dict:
    original_train = _prepare_examples(training_records)
    original_test = _prepare_examples(ood_records)
    if not original_train or not original_test:
        return {
            "status": "insufficient_examples",
            "train_examples": len(original_train),
            "test_examples": len(original_test),
        }

    truth = np.stack([row["target"] for row in original_test])
    m2_prediction = _ridge_predict(original_train, original_test, "M2_features")
    m3_prediction = _ridge_predict(original_train, original_test, "M3_features")
    m2_overall = _metrics(truth, m2_prediction)
    m3_overall = _metrics(truth, m3_prediction)

    regional_indices = _region_indices(original_test, ood_records)
    baseline_by_region = {}
    for region in REGIONS:
        indices = regional_indices.get(region, [])
        if not indices:
            baseline_by_region[region] = {"examples": 0}
            continue
        baseline_by_region[region] = {
            "examples": len(indices),
            "M2": _metrics(truth[indices], m2_prediction[indices]),
            "M3_original": _metrics(truth[indices], m3_prediction[indices]),
        }

    overall_shuffled_mae = []
    region_shuffled_mae = {region: [] for region in REGIONS}
    replicate_reports = []

    for seed in permutation_seeds:
        shuffled_train = _shuffled_examples_from_prepared(original_train, seed)
        shuffled_test = _shuffled_examples_from_prepared(original_test, seed + 1_000_000)
        prediction = _ridge_predict(shuffled_train, shuffled_test, "M3_features")
        metrics = _metrics(truth, prediction)
        overall_shuffled_mae.append(metrics["mae"])

        replicate = {"seed": int(seed), "overall_mae": float(metrics["mae"]), "by_region": {}}
        for region in REGIONS:
            indices = regional_indices.get(region, [])
            if not indices:
                continue
            region_mae = _metrics(truth[indices], prediction[indices])["mae"]
            region_shuffled_mae[region].append(region_mae)
            replicate["by_region"][region] = float(region_mae)
        replicate_reports.append(replicate)

    overall_summary = _summarize(overall_shuffled_mae)
    overall_effect = _attenuation(m2_overall["mae"], m3_overall["mae"], overall_summary["mean"])

    by_region = {}
    for region in REGIONS:
        baseline = baseline_by_region[region]
        if not baseline.get("examples"):
            by_region[region] = baseline
            continue
        shuffled_summary = _summarize(region_shuffled_mae[region])
        effect = _attenuation(
            baseline["M2"]["mae"],
            baseline["M3_original"]["mae"],
            shuffled_summary["mean"],
        )
        by_region[region] = {
            **baseline,
            "M3_shuffled_mae": shuffled_summary,
            **effect,
        }

    control_attenuation = by_region.get("control_heavy", {}).get("gain_attenuation_fraction")
    overall_attenuation = overall_effect["gain_attenuation_fraction"]

    return {
        "status": "completed",
        "design": "yoked_within_hand_history_destruction",
        "prediction_target": "continuous_pressure_control_chaos_weights",
        "training_distribution": "ordinary_dirichlet_mixtures",
        "evaluation_distribution": "prespecified_ood_simplex_regions",
        "permutation_seeds": [int(seed) for seed in permutation_seeds],
        "permutation_replicates": len(permutation_seeds),
        "preservation_contract": {
            "per_hand_action_multiset": True,
            "public_state_rows": True,
            "M0_M1_M2_features": True,
            "destroyed_information": "within-hand action order used by transition features",
        },
        "train_examples": len(original_train),
        "ood_test_examples": len(original_test),
        "overall": {
            "M2": m2_overall,
            "M3_original": m3_overall,
            "M3_shuffled_mae": overall_summary,
            **overall_effect,
        },
        "by_region": by_region,
        "prespecified_checks": {
            "original_M3_beats_M2_overall": m3_overall["mae"] < m2_overall["mae"],
            "overall_gain_at_least_50pct_attenuated": (
                overall_attenuation is not None and overall_attenuation >= 0.50
            ),
            "control_heavy_gain_at_least_50pct_attenuated": (
                control_attenuation is not None and control_attenuation >= 0.50
            ),
        },
        "replicates": replicate_reports,
        "warning": (
            "This is a synthetic feature-level negative control. Shuffled sequences are not "
            "replayed through the game engine and need not be legally realizable hands."
        ),
    }


def analyze_mixed_ood_history_destruction_files(
    training_path: str | Path,
    ood_path: str | Path,
    output_path: str | Path,
    permutation_seeds: tuple[int, ...] = DEFAULT_PERMUTATION_SEEDS,
) -> dict:
    report = analyze_mixed_ood_history_destruction(
        load_jsonl(training_path),
        load_jsonl(ood_path),
        permutation_seeds=permutation_seeds,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
