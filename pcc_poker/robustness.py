"""Frozen robustness surface for the engineered PCC payoff cycle."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import statistics

from .simulate import adaptive_pairwise_sweep


EDGE_KEYS = (
    "control_over_pressure",
    "chaos_over_control",
    "pressure_over_chaos",
)
FROZEN_POLICY_SHA256 = "ec6020ea7903365c5437ab10bf813cd1a77ab7f62613e118048d613870c0f962"


def _policy_source_sha256() -> str:
    policy_path = Path(__file__).with_name("families.py")
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def _edge_summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    margin = 1.96 * standard_error
    return {
        "mean_payoff_edge": mean,
        "standard_deviation_across_replicates": standard_deviation,
        "normal_95_interval": [mean - margin, mean + margin],
        "positive_replicates": sum(value > 0 for value in values),
        "replicates": len(values),
    }


def _run_task(task: tuple) -> dict:
    condition_id, replicate, temperature, purity, hands, seed = task
    sweep = adaptive_pairwise_sweep(
        hands_per_seat_order=hands,
        seed=seed,
        temperature=temperature,
        mode_purity=purity,
    )
    matrix = sweep["mean_payoff_focal_policy"]
    return {
        "condition_id": condition_id,
        "replicate": replicate,
        "seed": seed,
        "edges": {
            "control_over_pressure": matrix["control_vs_pressure"],
            "chaos_over_control": matrix["chaos_vs_control"],
            "pressure_over_chaos": matrix["pressure_vs_chaos"],
        },
    }


def _dominant_modes(edges: dict[str, float]) -> list[str]:
    return [
        mode
        for mode, wins in {
            "pressure": (
                edges["control_over_pressure"] < 0
                and edges["pressure_over_chaos"] > 0
            ),
            "control": (
                edges["control_over_pressure"] > 0
                and edges["chaos_over_control"] < 0
            ),
            "chaos": (
                edges["chaos_over_control"] > 0
                and edges["pressure_over_chaos"] < 0
            ),
        }.items()
        if wins
    ]


def _derived_summaries(condition_results: list[dict]) -> dict:
    stratified = {}
    for axis in ("temperature", "mode_purity", "hands_per_seat_order"):
        values = sorted({condition[axis] for condition in condition_results})
        stratified[axis] = {}
        for value in values:
            subset = [
                condition for condition in condition_results
                if condition[axis] == value
            ]
            cycles = sum(condition["complete_cycle"] for condition in subset)
            stratified[axis][str(value)] = {
                "cycle_conditions": cycles,
                "condition_count": len(subset),
                "cycle_fraction": cycles / len(subset),
            }
    failures = []
    for condition in condition_results:
        if condition["complete_cycle"]:
            continue
        edge_means = {
            key: condition["edges"][key]["mean_payoff_edge"] for key in EDGE_KEYS
        }
        failures.append({
            "condition_id": condition["condition_id"],
            "temperature": condition["temperature"],
            "mode_purity": condition["mode_purity"],
            "hands_per_seat_order": condition["hands_per_seat_order"],
            "failed_edges": [key for key, value in edge_means.items() if value <= 0],
            "edge_means": edge_means,
            "dominant_modes": condition["dominant_modes"],
        })
    return {"stratified": stratified, "failure_conditions": failures}


def enrich_robustness_report(report: dict) -> dict:
    """Add deterministic boundary summaries without rerunning simulations."""
    enriched = dict(report)
    enriched.update(_derived_summaries(report["condition_results"]))
    return enriched


def run_robustness_grid(
    temperatures: tuple[float, ...] = (0.20, 0.35, 0.50, 0.75),
    purities: tuple[float, ...] = (0.70, 0.80, 0.90, 1.00),
    hand_counts: tuple[int, ...] = (250, 1000, 4000),
    replicates: int = 10,
    seed: int = 41001,
    seed_stride: int = 20,
    minimum_cycle_fraction: float = 0.80,
    maximum_dominance_fraction: float = 0.20,
    workers: int | None = None,
) -> dict:
    """Map the v0.3 policies' cycle across a frozen parameter surface."""
    policy_sha256 = _policy_source_sha256()
    if policy_sha256 != FROZEN_POLICY_SHA256:
        raise RuntimeError(
            "Adaptive policy source differs from the frozen v0.3 mechanism"
        )
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("temperatures must contain positive values")
    if not purities or any(not 1 / 3 <= value <= 1 for value in purities):
        raise ValueError("purities must be between one-third and one")
    if not hand_counts or any(value < 1 for value in hand_counts):
        raise ValueError("hand_counts must contain positive integers")
    if replicates < 2:
        raise ValueError("at least two replicates are required")
    if seed_stride < 6:
        raise ValueError("seed_stride must be at least six")
    if not 0 <= minimum_cycle_fraction <= 1:
        raise ValueError("minimum_cycle_fraction must be between zero and one")
    if not 0 <= maximum_dominance_fraction <= 1:
        raise ValueError("maximum_dominance_fraction must be between zero and one")

    parameter_cells = list(product(temperatures, purities, hand_counts))
    conditions = [
        {
            "condition_id": f"condition-{index:03d}",
            "temperature": temperature,
            "mode_purity": purity,
            "hands_per_seat_order": hands,
        }
        for index, (temperature, purity, hands) in enumerate(parameter_cells)
    ]
    tasks = []
    for condition_index, condition in enumerate(conditions):
        for replicate in range(replicates):
            run_seed = seed + condition_index * 1000 + replicate * seed_stride
            tasks.append((
                condition["condition_id"],
                replicate,
                condition["temperature"],
                condition["mode_purity"],
                condition["hands_per_seat_order"],
                run_seed,
            ))

    effective_workers = workers or min(8, os.cpu_count() or 1)
    if effective_workers < 1:
        raise ValueError("workers must be positive")
    if effective_workers == 1:
        runs = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            runs = list(executor.map(_run_task, tasks))

    runs_by_condition = {condition["condition_id"]: [] for condition in conditions}
    for run in runs:
        runs_by_condition[run["condition_id"]].append(run)

    condition_results = []
    for condition in conditions:
        condition_runs = runs_by_condition[condition["condition_id"]]
        edges = {
            key: _edge_summary([run["edges"][key] for run in condition_runs])
            for key in EDGE_KEYS
        }
        edge_means = {key: edges[key]["mean_payoff_edge"] for key in EDGE_KEYS}
        complete_cycle = all(value > 0 for value in edge_means.values())
        condition_results.append({
            **condition,
            "edges": edges,
            "complete_cycle": complete_cycle,
            "dominant_modes": _dominant_modes(edge_means),
            "runs": condition_runs,
        })

    condition_count = len(condition_results)
    cycle_count = sum(result["complete_cycle"] for result in condition_results)
    edge_positive_counts = {
        key: sum(result["edges"][key]["mean_payoff_edge"] > 0 for result in condition_results)
        for key in EDGE_KEYS
    }
    dominance_counts = {
        mode: sum(mode in result["dominant_modes"] for result in condition_results)
        for mode in ("pressure", "control", "chaos")
    }
    cycle_fraction = cycle_count / condition_count
    dominance_fractions = {
        mode: count / condition_count for mode, count in dominance_counts.items()
    }
    acceptance = {
        "cycle_fraction_passed": cycle_fraction >= minimum_cycle_fraction,
        "no_mode_dominates_grid": (
            max(dominance_fractions.values()) <= maximum_dominance_fraction
        ),
    }
    acceptance["robustness_confirmed"] = all(acceptance.values())
    report = {
        "design": {
            "status": "frozen_out_of_sample_robustness_surface",
            "frozen_policy_version": "0.3.0",
            "frozen_policy_sha256": FROZEN_POLICY_SHA256,
            "observed_policy_sha256": policy_sha256,
            "policies_modified": False,
            "temperatures": list(temperatures),
            "purities": list(purities),
            "hand_counts": list(hand_counts),
            "replicates_per_condition": replicates,
            "conditions": condition_count,
            "total_sweeps": len(tasks),
            "total_hands": sum(
                condition["hands_per_seat_order"] * 6 * replicates
                for condition in conditions
            ),
            "seed": seed,
            "seed_stride": seed_stride,
            "seat_balancing": "every matchup is run in both seat orders",
            "minimum_cycle_fraction": minimum_cycle_fraction,
            "maximum_dominance_fraction": maximum_dominance_fraction,
            "condition_rule": "all three replicate-mean canonical edges are positive",
            "workers": effective_workers,
        },
        "aggregate": {
            "cycle_conditions": cycle_count,
            "condition_count": condition_count,
            "cycle_fraction": cycle_fraction,
            "edge_positive_conditions": edge_positive_counts,
            "edge_positive_fractions": {
                key: count / condition_count
                for key, count in edge_positive_counts.items()
            },
            "dominance_conditions": dominance_counts,
            "dominance_fractions": dominance_fractions,
            **acceptance,
        },
        "condition_results": condition_results,
        "warning": (
            "This is a robustness test of frozen engineered policies, not "
            "evidence that human poker behavior follows PCC dynamics."
        ),
    }
    return enrich_robustness_report(report)


def write_robustness_outputs(
    json_path: str | Path, csv_path: str | Path, **kwargs
) -> dict:
    report = run_robustness_grid(**kwargs)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "condition_id",
        "temperature",
        "mode_purity",
        "hands_per_seat_order",
        "complete_cycle",
        "dominant_modes",
    ]
    for edge in EDGE_KEYS:
        fieldnames.extend((f"{edge}_mean", f"{edge}_ci_low", f"{edge}_ci_high"))
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in report["condition_results"]:
            row = {
                "condition_id": condition["condition_id"],
                "temperature": condition["temperature"],
                "mode_purity": condition["mode_purity"],
                "hands_per_seat_order": condition["hands_per_seat_order"],
                "complete_cycle": condition["complete_cycle"],
                "dominant_modes": "|".join(condition["dominant_modes"]),
            }
            for edge in EDGE_KEYS:
                summary = condition["edges"][edge]
                row[f"{edge}_mean"] = summary["mean_payoff_edge"]
                row[f"{edge}_ci_low"] = summary["normal_95_interval"][0]
                row[f"{edge}_ci_high"] = summary["normal_95_interval"][1]
            writer.writerow(row)
    return report


def enrich_robustness_file(path: str | Path) -> dict:
    """Add derived summaries to an existing complete grid in place."""
    target = Path(path)
    report = json.loads(target.read_text(encoding="utf-8"))
    enriched = enrich_robustness_report(report)
    target.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
    return enriched
