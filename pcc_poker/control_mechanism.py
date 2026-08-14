"""Matched mechanism interventions for the Control-over-Pressure edge."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from .counterfactual_control import (
    FROZEN_POLICY_SHA256,
    FrozenOpponentModel,
    _calibrate_model,
    _evaluate_fixed_model,
    _model_size,
    _policy_source_sha256,
)
from .policies import OpponentModel


CONDITIONS = ("aligned", "round_swapped", "context_yoked")
TARGETS = ("pressure", "chaos")


def _copy_model(source: OpponentModel) -> OpponentModel:
    copied = OpponentModel()
    copied.context_actions = defaultdict(
        Counter,
        {context: Counter(counts) for context, counts in source.context_actions.items()},
    )
    return copied


def round_swapped_model(source: OpponentModel) -> OpponentModel:
    """Swap round-specific facing-action counts while preserving every count."""
    swapped = _copy_model(source)
    first = Counter(source.context_actions.get("r0|facing", Counter()))
    second = Counter(source.context_actions.get("r1|facing", Counter()))
    swapped.context_actions["r0|facing"] = second
    swapped.context_actions["r1|facing"] = first
    return swapped


def context_yoked_model(source: OpponentModel, seed: int) -> OpponentModel:
    """Randomize round alignment while preserving legal-action strata."""
    yoked = OpponentModel()
    rng = random.Random(seed)
    for stratum in ("open", "facing"):
        contexts = sorted(
            context
            for context in source.context_actions
            if context.endswith(f"|{stratum}")
        )
        actions = []
        context_sizes = {}
        for context in contexts:
            counts = source.context_actions[context]
            context_sizes[context] = sum(counts.values())
            for action, count in sorted(counts.items()):
                actions.extend([action] * count)
        rng.shuffle(actions)
        cursor = 0
        for context in contexts:
            size = context_sizes[context]
            yoked.context_actions[context].update(actions[cursor:cursor + size])
            cursor += size
    return yoked


def _stratum_action_totals(model: OpponentModel) -> dict[str, Counter]:
    totals = {"open": Counter(), "facing": Counter()}
    for context, counts in model.context_actions.items():
        totals[context.rsplit("|", 1)[-1]].update(counts)
    return totals


def _action_totals(model: OpponentModel) -> Counter:
    totals = Counter()
    for counts in model.context_actions.values():
        totals.update(counts)
    return totals


def _context_sizes(model: OpponentModel) -> dict[str, int]:
    return {
        context: sum(counts.values())
        for context, counts in model.context_actions.items()
    }


def _interval(values: list[float]) -> dict:
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "normal_95_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "replicates": len(values),
    }


def run_control_pressure_mechanism(
    replicates: int = 16,
    calibration_hands_per_seat: int = 250,
    evaluation_hands_per_seat: int = 500,
    calibration_seed: int = 91001,
    evaluation_seed: int = 101001,
    seed_stride: int = 2000,
    purities: tuple[float, ...] = (0.70, 0.80, 0.90),
    temperatures: tuple[float, ...] = (0.25, 0.35, 0.50),
) -> dict:
    """Test contextual prediction while holding model margins fixed."""
    if replicates < 2:
        raise ValueError("at least two replicates are required")
    if calibration_hands_per_seat < 1 or evaluation_hands_per_seat < 1:
        raise ValueError("hand counts must be positive")
    if calibration_seed == evaluation_seed:
        raise ValueError("calibration and evaluation seeds must differ")
    if not purities or not temperatures:
        raise ValueError("the robustness surface cannot be empty")
    if any(not 1 / 3 <= purity <= 1 for purity in purities):
        raise ValueError("purities must be between one-third and one")
    if any(temperature <= 0 for temperature in temperatures):
        raise ValueError("temperatures must be positive")
    policy_sha256 = _policy_source_sha256()
    if policy_sha256 != FROZEN_POLICY_SHA256:
        raise RuntimeError("Adaptive policy source differs from frozen v0.3")

    rows = []
    margin_checks = []
    for replicate in range(replicates):
        base_calibration_seed = calibration_seed + replicate * seed_stride
        base_evaluation_seed = evaluation_seed + replicate * seed_stride
        for target_index, target_mode in enumerate(TARGETS):
            for purity_index, purity in enumerate(purities):
                for temperature_index, temperature in enumerate(temperatures):
                    cell = target_index * 100 + purity_index * 10 + temperature_index
                    model = _calibrate_model(
                        target_mode,
                        calibration_hands_per_seat,
                        base_calibration_seed + cell,
                        purity,
                        temperature,
                    )
                    round_swapped = round_swapped_model(model)
                    context_yoked = context_yoked_model(
                        model,
                        seed=base_calibration_seed + 1000 + cell,
                    )
                    condition_models = {
                        "aligned": model,
                        "round_swapped": round_swapped,
                        "context_yoked": context_yoked,
                    }
                    margin_checks.append({
                        "replicate": replicate,
                        "target_mode": target_mode,
                        "purity": purity,
                        "temperature": temperature,
                        "observations": _model_size(model),
                        "round_swapped_action_totals_preserved": (
                            _action_totals(model) == _action_totals(round_swapped)
                        ),
                        "context_yoked_action_totals_preserved": (
                            _action_totals(model) == _action_totals(context_yoked)
                        ),
                        "context_yoked_context_sizes_preserved": (
                            _context_sizes(model) == _context_sizes(context_yoked)
                        ),
                        "context_yoked_legal_strata_preserved": (
                            _stratum_action_totals(model)
                            == _stratum_action_totals(context_yoked)
                        ),
                    })
                    common_seed = base_evaluation_seed + cell
                    payoffs = {
                        condition: _evaluate_fixed_model(
                            "control",
                            target_mode,
                            model_condition,
                            evaluation_hands_per_seat,
                            common_seed,
                            purity,
                            temperature,
                        )
                        for condition, model_condition in condition_models.items()
                    }
                    rows.append({
                        "replicate": replicate,
                        "target_mode": target_mode,
                        "purity": purity,
                        "temperature": temperature,
                        "common_evaluation_seed": common_seed,
                        "mean_payoff": payoffs,
                        "aligned_minus_round_swapped": (
                            payoffs["aligned"] - payoffs["round_swapped"]
                        ),
                        "aligned_minus_context_yoked": (
                            payoffs["aligned"] - payoffs["context_yoked"]
                        ),
                    })

    contrasts = (
        "aligned_minus_round_swapped",
        "aligned_minus_context_yoked",
    )
    replicate_target = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for contrast in contrasts:
            replicate_target[(row["replicate"], row["target_mode"])][contrast].append(
                row[contrast]
            )
    replicate_summary = []
    for (replicate, target_mode), values in sorted(replicate_target.items()):
        replicate_summary.append({
            "replicate": replicate,
            "target_mode": target_mode,
            **{
                contrast: statistics.mean(values[contrast])
                for contrast in contrasts
            },
        })

    target_summary = {}
    for target_mode in TARGETS:
        selected = [
            row for row in replicate_summary if row["target_mode"] == target_mode
        ]
        target_summary[target_mode] = {
            contrast: _interval([row[contrast] for row in selected])
            for contrast in contrasts
        }

    specificity = {}
    for contrast in contrasts:
        differences = []
        for replicate in range(replicates):
            values = {
                row["target_mode"]: row[contrast]
                for row in replicate_summary
                if row["replicate"] == replicate
            }
            differences.append(values["pressure"] - values["chaos"])
        specificity[contrast] = _interval(differences)

    pressure_cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["target_mode"] != "pressure":
            continue
        cell = f"p{row['purity']:.2f}|t{row['temperature']:.2f}"
        for contrast in contrasts:
            pressure_cells[cell][contrast].append(row[contrast])
    cell_summary = {
        cell: {
            contrast: _interval(values[contrast])
            for contrast in contrasts
        }
        for cell, values in sorted(pressure_cells.items())
    }
    jointly_positive_cells = sum(
        summary[contrasts[0]]["mean"] > 0
        and summary[contrasts[1]]["mean"] > 0
        for summary in cell_summary.values()
    )

    checks = {
        "aligned_beats_round_swapped_against_pressure": (
            target_summary["pressure"][contrasts[0]]["normal_95_interval"][0] > 0
        ),
        "aligned_beats_context_yoked_against_pressure": (
            target_summary["pressure"][contrasts[1]]["normal_95_interval"][0] > 0
        ),
        "round_timing_effect_is_pressure_specific": (
            specificity[contrasts[0]]["normal_95_interval"][0] > 0
        ),
        "context_alignment_effect_is_pressure_specific": (
            specificity[contrasts[1]]["normal_95_interval"][0] > 0
        ),
        "joint_effect_positive_in_at_least_six_of_nine_cells": (
            jointly_positive_cells >= 6
        ),
        "all_matched_margins_preserved": all(
            check["round_swapped_action_totals_preserved"]
            and check["context_yoked_action_totals_preserved"]
            and check["context_yoked_context_sizes_preserved"]
            and check["context_yoked_legal_strata_preserved"]
            for check in margin_checks
        ),
    }
    return {
        "status": "completed",
        "design": {
            "status": "frozen_control_pressure_mechanism_test",
            "policy_version": "0.3.0",
            "frozen_policy_sha256": FROZEN_POLICY_SHA256,
            "observed_policy_sha256": policy_sha256,
            "policies_modified": False,
            "focal_mode": "control",
            "targets": list(TARGETS),
            "conditions": list(CONDITIONS),
            "replicates": replicates,
            "calibration_hands_per_seat": calibration_hands_per_seat,
            "evaluation_hands_per_seat": evaluation_hands_per_seat,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "seed_stride": seed_stride,
            "purities": list(purities),
            "temperatures": list(temperatures),
            "common_random_numbers_within_comparison": True,
            "seat_balanced": True,
            "model_frozen_during_evaluation": True,
            "matched_model_observation_count": True,
            "matched_global_action_counts": True,
            "matched_legal_action_strata": True,
        },
        "estimand": (
            "held-out Control payoff attributable to correctly timed and "
            "context-aligned opponent-response information"
        ),
        "target_summary": target_summary,
        "pressure_minus_chaos_specificity": specificity,
        "pressure_cell_summary": cell_summary,
        "jointly_positive_pressure_cells": jointly_positive_cells,
        "prespecified_checks": checks,
        "control_pressure_mechanism_confirmed": all(checks.values()),
        "replicate_summary": replicate_summary,
        "condition_rows": rows,
        "margin_checks": margin_checks,
        "warning": (
            "This isolates a mechanism in engineered synthetic agents. It is "
            "not evidence that human poker players possess PCC states."
        ),
    }


def write_control_pressure_mechanism(path: str | Path, **kwargs) -> dict:
    report = run_control_pressure_mechanism(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
