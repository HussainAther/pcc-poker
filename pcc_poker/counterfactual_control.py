"""Causal opponent-model interventions for synthetic Control validation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from .families import AdaptiveMixturePolicy
from .policies import MODES, OpponentModel
from .simulate import mode_mixture, simulate_policy_payoff


FROZEN_POLICY_SHA256 = "ec6020ea7903365c5437ab10bf813cd1a77ab7f62613e118048d613870c0f962"
CONDITIONS = ("aligned", "swapped", "prior")
DONOR_MODE = {"pressure": "control", "control": "chaos", "chaos": "pressure"}


def _policy_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).with_name("families.py").read_bytes()).hexdigest()


class FrozenOpponentModel(OpponentModel):
    """Read-only copy used to isolate model identity during evaluation."""

    def __init__(self, source: OpponentModel | None = None) -> None:
        super().__init__()
        if source is not None:
            self.context_actions = defaultdict(
                Counter,
                {
                    context: Counter(counts)
                    for context, counts in source.context_actions.items()
                },
            )

    def observe(self, state, action: str) -> None:
        """Deliberately ignore evaluation actions so the intervention is fixed."""


def _merge_models(models: list[OpponentModel]) -> OpponentModel:
    merged = OpponentModel()
    for model in models:
        for context, counts in model.context_actions.items():
            merged.context_actions[context].update(counts)
    return merged


def _model_size(model: OpponentModel) -> int:
    return sum(sum(counts.values()) for counts in model.context_actions.values())


def _calibrate_model(
    target_mode: str,
    hands_per_seat: int,
    seed: int,
    purity: float,
    temperature: float,
) -> OpponentModel:
    """Observe one target mode using a balanced probe in both seat orders."""
    learned = []
    for probe_seat in (0, 1):
        probe = AdaptiveMixturePolicy(
            (1 / 3, 1 / 3, 1 / 3),
            seed=seed * 8 + probe_seat * 2,
            temperature=temperature,
            label="calibration-probe",
        )
        target = AdaptiveMixturePolicy(
            mode_mixture(target_mode, purity),
            seed=seed * 8 + probe_seat * 2 + 1,
            temperature=temperature,
            label=f"calibration-{target_mode}",
        )
        policies = (probe, target) if probe_seat == 0 else (target, probe)
        simulate_policy_payoff(
            hands_per_seat,
            policies[0],
            policies[1],
            seed + probe_seat,
        )
        learned.append(probe.opponent_model)
    return _merge_models(learned)


def _evaluate_fixed_model(
    focal_mode: str,
    target_mode: str,
    model: OpponentModel,
    hands_per_seat: int,
    seed: int,
    purity: float,
    temperature: float,
) -> float:
    """Return seat-balanced focal payoff with one frozen opponent model."""
    payoffs = []
    for focal_seat in (0, 1):
        focal = AdaptiveMixturePolicy(
            mode_mixture(focal_mode, purity),
            seed=seed * 8 + focal_seat * 2,
            temperature=temperature,
            label=focal_mode,
        )
        focal.opponent_model = FrozenOpponentModel(model)
        target = AdaptiveMixturePolicy(
            mode_mixture(target_mode, purity),
            seed=seed * 8 + focal_seat * 2 + 1,
            temperature=temperature,
            label=target_mode,
        )
        policies = (focal, target) if focal_seat == 0 else (target, focal)
        result = simulate_policy_payoff(
            hands_per_seat,
            policies[0],
            policies[1],
            seed + focal_seat,
        )
        payoffs.append(result[focal_seat])
    return statistics.mean(payoffs)


def _interval(values: list[float]) -> dict:
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values)) if values else 0.0
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "normal_95_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "replicates": len(values),
    }


def run_counterfactual_control_validation(
    replicates: int = 16,
    calibration_hands_per_seat: int = 250,
    evaluation_hands_per_seat: int = 500,
    calibration_seed: int = 71001,
    evaluation_seed: int = 81001,
    seed_stride: int = 1000,
    purity: float = 0.8,
    temperature: float = 0.35,
) -> dict:
    """Intervene on model alignment without changing the frozen policies."""
    if replicates < 2:
        raise ValueError("at least two replicates are required")
    if calibration_hands_per_seat < 1 or evaluation_hands_per_seat < 1:
        raise ValueError("hand counts must be positive")
    if calibration_seed == evaluation_seed:
        raise ValueError("calibration and evaluation seeds must differ")
    policy_sha256 = _policy_source_sha256()
    if policy_sha256 != FROZEN_POLICY_SHA256:
        raise RuntimeError("Adaptive policy source differs from the frozen v0.3 mechanism")

    rows = []
    calibration_sizes = []
    for replicate in range(replicates):
        replicate_calibration_seed = calibration_seed + replicate * seed_stride
        replicate_evaluation_seed = evaluation_seed + replicate * seed_stride
        models = {
            mode: _calibrate_model(
                mode,
                calibration_hands_per_seat,
                replicate_calibration_seed + mode_index * 10,
                purity,
                temperature,
            )
            for mode_index, mode in enumerate(MODES)
        }
        calibration_sizes.append({
            mode: _model_size(model) for mode, model in models.items()
        })
        for target_index, target_mode in enumerate(MODES):
            donor_mode = DONOR_MODE[target_mode]
            condition_models = {
                "aligned": models[target_mode],
                "swapped": models[donor_mode],
                "prior": OpponentModel(),
            }
            for focal_index, focal_mode in enumerate(MODES):
                common_seed = (
                    replicate_evaluation_seed
                    + target_index * 100
                    + focal_index * 10
                )
                payoffs = {
                    condition: _evaluate_fixed_model(
                        focal_mode,
                        target_mode,
                        model,
                        evaluation_hands_per_seat,
                        common_seed,
                        purity,
                        temperature,
                    )
                    for condition, model in condition_models.items()
                }
                rows.append({
                    "replicate": replicate,
                    "focal_mode": focal_mode,
                    "target_mode": target_mode,
                    "donor_mode": donor_mode,
                    "common_evaluation_seed": common_seed,
                    "mean_payoff": payoffs,
                    "aligned_minus_swapped": payoffs["aligned"] - payoffs["swapped"],
                    "aligned_minus_prior": payoffs["aligned"] - payoffs["prior"],
                })

    replicate_mode = defaultdict(lambda: defaultdict(list))
    target_mode = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for contrast in ("aligned_minus_swapped", "aligned_minus_prior"):
            replicate_mode[(row["replicate"], row["focal_mode"])][contrast].append(
                row[contrast]
            )
            target_mode[(row["target_mode"], row["focal_mode"])][contrast].append(
                row[contrast]
            )

    replicate_summary = []
    for (replicate, focal_mode), contrasts in sorted(replicate_mode.items()):
        replicate_summary.append({
            "replicate": replicate,
            "focal_mode": focal_mode,
            **{
                contrast: statistics.mean(values)
                for contrast, values in contrasts.items()
            },
        })

    mode_summary = {}
    for focal_mode in MODES:
        mode_rows = [row for row in replicate_summary if row["focal_mode"] == focal_mode]
        mode_summary[focal_mode] = {
            contrast: _interval([row[contrast] for row in mode_rows])
            for contrast in ("aligned_minus_swapped", "aligned_minus_prior")
        }

    specificity_values = []
    for replicate in range(replicates):
        by_mode = {
            row["focal_mode"]: row["aligned_minus_swapped"]
            for row in replicate_summary
            if row["replicate"] == replicate
        }
        specificity_values.append(
            by_mode["control"] - max(by_mode["pressure"], by_mode["chaos"])
        )
    specificity = _interval(specificity_values)

    by_target = {}
    control_target_wins = 0
    for target in MODES:
        by_target[target] = {}
        target_means = {}
        for focal in MODES:
            values = target_mode[(target, focal)]["aligned_minus_swapped"]
            by_target[target][focal] = _interval(values)
            target_means[focal] = statistics.mean(values)
        if target_means["control"] > max(target_means["pressure"], target_means["chaos"]):
            control_target_wins += 1

    checks = {
        "control_aligned_beats_swapped": (
            mode_summary["control"]["aligned_minus_swapped"]["normal_95_interval"][0] > 0
        ),
        "control_aligned_beats_prior": (
            mode_summary["control"]["aligned_minus_prior"]["normal_95_interval"][0] > 0
        ),
        "control_model_dependence_is_discriminant": (
            specificity["normal_95_interval"][0] > 0
        ),
        "control_largest_for_at_least_two_targets": control_target_wins >= 2,
    }
    return {
        "status": "completed",
        "design": {
            "status": "frozen_counterfactual_control_validation",
            "policy_version": "0.3.0",
            "frozen_policy_sha256": FROZEN_POLICY_SHA256,
            "observed_policy_sha256": policy_sha256,
            "policies_modified": False,
            "replicates": replicates,
            "calibration_hands_per_seat": calibration_hands_per_seat,
            "evaluation_hands_per_seat": evaluation_hands_per_seat,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "seed_stride": seed_stride,
            "purity": purity,
            "temperature": temperature,
            "conditions": list(CONDITIONS),
            "common_random_numbers_within_comparison": True,
            "seat_balanced": True,
            "model_frozen_during_evaluation": True,
            "donor_mapping": DONOR_MODE,
        },
        "estimand": (
            "change in held-out, seat-balanced chip payoff caused only by replacing "
            "the focal policy's calibrated opponent model"
        ),
        "calibration_model_observations": calibration_sizes,
        "mode_summary": mode_summary,
        "control_specificity": specificity,
        "by_target": by_target,
        "control_target_wins": control_target_wins,
        "prespecified_checks": checks,
        "counterfactual_control_confirmed": all(checks.values()),
        "replicate_summary": replicate_summary,
        "condition_rows": rows,
        "warning": (
            "This is a causal intervention on engineered synthetic policies. It "
            "does not establish that human poker players possess a PCC Control state."
        ),
    }


def write_counterfactual_control_validation(path: str | Path, **kwargs) -> dict:
    report = run_counterfactual_control_validation(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
