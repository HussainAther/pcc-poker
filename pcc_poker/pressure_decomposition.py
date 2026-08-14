"""Prospective decomposition of the engineered Pressure mechanism."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .counterfactual_control import (
    FROZEN_POLICY_SHA256,
    FrozenOpponentModel,
    _model_size,
    _policy_source_sha256,
)
from .control_mechanism import context_yoked_model
from .engine import State, equity
from .families import AdaptiveMixturePolicy, _softmax
from .policies import OpponentModel
from .simulate import mode_mixture, simulate_policy_payoff


VARIANTS = ("full", "no_fold_leverage", "no_strength_selectivity")


class PressureDecompositionPolicy(AdaptiveMixturePolicy):
    """Adaptive PCC policy with one prespecified Pressure term removed.

    Only the Pressure component is changed. Control and Chaos are inherited
    unchanged from the frozen v0.3 Adaptive policy implementation.
    """

    def __init__(self, *args, pressure_variant: str = "full", **kwargs) -> None:
        if pressure_variant not in VARIANTS:
            raise ValueError(f"unknown Pressure variant: {pressure_variant}")
        super().__init__(*args, **kwargs)
        self.pressure_variant = pressure_variant

    def _coercive_distribution(self, state: State) -> dict[str, float]:
        if self.pressure_variant == "full":
            return super()._coercive_distribution(state)

        eq = equity(state, state.actor)
        strength = 2 * eq - 1
        fold_leverage = self.opponent_model.fold_probability(state)
        if self.pressure_variant == "no_fold_leverage":
            fold_leverage = 0.0
        elif self.pressure_variant == "no_strength_selectivity":
            strength = 0.0

        scores = {}
        for action in state.legal_actions():
            if action in {"bet", "raise"}:
                scores[action] = 1.0 + fold_leverage + strength
            elif action == "call":
                scores[action] = -0.1 + 4.0 * strength
            elif action == "check":
                scores[action] = 0.1 - 0.2 * strength
            else:
                scores[action] = -0.4 - 3.0 * strength
        return _softmax(scores, self.temperature)


def _interval(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": sd,
        "normal_95_interval": [mean - 1.96 * se, mean + 1.96 * se],
        "replicates": len(values),
    }


def _calibrate_variant(
    variant: str,
    hands_per_seat: int,
    seed: int,
    purity: float,
    temperature: float,
) -> OpponentModel:
    learned = []
    for probe_seat in (0, 1):
        probe = AdaptiveMixturePolicy(
            (1 / 3, 1 / 3, 1 / 3),
            seed=seed * 8 + probe_seat * 2,
            temperature=temperature,
            label="decomposition-probe",
        )
        target = PressureDecompositionPolicy(
            mode_mixture("pressure", purity),
            seed=seed * 8 + probe_seat * 2 + 1,
            temperature=temperature,
            label=f"pressure-{variant}",
            pressure_variant=variant,
        )
        policies = (probe, target) if probe_seat == 0 else (target, probe)
        simulate_policy_payoff(hands_per_seat, policies[0], policies[1], seed + probe_seat)
        learned.append(probe.opponent_model)

    merged = OpponentModel()
    for model in learned:
        for context, counts in model.context_actions.items():
            merged.context_actions[context].update(counts)
    return merged


def _evaluate_variant(
    variant: str,
    model: OpponentModel,
    hands_per_seat: int,
    seed: int,
    purity: float,
    temperature: float,
) -> float:
    payoffs = []
    for focal_seat in (0, 1):
        focal = AdaptiveMixturePolicy(
            mode_mixture("control", purity),
            seed=seed * 8 + focal_seat * 2,
            temperature=temperature,
            label="control",
        )
        focal.opponent_model = FrozenOpponentModel(model)
        target = PressureDecompositionPolicy(
            mode_mixture("pressure", purity),
            seed=seed * 8 + focal_seat * 2 + 1,
            temperature=temperature,
            label=f"pressure-{variant}",
            pressure_variant=variant,
        )
        policies = (focal, target) if focal_seat == 0 else (target, focal)
        result = simulate_policy_payoff(hands_per_seat, policies[0], policies[1], seed + focal_seat)
        payoffs.append(result[focal_seat])
    return statistics.mean(payoffs)


def run_pressure_decomposition(
    replicates: int = 16,
    calibration_hands_per_seat: int = 250,
    evaluation_hands_per_seat: int = 500,
    calibration_seed: int = 111001,
    evaluation_seed: int = 121001,
    seed_stride: int = 2000,
    purity: float = 0.8,
    temperature: float = 0.35,
    minimum_attenuation: float = 0.50,
) -> dict:
    """Ask which Pressure term sustains Control's contextual-alignment gain."""
    if replicates < 2:
        raise ValueError("at least two replicates are required")
    if calibration_hands_per_seat < 1 or evaluation_hands_per_seat < 1:
        raise ValueError("hand counts must be positive")
    if calibration_seed == evaluation_seed:
        raise ValueError("calibration and evaluation seeds must differ")
    if not 1 / 3 <= purity <= 1:
        raise ValueError("purity must be between one-third and one")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < minimum_attenuation < 1:
        raise ValueError("minimum attenuation must be between zero and one")
    if _policy_source_sha256() != FROZEN_POLICY_SHA256:
        raise RuntimeError("Adaptive policy source differs from frozen v0.3")

    rows = []
    for replicate in range(replicates):
        calibration_base = calibration_seed + replicate * seed_stride
        evaluation_base = evaluation_seed + replicate * seed_stride
        for variant_index, variant in enumerate(VARIANTS):
            model = _calibrate_variant(
                variant,
                calibration_hands_per_seat,
                calibration_base + variant_index * 100,
                purity,
                temperature,
            )
            yoked = context_yoked_model(
                model,
                seed=calibration_base + variant_index * 100 + 50,
            )
            common_seed = evaluation_base + variant_index * 100
            aligned = _evaluate_variant(
                variant, model, evaluation_hands_per_seat, common_seed, purity, temperature
            )
            misaligned = _evaluate_variant(
                variant, yoked, evaluation_hands_per_seat, common_seed, purity, temperature
            )
            rows.append({
                "replicate": replicate,
                "pressure_variant": variant,
                "model_observations": _model_size(model),
                "common_evaluation_seed": common_seed,
                "aligned_payoff": aligned,
                "context_yoked_payoff": misaligned,
                "alignment_effect": aligned - misaligned,
            })

    by_variant = {
        variant: _interval([
            row["alignment_effect"] for row in rows if row["pressure_variant"] == variant
        ])
        for variant in VARIANTS
    }
    attenuation = {}
    full_by_rep = {
        row["replicate"]: row["alignment_effect"]
        for row in rows if row["pressure_variant"] == "full"
    }
    for variant in VARIANTS[1:]:
        differences = [
            full_by_rep[row["replicate"]] - row["alignment_effect"]
            for row in rows if row["pressure_variant"] == variant
        ]
        attenuation[variant] = _interval(differences)

    full_mean = by_variant["full"]["mean"]
    attenuation_fraction = {
        variant: (
            attenuation[variant]["mean"] / full_mean if full_mean != 0 else None
        )
        for variant in VARIANTS[1:]
    }
    qualifying = [
        variant for variant in VARIANTS[1:]
        if attenuation[variant]["normal_95_interval"][0] > 0
        and attenuation_fraction[variant] is not None
        and attenuation_fraction[variant] >= minimum_attenuation
    ]
    checks = {
        "full_pressure_alignment_effect_positive": (
            by_variant["full"]["normal_95_interval"][0] > 0
        ),
        "at_least_one_ablation_reduces_effect": bool(qualifying),
        "qualifying_ablation_reduces_effect_by_threshold": bool(qualifying),
        "frozen_control_policy_unchanged": True,
    }
    return {
        "status": "completed",
        "design": {
            "status": "prospective_pressure_decomposition",
            "policy_version": "0.3.0",
            "frozen_policy_sha256": FROZEN_POLICY_SHA256,
            "observed_policy_sha256": _policy_source_sha256(),
            "control_policy_modified": False,
            "pressure_variants": list(VARIANTS),
            "replicates": replicates,
            "calibration_hands_per_seat": calibration_hands_per_seat,
            "evaluation_hands_per_seat": evaluation_hands_per_seat,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "seed_stride": seed_stride,
            "purity": purity,
            "temperature": temperature,
            "minimum_attenuation": minimum_attenuation,
            "common_random_numbers_within_alignment_comparison": True,
            "seat_balanced": True,
        },
        "estimand": "loss of Control contextual-alignment advantage after removing one engineered Pressure term",
        "variant_summary": by_variant,
        "full_minus_ablation": attenuation,
        "attenuation_fraction_of_full_effect": attenuation_fraction,
        "qualifying_mechanisms": qualifying,
        "prespecified_checks": checks,
        "pressure_mechanism_decomposition_confirmed": all(checks.values()),
        "rows": rows,
        "warning": (
            "This decomposes engineered Adaptive Pressure. It does not establish "
            "that human poker players instantiate these mechanisms."
        ),
    }


def write_pressure_decomposition(path: str | Path, **kwargs) -> dict:
    report = run_pressure_decomposition(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
