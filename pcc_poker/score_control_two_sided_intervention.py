"""Prospective post-v0.8 two-sided Score-Control intervention.

The previously frozen aggressive contextual term is preserved exactly:

    3.35 * (opponent_fold_probability - 1/3)

This one-shot extension adds the missing passive/optionality half identified by
matched-state Adaptive-vs-Score decomposition. For check/call actions only:

    1.24 * (1/3 - opponent_fold_probability)

The passive gain 1.24 is fixed prospectively as
3.35 * (0.170154 / 0.460473), using only the preceding frozen matched-state
summary: Adaptive's low-fold passive-mass advantage divided by its matched-state
mean value advantage, mapped onto the existing contextual score scale.

No recovery threshold, seed, human-facing contract, Pressure/Chaos component, or
Adaptive-family policy is changed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .behavioral import CounterfactualOracle, PublicActionModel
from .contextual_control_observable import FrozenAlignedYokedHistoryModel
from .control_structural_recovery import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    DEFAULT_YOKE_SEED,
    summarize_control_structural_recovery,
)
from .engine import equity
from .families import AdaptiveMixturePolicy
from .policies import Decision, MODES, PCCPolicy, _softmax, component_scores
from .score_control_intervention import CONTEXT_RESPONSE_GAIN, NEUTRAL_FOLD_PRIOR
from .simulate import sample_simplex, simulate_policy_match

MATCHED_PASSIVE_MASS_ADVANTAGE = 0.1701541715043185
MATCHED_VALUE_ADVANTAGE = 0.46047259137951846
PASSIVE_OPTIONALITY_GAIN = round(
    CONTEXT_RESPONSE_GAIN * MATCHED_PASSIVE_MASS_ADVANTAGE / MATCHED_VALUE_ADVANTAGE,
    2,
)


class TwoSidedContextualScorePolicy(PCCPolicy):
    """Score family with frozen aggression gain plus passive resistance response."""

    family_name = "score_two_sided_context_extension"

    def decide(self, state):
        components = component_scores(state, self.opponent_model, self.action_history)
        fold_probability = self.opponent_model.fold_probability(state)
        aggressive_delta = CONTEXT_RESPONSE_GAIN * (fold_probability - NEUTRAL_FOLD_PRIOR)
        passive_delta = PASSIVE_OPTIONALITY_GAIN * (NEUTRAL_FOLD_PRIOR - fold_probability)
        for action in state.legal_actions():
            if action in {"bet", "raise"}:
                components["control"][action] += aggressive_delta
            elif action in {"check", "call"}:
                components["control"][action] += passive_delta

        combined = {
            action: sum(
                self.weights[index] * components[mode][action]
                for index, mode in enumerate(MODES)
            )
            for action in state.legal_actions()
        }
        probabilities = _softmax(combined, self.temperature)
        threshold = self.rng.random()
        cumulative = 0.0
        selected = next(iter(probabilities))
        for action, probability in probabilities.items():
            cumulative += probability
            if threshold <= cumulative:
                selected = action
                break
        self.action_history.observe(state, selected)
        return Decision(
            selected,
            probabilities,
            components,
            dict(zip(MODES, self.weights.tolist())),
            equity(state, state.actor),
        )


def _generate_dataset(
    family: str,
    mixtures: int,
    hands_per_seat: int,
    seed: int,
    alpha: float = 0.7,
    focal_temperature: float = 0.35,
    measurement_oracle=None,
):
    policy_class = TwoSidedContextualScorePolicy if family == "score" else AdaptiveMixturePolicy
    rng = random.Random(seed)
    balanced = (1 / 3, 1 / 3, 1 / 3)
    records = []
    groups = []
    for mixture_index in range(mixtures):
        mixture_id = f"{family}-mix-{seed}-{mixture_index:04d}"
        weights = sample_simplex(rng, alpha)
        for focal_seat in (0, 1):
            simulation_seed = seed * 10_000 + mixture_index * 2 + focal_seat
            focal = policy_class(
                weights,
                seed=simulation_seed * 2 + focal_seat,
                temperature=focal_temperature,
                label=mixture_id,
            )
            reference = PCCPolicy(
                balanced,
                seed=simulation_seed * 2 + 10,
                label="balanced_reference",
            )
            policies = [reference, reference]
            policies[focal_seat] = focal
            batch, _ = simulate_policy_match(
                hands_per_seat,
                policies[0],
                policies[1],
                simulation_seed,
                measurement_oracle,
            )
            for record in batch:
                record["mixture_id"] = mixture_id
                record["simulation_seed"] = simulation_seed
                record["focal_seat"] = focal_seat
                record["is_focal_policy"] = record["actor"] == focal_seat
                record["target_pcc_weights"] = dict(zip(MODES, weights))
                record["policy_family"] = family
            records.extend(batch)
        groups.append({"mixture_id": mixture_id, "weights": dict(zip(MODES, weights))})
    return records, {
        "family": family,
        "mixtures": mixtures,
        "hands_per_seat": hands_per_seat,
        "total_hands": mixtures * hands_per_seat * 2,
        "seed": seed,
        "dirichlet_alpha": alpha,
        "focal_temperature": focal_temperature,
        "groups": groups,
    }


def run_score_control_two_sided_intervention(
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 30,
    evaluation_mixtures: int = 40,
    evaluation_hands_per_seat: int = 60,
    score_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["score"],
    adaptive_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["adaptive"],
    score_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["score"],
    adaptive_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["adaptive"],
    yoke_seed: int = DEFAULT_YOKE_SEED,
) -> dict:
    calibration = []
    for family, seed in (("score", score_calibration_seed), ("adaptive", adaptive_calibration_seed)):
        batch, _ = _generate_dataset(family, calibration_mixtures, calibration_hands_per_seat, seed)
        calibration.extend(batch)

    history_model = FrozenAlignedYokedHistoryModel.from_records(calibration, seed=yoke_seed)
    value_oracle = CounterfactualOracle(PublicActionModel.from_records(calibration))

    evaluation = []
    for family, seed in (("score", score_evaluation_seed), ("adaptive", adaptive_evaluation_seed)):
        batch, _ = _generate_dataset(
            family,
            evaluation_mixtures,
            evaluation_hands_per_seat,
            seed,
            measurement_oracle=value_oracle,
        )
        evaluation.extend(batch)

    report = summarize_control_structural_recovery(evaluation, history_model)
    report["intervention"] = {
        "status": "prospective_post_v0.8_two_sided_score_control",
        "aggressive_context_gain": CONTEXT_RESPONSE_GAIN,
        "passive_optionality_gain": PASSIVE_OPTIONALITY_GAIN,
        "neutral_fold_prior": NEUTRAL_FOLD_PRIOR,
        "passive_gain_derivation": "3.35 * (0.1701541715 / 0.4604725914), rounded prospectively to 1.24",
        "changed_component": "Score-family Control check/call contextual optionality only",
        "unchanged_components": [
            "existing Score aggressive contextual gain",
            "Score card/showdown value",
            "Score flexibility",
            "Score commitment-risk penalty",
            "Pressure component",
            "Chaos component",
            "Adaptive family",
            "three-stage recovery seeds and thresholds",
        ],
        "human_data_accessed": False,
        "frozen_v0.8_human_panel_modified": False,
    }
    report["design"] = {
        "status": "post_v0.8_score_control_two_sided_intervention",
        "calibration_seeds": {"score": score_calibration_seed, "adaptive": adaptive_calibration_seed},
        "evaluation_seeds": {"score": score_evaluation_seed, "adaptive": adaptive_evaluation_seed},
        "yoke_seed": yoke_seed,
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "weight_boundary": "Synthetic PCC weights are used only after trajectory aggregation for construct-validity correlations.",
    }
    return report


def write_score_control_two_sided_intervention(path: str | Path, **kwargs) -> dict:
    report = run_score_control_two_sided_intervention(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
