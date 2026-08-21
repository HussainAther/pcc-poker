"""Prospective post-v0.8 intervention for Score-family Poker Control.

This extension makes one prespecified change to the Score Control component:
an additive, zero-centered contextual response term on aggressive actions,

    3.35 * (opponent_fold_probability - 1/3)

The gain is fixed from the preceding Score-vs-Adaptive sensitivity decomposition:
the original Control fold-leverage coefficient (0.55) multiplied by the observed
Adaptive/Score total-variation sensitivity ratio (~6.09). At the neutral prior
fold probability of 1/3 the original Score policy is unchanged. Existing card
value, flexibility, commitment risk, Pressure, and Chaos terms are untouched.

The intervention is evaluated with the already-frozen three-stage Control
structural-recovery gate. Human data are never accessed and the v0.8 human-facing
panel is not modified.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .behavioral import CounterfactualOracle, PublicActionModel
from .contextual_control_observable import FrozenAlignedYokedHistoryModel
from .control_structural_recovery import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    DEFAULT_YOKE_SEED,
    summarize_control_structural_recovery,
)
from .families import AdaptiveMixturePolicy
from .policies import Decision, MODES, PCCPolicy, _softmax, component_scores
from .simulate import sample_simplex, simulate_policy_match
from .engine import equity

CONTEXT_RESPONSE_GAIN = 3.35
NEUTRAL_FOLD_PRIOR = 1.0 / 3.0


class ContextualScorePolicy(PCCPolicy):
    """Score-family policy with one prospective contextual Control gain."""

    family_name = "score_contextual_response_extension"

    def decide(self, state):
        components = component_scores(state, self.opponent_model, self.action_history)
        fold_probability = self.opponent_model.fold_probability(state)
        contextual_delta = CONTEXT_RESPONSE_GAIN * (fold_probability - NEUTRAL_FOLD_PRIOR)
        for action in state.legal_actions():
            if action in {"bet", "raise"}:
                components["control"][action] += contextual_delta

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
    policy_class = ContextualScorePolicy if family == "score" else AdaptiveMixturePolicy
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
                # Keep the existing gate's family names unchanged.
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


def run_score_control_intervention(
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
        "status": "prospective_post_v0.8_score_control_context_gain",
        "context_response_gain": CONTEXT_RESPONSE_GAIN,
        "neutral_fold_prior": NEUTRAL_FOLD_PRIOR,
        "changed_component": "Score-family Control aggressive action score only",
        "unchanged_components": [
            "Score card/showdown value",
            "Score flexibility",
            "Score commitment-risk penalty",
            "Pressure component",
            "Chaos component",
            "Adaptive family",
            "three-stage recovery thresholds",
        ],
        "human_data_accessed": False,
        "frozen_v0.8_human_panel_modified": False,
    }
    report["design"] = {
        "status": "post_v0.8_score_control_prospective_intervention",
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


def write_score_control_intervention(path: str | Path, **kwargs) -> dict:
    report = run_score_control_intervention(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
