"""Prospective value-aware Score-Control intervention.

This post-v0.8 synthetic extension makes one change to the already-tested
ContextualScorePolicy: the existing contextual response term

    3.35 * (opponent_fold_probability - 1/3)

may affect an aggressive action only when a frozen, label-free counterfactual
oracle estimates that action's control efficiency at or above 0.80. The 0.80
threshold is inherited unchanged from the preceding value-bottleneck
decomposition; the contextual gain, three-stage recovery measurements, seeds,
and recovery thresholds are unchanged.

A separate synthetic-only oracle calibration is performed before intervention
calibration/evaluation. Human data are never accessed and the v0.8 human-facing
panel is not modified.
"""
from __future__ import annotations

import json
import math
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
from .score_control_intervention import (
    CONTEXT_RESPONSE_GAIN,
    NEUTRAL_FOLD_PRIOR,
    ContextualScorePolicy,
)
from .score_control_value_decomposition import LOW_EFFICIENCY_THRESHOLD
from .simulate import sample_simplex, simulate_policy_match

DECISION_ORACLE_SEEDS = {"score": 5501, "adaptive": 5509}


class ValueAwareContextualScorePolicy(PCCPolicy):
    """Context-sensitive Score Control gated by counterfactual adequacy."""

    family_name = "score_contextual_value_guarded_extension"

    def __init__(self, *args, decision_oracle: CounterfactualOracle, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_oracle = decision_oracle

    def decide(self, state):
        components = component_scores(state, self.opponent_model, self.action_history)
        fold_probability = self.opponent_model.fold_probability(state)
        contextual_delta = CONTEXT_RESPONSE_GAIN * (
            fold_probability - NEUTRAL_FOLD_PRIOR
        )

        values = self.decision_oracle.action_values(state)
        best_value = max(values.values())
        payoff_scale = max(state.pot + state.bet_size, 1)
        aggressive_efficiency = {}
        for action in state.legal_actions():
            if action not in {"bet", "raise"}:
                continue
            regret = max(best_value - values[action], 0.0)
            efficiency = math.exp(-regret / payoff_scale)
            aggressive_efficiency[action] = efficiency
            if efficiency >= LOW_EFFICIENCY_THRESHOLD:
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
    decision_oracle: CounterfactualOracle | None = None,
    intervention: bool = True,
):
    if family == "score":
        if intervention:
            if decision_oracle is None:
                raise ValueError("Score intervention requires a frozen decision oracle")
            policy_class = ValueAwareContextualScorePolicy
        else:
            policy_class = ContextualScorePolicy
    else:
        policy_class = AdaptiveMixturePolicy

    rng = random.Random(seed)
    balanced = (1 / 3, 1 / 3, 1 / 3)
    records = []
    groups = []
    for mixture_index in range(mixtures):
        mixture_id = f"{family}-mix-{seed}-{mixture_index:04d}"
        weights = sample_simplex(rng, alpha)
        for focal_seat in (0, 1):
            simulation_seed = seed * 10_000 + mixture_index * 2 + focal_seat
            kwargs = {
                "seed": simulation_seed * 2 + focal_seat,
                "temperature": focal_temperature,
                "label": mixture_id,
            }
            if family == "score" and intervention:
                kwargs["decision_oracle"] = decision_oracle
            focal = policy_class(weights, **kwargs)
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


def _build_decision_oracle(
    mixtures: int,
    hands_per_seat: int,
    score_seed: int,
    adaptive_seed: int,
) -> CounterfactualOracle:
    records = []
    # The oracle calibration is outcome-blind to the new intervention: Score uses
    # the preceding contextual-only policy and Adaptive remains unchanged.
    for family, seed in (("score", score_seed), ("adaptive", adaptive_seed)):
        batch, _ = _generate_dataset(
            family,
            mixtures,
            hands_per_seat,
            seed,
            intervention=False,
        )
        records.extend(batch)
    return CounterfactualOracle(PublicActionModel.from_records(records))


def run_score_control_value_intervention(
    oracle_mixtures: int = 20,
    oracle_hands_per_seat: int = 30,
    calibration_mixtures: int = 20,
    calibration_hands_per_seat: int = 30,
    evaluation_mixtures: int = 40,
    evaluation_hands_per_seat: int = 60,
    oracle_score_seed: int = DECISION_ORACLE_SEEDS["score"],
    oracle_adaptive_seed: int = DECISION_ORACLE_SEEDS["adaptive"],
    score_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["score"],
    adaptive_calibration_seed: int = DEFAULT_CALIBRATION_SEEDS["adaptive"],
    score_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["score"],
    adaptive_evaluation_seed: int = DEFAULT_EVALUATION_SEEDS["adaptive"],
    yoke_seed: int = DEFAULT_YOKE_SEED,
) -> dict:
    decision_oracle = _build_decision_oracle(
        oracle_mixtures,
        oracle_hands_per_seat,
        oracle_score_seed,
        oracle_adaptive_seed,
    )

    calibration = []
    for family, seed in (
        ("score", score_calibration_seed),
        ("adaptive", adaptive_calibration_seed),
    ):
        batch, _ = _generate_dataset(
            family,
            calibration_mixtures,
            calibration_hands_per_seat,
            seed,
            decision_oracle=decision_oracle,
        )
        calibration.extend(batch)

    history_model = FrozenAlignedYokedHistoryModel.from_records(calibration, seed=yoke_seed)
    measurement_oracle = CounterfactualOracle(PublicActionModel.from_records(calibration))

    evaluation = []
    for family, seed in (
        ("score", score_evaluation_seed),
        ("adaptive", adaptive_evaluation_seed),
    ):
        batch, _ = _generate_dataset(
            family,
            evaluation_mixtures,
            evaluation_hands_per_seat,
            seed,
            measurement_oracle=measurement_oracle,
            decision_oracle=decision_oracle,
        )
        evaluation.extend(batch)

    report = summarize_control_structural_recovery(evaluation, history_model)
    report["intervention"] = {
        "status": "prospective_post_v0.8_score_control_value_guard",
        "context_response_gain": CONTEXT_RESPONSE_GAIN,
        "neutral_fold_prior": NEUTRAL_FOLD_PRIOR,
        "minimum_aggressive_counterfactual_efficiency": LOW_EFFICIENCY_THRESHOLD,
        "changed_component": (
            "Score-family contextual Control term is applied to bet/raise only when "
            "a frozen synthetic counterfactual oracle estimates action efficiency >= 0.80"
        ),
        "unchanged_components": [
            "contextual response gain 3.35",
            "neutral fold prior 1/3",
            "Score card/showdown value",
            "Score flexibility",
            "Score commitment-risk penalty",
            "Pressure component",
            "Chaos component",
            "Adaptive family",
            "three-stage recovery measurements",
            "three-stage recovery thresholds",
            "three-stage recovery calibration/evaluation seeds",
        ],
        "human_data_accessed": False,
        "frozen_v0.8_human_panel_modified": False,
    }
    report["design"] = {
        "status": "post_v0.8_score_control_prospective_value_intervention",
        "decision_oracle_seeds": {
            "score": oracle_score_seed,
            "adaptive": oracle_adaptive_seed,
        },
        "decision_oracle_mixtures": oracle_mixtures,
        "decision_oracle_hands_per_seat": oracle_hands_per_seat,
        "calibration_seeds": {
            "score": score_calibration_seed,
            "adaptive": adaptive_calibration_seed,
        },
        "evaluation_seeds": {
            "score": score_evaluation_seed,
            "adaptive": adaptive_evaluation_seed,
        },
        "yoke_seed": yoke_seed,
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "weight_boundary": (
            "Synthetic PCC weights are used only after trajectory aggregation for "
            "construct-validity correlations. The decision oracle uses only the actor's "
            "information state and a separately frozen synthetic public continuation model."
        ),
    }
    return report


def write_score_control_value_intervention(path: str | Path, **kwargs) -> dict:
    report = run_score_control_value_intervention(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
