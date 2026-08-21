"""Post-v0.8 synthetic strong falsification for Poker Chaos.

This experiment operationalizes the cross-game proposition that Chaos is not
randomness.  A Chaos candidate must combine behavioral unpredictability with
preserved value and resistance to an exploiter calibrated only against a
predictable baseline.  Human data are never accessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics

from .engine import equity
from .families import AdaptiveMixturePolicy, IndependentMixturePolicy
from .policies import Decision, OpponentModel, PCCPolicy
from .simulate import simulate_policy_match
from .value_floor import UniformContinuationValueModel

CHAOS_WEIGHTS = (0.1, 0.1, 0.8)
NEUTRAL_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

CALIBRATION_CANDIDATES = (
    ("adaptive-control-cold", (0.1, 0.8, 0.1), 0.15),
    ("adaptive-pressure-control-cold", (0.45, 0.45, 0.1), 0.15),
    ("adaptive-pressure-cold", (0.8, 0.1, 0.1), 0.15),
    ("adaptive-control-standard", (0.1, 0.8, 0.1), 0.35),
    ("adaptive-pressure-control-standard", (0.45, 0.45, 0.1), 0.35),
)

MIN_ENTROPY_GAIN_OVER_PREDICTABLE = 0.20
MIN_VALUE_GAIN_OVER_RANDOM = 0.30
MAX_VALUE_LOSS_VS_PREDICTABLE = 0.10
MIN_EXPLOITER_GAIN_OVER_PREDICTABLE = 0.10
MIN_EXPLOITER_GAIN_OVER_RANDOM = 0.30


class PredictableValuePolicy:
    """Deterministic public-information value baseline, not a PCC policy."""

    def __init__(self, seed: int = 0, label: str = "predictable-value") -> None:
        self.rng = random.Random(seed)
        self.label = label
        self.opponent_model = OpponentModel()
        self.action_history = OpponentModel()
        self.value_model = UniformContinuationValueModel()

    def decide(self, state) -> Decision:
        values = self.value_model.action_values(state)
        best = max(values.values())
        action = next(a for a in state.legal_actions() if abs(values[a] - best) < 1e-12)
        probabilities = {a: 1.0 if a == action else 0.0 for a in state.legal_actions()}
        self.action_history.observe(state, action)
        return Decision(action, probabilities, {}, {"pressure": 0.0, "control": 0.0, "chaos": 0.0}, equity(state, state.actor))


class UniformRandomPolicy:
    """Uniform legal-action baseline representing randomness without adequacy."""

    def __init__(self, seed: int = 0, label: str = "uniform-random") -> None:
        self.rng = random.Random(seed)
        self.label = label
        self.opponent_model = OpponentModel()
        self.action_history = OpponentModel()

    def decide(self, state) -> Decision:
        legal = state.legal_actions()
        action = self.rng.choice(list(legal))
        probabilities = {a: 1.0 / len(legal) for a in legal}
        self.action_history.observe(state, action)
        return Decision(action, probabilities, {}, {"pressure": 0.0, "control": 0.0, "chaos": 0.0}, equity(state, state.actor))


def _policy_factory(kind: str, seed: int, exploiter_spec: tuple | None = None):
    if kind == "predictable":
        return PredictableValuePolicy(seed)
    if kind == "random":
        return UniformRandomPolicy(seed)
    if kind == "score-chaos":
        return PCCPolicy(CHAOS_WEIGHTS, seed=seed, temperature=0.35, label="score-chaos")
    if kind == "independent-chaos":
        return IndependentMixturePolicy(CHAOS_WEIGHTS, seed=seed, temperature=0.35, label="independent-chaos")
    if kind == "neutral":
        return AdaptiveMixturePolicy(NEUTRAL_WEIGHTS, seed=seed, temperature=0.35, label="neutral")
    if kind == "exploiter":
        if exploiter_spec is None:
            raise ValueError("exploiter_spec required")
        label, weights, temperature = exploiter_spec
        return AdaptiveMixturePolicy(weights, seed=seed, temperature=temperature, label=label)
    raise ValueError(f"unknown policy kind: {kind}")


def _normalized_entropy(probabilities: dict[str, float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    return entropy / math.log(n)


def _seat_balanced_metrics(kind: str, opponent_kind: str, hands: int, seed: int, *, exploiter_spec=None) -> dict:
    p0 = _policy_factory(kind, seed * 8 + 1, exploiter_spec)
    p1 = _policy_factory(opponent_kind, seed * 8 + 2, exploiter_spec)
    records_a, summary_a = simulate_policy_match(hands, p0, p1, seed)

    p0 = _policy_factory(opponent_kind, (seed + 1) * 8 + 1, exploiter_spec)
    p1 = _policy_factory(kind, (seed + 1) * 8 + 2, exploiter_spec)
    records_b, summary_b = simulate_policy_match(hands, p0, p1, seed + 1)

    focal_records = [r for r in records_a if r["actor"] == 0] + [r for r in records_b if r["actor"] == 1]
    entropy = statistics.fmean(_normalized_entropy(r["action_probabilities"]) for r in focal_records)
    return {
        "mean_payoff": (summary_a["mean_payoff0"] + summary_b["mean_payoff1"]) / 2.0,
        "mean_normalized_policy_entropy": entropy,
        "focal_decisions": len(focal_records),
    }


def _calibrate_exploiter(hands_per_seat: int, seed: int) -> tuple[tuple, list[dict]]:
    rows = []
    for index, spec in enumerate(CALIBRATION_CANDIDATES):
        result = _seat_balanced_metrics("predictable", "exploiter", hands_per_seat, seed + index * 20, exploiter_spec=spec)
        rows.append({
            "label": spec[0],
            "weights": list(spec[1]),
            "temperature": spec[2],
            "predictable_mean_payoff": result["mean_payoff"],
        })
    selected_row = min(rows, key=lambda row: row["predictable_mean_payoff"])
    selected = next(spec for spec in CALIBRATION_CANDIDATES if spec[0] == selected_row["label"])
    return selected, rows


def run_chaos_strong_falsification(
    calibration_hands_per_seat: int = 500,
    evaluation_hands_per_seat: int = 400,
    replicates: int = 6,
    calibration_seed: int = 181001,
    neutral_seed: int = 191001,
    exploiter_seed: int = 201001,
    seed_stride: int = 40,
) -> dict:
    selected, calibration_rows = _calibrate_exploiter(calibration_hands_per_seat, calibration_seed)

    kinds = ("predictable", "random", "score-chaos", "independent-chaos")
    raw = {kind: {"neutral": [], "exploiter": [], "entropy": []} for kind in kinds}
    for rep in range(replicates):
        for kind in kinds:
            neutral = _seat_balanced_metrics(kind, "neutral", evaluation_hands_per_seat, neutral_seed + rep * seed_stride)
            exploit = _seat_balanced_metrics(kind, "exploiter", evaluation_hands_per_seat, exploiter_seed + rep * seed_stride, exploiter_spec=selected)
            raw[kind]["neutral"].append(neutral["mean_payoff"])
            raw[kind]["exploiter"].append(exploit["mean_payoff"])
            raw[kind]["entropy"].append(neutral["mean_normalized_policy_entropy"])

    summaries = {}
    for kind in kinds:
        summaries[kind] = {
            "mean_normalized_policy_entropy": statistics.fmean(raw[kind]["entropy"]),
            "mean_payoff_vs_neutral": statistics.fmean(raw[kind]["neutral"]),
            "mean_payoff_vs_frozen_exploiter": statistics.fmean(raw[kind]["exploiter"]),
            "replicate_payoffs_vs_neutral": raw[kind]["neutral"],
            "replicate_payoffs_vs_frozen_exploiter": raw[kind]["exploiter"],
        }

    predictable = summaries["predictable"]
    random_baseline = summaries["random"]
    family_results = {}
    for family, kind in (("score", "score-chaos"), ("independent", "independent-chaos")):
        chaos = summaries[kind]
        checks = {
            "unpredictability_exceeds_predictable_by_0_20": chaos["mean_normalized_policy_entropy"] >= predictable["mean_normalized_policy_entropy"] + MIN_ENTROPY_GAIN_OVER_PREDICTABLE,
            "value_exceeds_random_by_0_30": chaos["mean_payoff_vs_neutral"] >= random_baseline["mean_payoff_vs_neutral"] + MIN_VALUE_GAIN_OVER_RANDOM,
            "value_not_worse_than_predictable_by_more_than_0_10": chaos["mean_payoff_vs_neutral"] >= predictable["mean_payoff_vs_neutral"] - MAX_VALUE_LOSS_VS_PREDICTABLE,
            "resists_exploiter_better_than_predictable_by_0_10": chaos["mean_payoff_vs_frozen_exploiter"] >= predictable["mean_payoff_vs_frozen_exploiter"] + MIN_EXPLOITER_GAIN_OVER_PREDICTABLE,
            "resists_exploiter_better_than_random_by_0_30": chaos["mean_payoff_vs_frozen_exploiter"] >= random_baseline["mean_payoff_vs_frozen_exploiter"] + MIN_EXPLOITER_GAIN_OVER_RANDOM,
        }
        family_results[family] = {
            "policy_kind": kind,
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "metrics": chaos,
        }

    selected_row = next(row for row in calibration_rows if row["label"] == selected[0])
    calibration_competent = selected_row["predictable_mean_payoff"] <= -0.20
    confirmed = calibration_competent and all(v["all_checks_pass"] for v in family_results.values())
    return {
        "poker_chaos_strong_falsification_confirmed": confirmed,
        "headline": "Chaos is not randomness: confirmation requires unpredictability, preserved value, and resistance to a separately calibrated exploiter across both synthetic families.",
        "exploiter_calibration": {
            "selected": selected_row,
            "predictable_exploited_by_at_least_0_20": calibration_competent,
            "candidate_grid": calibration_rows,
            "selection_used_only_predictable_baseline": True,
        },
        "baselines": {
            "predictable": predictable,
            "uniform_random": random_baseline,
        },
        "families": family_results,
        "thresholds": {
            "minimum_entropy_gain_over_predictable": MIN_ENTROPY_GAIN_OVER_PREDICTABLE,
            "minimum_value_gain_over_random": MIN_VALUE_GAIN_OVER_RANDOM,
            "maximum_value_loss_vs_predictable": MAX_VALUE_LOSS_VS_PREDICTABLE,
            "minimum_exploiter_gain_over_predictable": MIN_EXPLOITER_GAIN_OVER_PREDICTABLE,
            "minimum_exploiter_gain_over_random": MIN_EXPLOITER_GAIN_OVER_RANDOM,
        },
        "design": {
            "post_v0_8_synthetic_extension": True,
            "human_data_used": False,
            "calibration_hands_per_seat": calibration_hands_per_seat,
            "evaluation_hands_per_seat": evaluation_hands_per_seat,
            "replicates": replicates,
            "calibration_seed": calibration_seed,
            "neutral_seed": neutral_seed,
            "exploiter_seed": exploiter_seed,
            "seed_stride": seed_stride,
            "seat_balanced": True,
            "exploiter_frozen_before_chaos_evaluation": True,
            "score_and_independent_chaos_policies_not_modified": True,
        },
    }


def write_chaos_strong_falsification(path: str | Path, **kwargs) -> dict:
    report = run_chaos_strong_falsification(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
