"""Simulation and experiment logging."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from .engine import apply_action, initial_state, utility
from .families import AdaptiveMixturePolicy, IndependentMixturePolicy
from .policies import MODES, PCCPolicy, PURE_MIXTURES


def play_hand(
    policy0: PCCPolicy,
    policy1: PCCPolicy,
    deck: list[int],
    hand_id: str,
    measurement_oracle=None,
) -> tuple[list[dict], tuple[float, float]]:
    state = initial_state(deck); policies = (policy0, policy1); records = []
    while not state.terminal:
        actor = state.actor; decision = policies[actor].decide(state)
        record = {
            "hand_id": hand_id, "decision_index": len(records), "actor": actor,
            "round_index": state.round_index, "public_rank": state.public,
            "private_rank": state.private[actor], "pot": state.pot,
            "to_call": state.to_call, "legal_actions": list(state.legal_actions()),
            "history": list(state.history), "action": decision.action,
            "action_probabilities": decision.probabilities,
            "component_scores": decision.component_scores,
            "hidden_pcc_weights": decision.weights,
            "policy_label": policies[actor].label, "showdown_equity": decision.equity,
        }
        if measurement_oracle is not None:
            record["behavioral_measurements"] = measurement_oracle.measure(
                state, decision.action
            ).as_dict()
        records.append(record)
        policies[1 - actor].opponent_model.observe(state, decision.action)
        state = apply_action(state, decision.action)
    payoffs = (utility(state, 0), utility(state, 1))
    for record in records:
        record["terminal_payoff"] = payoffs[record["actor"]]
    return records, payoffs


def simulate_match(
    hands: int,
    mixture0=(0.8, 0.1, 0.1), mixture1=(1 / 3, 1 / 3, 1 / 3),
    seed: int = 7, label0: str | None = None, label1: str | None = None,
    temperature0: float = 0.35, temperature1: float = 0.35,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    policy0 = PCCPolicy(
        mixture0, seed=seed * 2 + 1, label=label0, temperature=temperature0
    )
    policy1 = PCCPolicy(
        mixture1, seed=seed * 2 + 2, label=label1, temperature=temperature1
    )
    records = []; totals = [0.0, 0.0]
    for index in range(hands):
        deck = [0, 0, 1, 1, 2, 2]; rng.shuffle(deck)
        hand_records, payoffs = play_hand(policy0, policy1, deck, f"match-{seed}-hand-{index}")
        records.extend(hand_records); totals[0] += payoffs[0]; totals[1] += payoffs[1]
    return records, {
        "hands": hands, "seed": seed, "policy0": policy0.label, "policy1": policy1.label,
        "mixture0": list(policy0.weights), "mixture1": list(policy1.weights),
        "temperature0": temperature0, "temperature1": temperature1,
        "mean_payoff0": totals[0] / hands, "mean_payoff1": totals[1] / hands,
    }


def simulate_policy_match(
    hands: int,
    policy0,
    policy1,
    seed: int,
    measurement_oracle=None,
) -> tuple[list[dict], dict]:
    """Play already-constructed policies from any compatible family."""
    rng = random.Random(seed)
    records = []
    totals = [0.0, 0.0]
    for index in range(hands):
        deck = [0, 0, 1, 1, 2, 2]
        rng.shuffle(deck)
        hand_records, payoffs = play_hand(
            policy0,
            policy1,
            deck,
            f"family-{seed}-hand-{index}",
            measurement_oracle,
        )
        records.extend(hand_records)
        totals[0] += payoffs[0]
        totals[1] += payoffs[1]
    return records, {
        "hands": hands,
        "seed": seed,
        "policy0": policy0.label,
        "policy1": policy1.label,
        "mean_payoff0": totals[0] / hands,
        "mean_payoff1": totals[1] / hands,
    }


def simulate_policy_payoff(hands: int, policy0, policy1, seed: int) -> tuple[float, float]:
    """Fast payoff-only simulation for replicated balance experiments."""
    if hands < 1:
        raise ValueError("hands must be positive")
    rng = random.Random(seed)
    totals = [0.0, 0.0]
    policies = (policy0, policy1)
    for _ in range(hands):
        deck = [0, 0, 1, 1, 2, 2]
        rng.shuffle(deck)
        state = initial_state(deck)
        while not state.terminal:
            actor = state.actor
            decision = policies[actor].decide(state)
            policies[1 - actor].opponent_model.observe(state, decision.action)
            state = apply_action(state, decision.action)
        totals[0] += utility(state, 0)
        totals[1] += utility(state, 1)
    return totals[0] / hands, totals[1] / hands


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def pairwise_sweep(hands_per_matchup: int = 2000, seed: int = 17) -> dict:
    matrix = {}; modes = list(MODES); matchup_index = 0
    for left_index, left in enumerate(modes):
        matrix[f"{left}_vs_{left}"] = 0.0
        for right in modes[left_index + 1:]:
            matchup_seed = seed + matchup_index * 2
            _, left_first = simulate_match(
                hands_per_matchup, PURE_MIXTURES[left], PURE_MIXTURES[right],
                matchup_seed, left, right,
            )
            _, right_first = simulate_match(
                hands_per_matchup, PURE_MIXTURES[right], PURE_MIXTURES[left],
                matchup_seed + 1, right, left,
            )
            matrix[f"{left}_vs_{right}"] = (
                left_first["mean_payoff0"] + right_first["mean_payoff1"]
            ) / 2
            matrix[f"{right}_vs_{left}"] = -matrix[f"{left}_vs_{right}"]
            matchup_index += 1
    proposed = {
        "control_over_pressure": matrix["control_vs_pressure"] > 0,
        "chaos_over_control": matrix["chaos_vs_control"] > 0,
        "pressure_over_chaos": matrix["pressure_vs_chaos"] > 0,
    }
    return {
        "hands_per_matchup": hands_per_matchup, "seed": seed,
        "mean_payoff_focal_policy": matrix, "proposed_cycle": proposed,
        "complete_cycle_observed": all(proposed.values()),
        "design": "each ordered comparison averages focal-policy payoff across both seats",
        "warning": "No cyclic bonuses are encoded; this is a policy-specific simulation result.",
    }


def mode_mixture(mode: str, purity: float) -> tuple[float, float, float]:
    """Create a symmetric PCC mixture with ``purity`` on the named axis."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if not 1 / 3 <= purity <= 1:
        raise ValueError("purity must be between one-third and one")
    remainder = (1.0 - purity) / 2.0
    return tuple(purity if candidate == mode else remainder for candidate in MODES)


def adaptive_pairwise_sweep(
    hands_per_seat_order: int = 4000,
    seed: int = 901,
    temperature: float = 0.35,
    mode_purity: float = 0.8,
) -> dict:
    """Seat-balanced payoff matrix for the three playable Adaptive PCC AIs."""
    if hands_per_seat_order < 1:
        raise ValueError("hands_per_seat_order must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    matrix = {}
    modes = list(MODES)
    matchup_index = 0
    for left_index, left in enumerate(modes):
        matrix[f"{left}_vs_{left}"] = 0.0
        for right in modes[left_index + 1:]:
            matchup_seed = seed + matchup_index * 2
            left_policy = AdaptiveMixturePolicy(
                mode_mixture(left, mode_purity),
                seed=matchup_seed * 2,
                label=left,
                temperature=temperature,
            )
            right_policy = AdaptiveMixturePolicy(
                mode_mixture(right, mode_purity),
                seed=matchup_seed * 2 + 1,
                label=right,
                temperature=temperature,
            )
            left_first = simulate_policy_payoff(
                hands_per_seat_order, left_policy, right_policy, matchup_seed
            )
            right_policy = AdaptiveMixturePolicy(
                mode_mixture(right, mode_purity),
                seed=(matchup_seed + 1) * 2,
                label=right,
                temperature=temperature,
            )
            left_policy = AdaptiveMixturePolicy(
                mode_mixture(left, mode_purity),
                seed=(matchup_seed + 1) * 2 + 1,
                label=left,
                temperature=temperature,
            )
            right_first = simulate_policy_payoff(
                hands_per_seat_order,
                right_policy,
                left_policy,
                matchup_seed + 1,
            )
            value = (left_first[0] + right_first[1]) / 2
            matrix[f"{left}_vs_{right}"] = value
            matrix[f"{right}_vs_{left}"] = -value
            matchup_index += 1
    proposed = {
        "control_over_pressure": matrix["control_vs_pressure"] > 0,
        "chaos_over_control": matrix["chaos_vs_control"] > 0,
        "pressure_over_chaos": matrix["pressure_vs_chaos"] > 0,
    }
    return {
        "hands_per_seat_order": hands_per_seat_order,
        "seed": seed,
        "temperature": temperature,
        "mode_purity": mode_purity,
        "mean_payoff_focal_policy": matrix,
        "proposed_cycle": proposed,
        "complete_cycle_observed": all(proposed.values()),
        "balance_status": "unbalanced" if not all(proposed.values()) else "candidate_cycle",
        "warning": (
            "This is a game-balance diagnostic for engineered AIs, not evidence "
            "of a natural PCC cycle. No cyclic payoff bonuses are encoded."
        ),
    }


def balanced_cycle_confirmation(
    replicates: int = 12,
    hands_per_seat_order: int = 1000,
    seed: int = 23001,
    seed_stride: int = 20,
    maximum_edge_ratio: float = 3.0,
) -> dict:
    """Confirm the frozen engineered cycle on independent replicated sweeps.

    Each replicate contains all three matchups in both seat orders. Normal
    intervals summarize variation across replicate-level, seat-balanced edges;
    individual hands are not treated as independent replicates.
    """
    if replicates < 2:
        raise ValueError("at least two replicates are required")
    if hands_per_seat_order < 1:
        raise ValueError("hands_per_seat_order must be positive")
    if maximum_edge_ratio < 1:
        raise ValueError("maximum_edge_ratio must be at least one")

    edge_keys = (
        "control_over_pressure",
        "chaos_over_control",
        "pressure_over_chaos",
    )
    edge_runs = {key: [] for key in edge_keys}
    runs = []
    for replicate in range(replicates):
        run_seed = seed + replicate * seed_stride
        sweep = adaptive_pairwise_sweep(hands_per_seat_order, run_seed)
        matrix = sweep["mean_payoff_focal_policy"]
        edges = {
            "control_over_pressure": matrix["control_vs_pressure"],
            "chaos_over_control": matrix["chaos_vs_control"],
            "pressure_over_chaos": matrix["pressure_vs_chaos"],
        }
        for key, value in edges.items():
            edge_runs[key].append(value)
        runs.append({"replicate": replicate, "seed": run_seed, "edges": edges})

    edge_summary = {}
    for key, values in edge_runs.items():
        mean = statistics.mean(values)
        standard_deviation = statistics.stdev(values)
        standard_error = standard_deviation / math.sqrt(replicates)
        margin = 1.96 * standard_error
        edge_summary[key] = {
            "mean_payoff_edge": mean,
            "standard_deviation_across_replicates": standard_deviation,
            "normal_95_interval": [mean - margin, mean + margin],
            "positive_replicates": sum(value > 0 for value in values),
            "replicates": replicates,
        }

    means = [edge_summary[key]["mean_payoff_edge"] for key in edge_keys]
    edge_ratio = max(means) / min(means) if min(means) > 0 else None
    directional_confirmation = all(
        edge_summary[key]["normal_95_interval"][0] > 0 for key in edge_keys
    )
    strength_balance = edge_ratio is not None and edge_ratio <= maximum_edge_ratio
    return {
        "design": {
            "status": "frozen_candidate_confirmation",
            "replicates": replicates,
            "hands_per_seat_order": hands_per_seat_order,
            "hands_per_replicate": hands_per_seat_order * 6,
            "seed": seed,
            "seed_stride": seed_stride,
            "seat_balancing": "every matchup is run in both seat orders",
            "interval_unit": "replicate-level seat-balanced payoff edge",
            "maximum_edge_ratio": maximum_edge_ratio,
        },
        "edges": edge_summary,
        "edge_strength_ratio": edge_ratio,
        "directional_confirmation": directional_confirmation,
        "strength_balance": strength_balance,
        "balanced_cycle_confirmed": directional_confirmation and strength_balance,
        "runs": runs,
        "warning": (
            "This confirms balance of engineered synthetic policies only. It is "
            "not empirical evidence that human play follows PCC dynamics."
        ),
    }


def generate_recovery_dataset(hands_per_seat: int = 500, seed: int = 23) -> tuple[list[dict], dict]:
    """Generate seat-balanced focal-mode data against one reference policy."""
    records = []
    batches = []
    balanced = (1 / 3, 1 / 3, 1 / 3)
    next_seed = seed
    for mode in MODES:
        for focal_seat in (0, 1):
            mixtures = [balanced, balanced]
            labels = ["balanced_reference", "balanced_reference"]
            mixtures[focal_seat] = PURE_MIXTURES[mode]
            labels[focal_seat] = mode
            batch, summary = simulate_match(
                hands_per_seat, mixtures[0], mixtures[1], next_seed,
                labels[0], labels[1],
            )
            records.extend(batch)
            batches.append({"mode": mode, "focal_seat": focal_seat, **summary})
            next_seed += 1
    return records, {
        "hands_per_seat": hands_per_seat,
        "total_hands": hands_per_seat * len(MODES) * 2,
        "seed": seed,
        "design": "each pure mode in both seats versus a fixed balanced reference",
        "batches": batches,
    }


def sample_simplex(rng: random.Random, alpha: float = 0.7) -> tuple[float, float, float]:
    """Sample continuous PCC weights from a symmetric Dirichlet distribution."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    values = [rng.gammavariate(alpha, 1.0) for _ in MODES]
    total = sum(values)
    return tuple(value / total for value in values)


def generate_mixed_dataset(
    mixtures: int = 60,
    hands_per_seat: int = 100,
    seed: int = 41,
    alpha: float = 0.7,
    focal_temperature: float = 0.35,
    reference_temperature: float = 0.35,
) -> tuple[list[dict], dict]:
    """Generate continuous-mixture data with each mixture represented in both seats.

    Mixture vectors and their simulation seeds are unique to a group. Downstream
    analysis splits on ``mixture_id``, keeping both seats and all associated seeds
    together and preventing the same target vector from appearing in train/test.
    """
    if mixtures < 5:
        raise ValueError("at least five mixtures are required for grouped evaluation")
    if hands_per_seat < 1:
        raise ValueError("hands_per_seat must be positive")

    rng = random.Random(seed)
    balanced = (1 / 3, 1 / 3, 1 / 3)
    records: list[dict] = []
    groups = []
    for mixture_index in range(mixtures):
        mixture_id = f"mix-{mixture_index:04d}"
        weights = sample_simplex(rng, alpha)
        group_seeds = []
        for focal_seat in (0, 1):
            simulation_seed = seed * 10_000 + mixture_index * 2 + focal_seat
            group_seeds.append(simulation_seed)
            policy_weights = [balanced, balanced]
            policy_labels = ["balanced_reference", "balanced_reference"]
            policy_weights[focal_seat] = weights
            policy_labels[focal_seat] = mixture_id
            temperatures = [reference_temperature, reference_temperature]
            temperatures[focal_seat] = focal_temperature
            batch, _ = simulate_match(
                hands_per_seat,
                policy_weights[0],
                policy_weights[1],
                simulation_seed,
                policy_labels[0],
                policy_labels[1],
                temperatures[0],
                temperatures[1],
            )
            for record in batch:
                record["mixture_id"] = mixture_id
                record["simulation_seed"] = simulation_seed
                record["focal_seat"] = focal_seat
                record["is_focal_policy"] = record["actor"] == focal_seat
                record["target_pcc_weights"] = dict(zip(MODES, weights))
            records.extend(batch)
        groups.append({
            "mixture_id": mixture_id,
            "weights": dict(zip(MODES, weights)),
            "simulation_seeds": group_seeds,
        })
    return records, {
        "mixtures": mixtures,
        "hands_per_seat": hands_per_seat,
        "total_hands": mixtures * hands_per_seat * 2,
        "seed": seed,
        "dirichlet_alpha": alpha,
        "focal_temperature": focal_temperature,
        "reference_temperature": reference_temperature,
        "split_unit": "mixture_id (both seats and all associated seeds)",
        "groups": groups,
    }


def generate_family_dataset(
    family: str,
    mixtures: int = 60,
    hands_per_seat: int = 100,
    seed: int = 61,
    alpha: float = 0.7,
    focal_temperature: float = 0.35,
    measurement_oracle=None,
) -> tuple[list[dict], dict]:
    """Generate grouped mixtures from a selected policy implementation family."""
    families = {
        "score": PCCPolicy,
        "independent": IndependentMixturePolicy,
        "adaptive": AdaptiveMixturePolicy,
    }
    if family not in families:
        raise ValueError(f"unknown policy family {family!r}; choices={tuple(families)}")
    policy_class = families[family]
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
        groups.append({
            "mixture_id": mixture_id,
            "weights": dict(zip(MODES, weights)),
        })
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
