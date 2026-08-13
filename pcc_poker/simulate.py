"""Simulation and experiment logging."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .engine import apply_action, initial_state, utility
from .policies import MODES, PCCPolicy, PURE_MIXTURES


def play_hand(policy0: PCCPolicy, policy1: PCCPolicy, deck: list[int], hand_id: str) -> tuple[list[dict], tuple[float, float]]:
    state = initial_state(deck); policies = (policy0, policy1); records = []
    while not state.terminal:
        actor = state.actor; decision = policies[actor].decide(state)
        records.append({
            "hand_id": hand_id, "decision_index": len(records), "actor": actor,
            "round_index": state.round_index, "public_rank": state.public,
            "private_rank": state.private[actor], "pot": state.pot,
            "to_call": state.to_call, "legal_actions": list(state.legal_actions()),
            "history": list(state.history), "action": decision.action,
            "action_probabilities": decision.probabilities,
            "component_scores": decision.component_scores,
            "hidden_pcc_weights": decision.weights,
            "policy_label": policies[actor].label, "showdown_equity": decision.equity,
        })
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
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    policy0 = PCCPolicy(mixture0, seed=seed * 2 + 1, label=label0)
    policy1 = PCCPolicy(mixture1, seed=seed * 2 + 2, label=label1)
    records = []; totals = [0.0, 0.0]
    for index in range(hands):
        deck = [0, 0, 1, 1, 2, 2]; rng.shuffle(deck)
        hand_records, payoffs = play_hand(policy0, policy1, deck, f"match-{seed}-hand-{index}")
        records.extend(hand_records); totals[0] += payoffs[0]; totals[1] += payoffs[1]
    return records, {
        "hands": hands, "seed": seed, "policy0": policy0.label, "policy1": policy1.label,
        "mixture0": list(policy0.weights), "mixture1": list(policy1.weights),
        "mean_payoff0": totals[0] / hands, "mean_payoff1": totals[1] / hands,
    }


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
