"""Interactive human-versus-PCC Leduc poker."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Callable

from .engine import State, apply_action, initial_state, utility, winner
from .families import AdaptiveMixturePolicy
from .policies import PURE_MIXTURES

RANK_NAMES = {0: "J", 1: "Q", 2: "K"}


def make_opponent(mode: str, seed: int) -> AdaptiveMixturePolicy:
    if mode not in ("pressure", "control", "chaos"):
        raise ValueError("opponent must be pressure, control, or chaos")
    return AdaptiveMixturePolicy(
        PURE_MIXTURES[mode], seed=seed, temperature=0.35, label=f"{mode}-ai"
    )


def _state_lines(state: State, human_seat: int) -> list[str]:
    public = "—" if state.public is None else RANK_NAMES[state.public]
    return [
        f"Your card: {RANK_NAMES[state.private[human_seat]]}   Public: {public}",
        f"Pot: {state.pot}   To call: {state.to_call}",
        "History: " + (" ".join(state.history) if state.history else "—"),
    ]


def _choose_human_action(
    state: State,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    legal = state.legal_actions()
    aliases = {str(index + 1): action for index, action in enumerate(legal)}
    while True:
        menu = "  ".join(
            f"[{index + 1}] {action}" for index, action in enumerate(legal)
        )
        output_fn(menu)
        answer = input_fn("Your action: ").strip().lower()
        action = aliases.get(answer, answer)
        if action in legal:
            return action
        output_fn(f"Choose one of: {', '.join(legal)}")


def play_session(
    hands: int,
    opponent_mode: str,
    seed: int = 701,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    auto_human: bool = False,
) -> tuple[list[dict], dict]:
    """Play a seat-alternating session and return anonymous decision telemetry."""
    if hands < 1:
        raise ValueError("hands must be positive")
    rng = random.Random(seed)
    opponent = make_opponent(opponent_mode, seed * 2 + 1)
    records = []
    totals = [0.0, 0.0]
    human_total = 0.0
    output_fn("PCC Poker — heads-up limit Leduc")
    output_fn(f"Opponent: {opponent_mode.title()} AI   Hands: {hands}")

    for hand_index in range(hands):
        human_seat = hand_index % 2
        deck = [0, 0, 1, 1, 2, 2]
        rng.shuffle(deck)
        state = initial_state(deck)
        output_fn(f"\n=== Hand {hand_index + 1}/{hands} — you are seat {human_seat} ===")
        while not state.terminal:
            actor = state.actor
            for line in _state_lines(state, human_seat):
                output_fn(line)
            if actor == human_seat:
                if auto_human:
                    action = state.legal_actions()[0]
                    output_fn(f"You: {action}")
                else:
                    action = _choose_human_action(state, input_fn, output_fn)
                opponent.opponent_model.observe(state, action)
                source = "human"
            else:
                decision = opponent.decide(state)
                action = decision.action
                output_fn(f"{opponent_mode.title()} AI: {action}")
                source = "ai"
            records.append({
                "session_seed": seed,
                "hand_index": hand_index,
                "decision_index": len(records),
                "human_seat": human_seat,
                "actor": actor,
                "source": source,
                "opponent_mode": opponent_mode,
                "round_index": state.round_index,
                "public_rank": state.public,
                "private_rank": state.private[actor],
                "pot": state.pot,
                "to_call": state.to_call,
                "legal_actions": list(state.legal_actions()),
                "history": list(state.history),
                "action": action,
            })
            state = apply_action(state, action)

        payoffs = (utility(state, 0), utility(state, 1))
        totals[0] += payoffs[0]
        totals[1] += payoffs[1]
        human_total += payoffs[human_seat]
        for record in records:
            if record["hand_index"] == hand_index:
                record["terminal_payoff"] = payoffs[record["actor"]]
        result = winner(state)
        if result is None:
            result_text = "Tie"
        elif result == human_seat:
            result_text = "You win"
        else:
            result_text = f"{opponent_mode.title()} AI wins"
        output_fn(
            f"Result: {result_text} | cards "
            f"{RANK_NAMES[state.private[human_seat]]} vs "
            f"{RANK_NAMES[state.private[1 - human_seat]]} | "
            f"payoff {payoffs[human_seat]:+.1f}"
        )

    summary = {
        "hands": hands,
        "seed": seed,
        "opponent_mode": opponent_mode,
        "seat0_total": totals[0],
        "seat1_total": totals[1],
        "human_total": human_total,
        "ai_total": -human_total,
        "decisions": len(records),
        "note": "personal gameplay/debugging; not human-subject evidence",
    }
    output_fn(
        f"\nSession complete. You {human_total:+.1f} | "
        f"{opponent_mode.title()} AI {-human_total:+.1f}"
    )
    return records, summary


def write_session(path: str | Path, records: list[dict], summary: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    target.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
