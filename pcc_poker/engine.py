"""Minimal, auditable heads-up limit Leduc poker engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

RANKS = (0, 1, 2)  # J, Q, K
ACTIONS = ("check", "bet", "fold", "call", "raise")


@dataclass(frozen=True)
class State:
    private: tuple[int, int]
    public: int | None
    deck: tuple[int, ...]
    round_index: int = 0
    actor: int = 0
    contributions: tuple[int, int] = (1, 1)
    round_contributions: tuple[int, int] = (0, 0)
    checks: int = 0
    raises: int = 0
    history: tuple[str, ...] = ()
    terminal: bool = False
    folded: int | None = None

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    @property
    def bet_size(self) -> int:
        return 2 if self.round_index == 0 else 4

    @property
    def to_call(self) -> int:
        return max(self.round_contributions) - self.round_contributions[self.actor]

    def legal_actions(self) -> tuple[str, ...]:
        if self.terminal:
            return ()
        if self.to_call == 0:
            return ("check", "bet")
        actions = ["fold", "call"]
        if self.raises < 2:
            actions.append("raise")
        return tuple(actions)

    def information_key(self, player: int) -> str:
        public = "-" if self.public is None else str(self.public)
        return f"r{self.round_index}|h{self.private[player]}|b{public}|a{self.actor}|tc{self.to_call}|{'/'.join(self.history)}"


def initial_state(deck: Iterable[int]) -> State:
    cards = tuple(deck)
    if len(cards) != 6 or sorted(cards) != [0, 0, 1, 1, 2, 2]:
        raise ValueError("Leduc deck must contain two copies of each of three ranks")
    return State(private=(cards[0], cards[1]), public=None, deck=cards[2:])


def _add(values: tuple[int, int], player: int, amount: int) -> tuple[int, int]:
    result = list(values); result[player] += amount
    return tuple(result)


def _advance_round(state: State) -> State:
    if state.round_index == 1:
        return replace(state, terminal=True)
    return replace(
        state,
        public=state.deck[0], deck=state.deck[1:], round_index=1, actor=0,
        round_contributions=(0, 0), checks=0, raises=0,
        history=state.history + ("/",),
    )


def apply_action(state: State, action: str) -> State:
    if action not in state.legal_actions():
        raise ValueError(f"Illegal action {action!r}; legal={state.legal_actions()}")
    actor = state.actor; other = 1 - actor
    history = state.history + (action,)
    if action == "fold":
        return replace(state, history=history, terminal=True, folded=actor)
    if action == "check":
        checked = replace(state, history=history, actor=other, checks=state.checks + 1)
        return _advance_round(checked) if checked.checks == 2 else checked
    if action == "bet":
        amount = state.bet_size
        return replace(
            state, history=history, actor=other, checks=0,
            contributions=_add(state.contributions, actor, amount),
            round_contributions=_add(state.round_contributions, actor, amount),
        )
    if action == "call":
        called = replace(
            state, history=history,
            contributions=_add(state.contributions, actor, state.to_call),
            round_contributions=_add(state.round_contributions, actor, state.to_call),
        )
        return _advance_round(called)
    amount = state.to_call + state.bet_size
    return replace(
        state, history=history, actor=other, checks=0, raises=state.raises + 1,
        contributions=_add(state.contributions, actor, amount),
        round_contributions=_add(state.round_contributions, actor, amount),
    )


def hand_strength(private: int, public: int) -> tuple[int, int]:
    return (1, private) if private == public else (0, private)


def winner(state: State) -> int | None:
    if not state.terminal:
        raise ValueError("Winner is defined only for terminal states")
    if state.folded is not None:
        return 1 - state.folded
    left = hand_strength(state.private[0], state.public)
    right = hand_strength(state.private[1], state.public)
    return 0 if left > right else 1 if right > left else None


def utility(state: State, player: int) -> float:
    result = winner(state)
    if result is None:
        return state.pot / 2 - state.contributions[player]
    return state.pot - state.contributions[player] if result == player else -state.contributions[player]


def equity(state: State, player: int) -> float:
    """Exact showdown equity over cards consistent with the player's information."""
    own = state.private[player]
    known = [own] + ([] if state.public is None else [state.public])
    remaining = [rank for rank in RANKS for _ in range(2)]
    for card in known:
        remaining.remove(card)
    wins = ties = total = 0
    for opponent in remaining:
        residual = remaining.copy(); residual.remove(opponent)
        publics = [state.public] if state.public is not None else residual
        for public in publics:
            own_strength = hand_strength(own, public)
            opponent_strength = hand_strength(opponent, public)
            wins += own_strength > opponent_strength
            ties += own_strength == opponent_strength
            total += 1
    return (wins + 0.5 * ties) / total


def all_deals() -> list[tuple[int, ...]]:
    """Unique ordered six-card deck arrangements."""
    return _unique_deals()


def _unique_deals() -> list[tuple[int, ...]]:
    from itertools import permutations
    return sorted(set(permutations((0, 0, 1, 1, 2, 2))))
