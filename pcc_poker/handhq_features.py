"""Public game-state reconstruction for sanitized HandHQ/PHH records.

This layer consumes :class:`~pcc_poker.handhq.SanitizedHand` objects only. It
never accepts raw HandHQ records, which keeps identifiers and excluded source
metadata outside the feature-building boundary.

The reconstructed state is intentionally conservative. PHH ``cbr X`` events
are interpreted as setting the actor's total contribution on the current
street to ``X``. ``cc`` contributes the amount required to match the current
street maximum, or zero when checking. Public board deals advance the street
and reset street commitments. All feature rows describe the state *before* the
focal action and never include outcomes or future actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .handhq import ActionEvent, SanitizedHand


_STREET_NAMES = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True)
class PublicDecisionState:
    """One sanitized decision and its pre-decision public game state."""

    study_hand_id: str
    decision_index: int
    actor_position: int
    actor_seat: int
    actor_player_id: str
    observed_action: str
    observed_amount: float | None

    street: str
    street_index: int
    public_board: tuple[str, ...]
    prior_action_sequence: tuple[tuple[str, int | None, float | None], ...]
    street_action_sequence: tuple[tuple[str, int | None, float | None], ...]

    pot_size: float
    current_bet: float
    to_call: float
    actor_street_contribution: float
    actor_total_contribution: float
    actor_stack_remaining: float
    effective_stack: float
    active_players: int
    raises_this_street: int

    legal_actions: tuple[str, ...]


def _street_from_board(board_size: int) -> tuple[str, int]:
    if board_size == 0:
        return "preflop", 0
    if board_size == 3:
        return "flop", 1
    if board_size == 4:
        return "turn", 2
    if board_size == 5:
        return "river", 3
    raise ValueError(f"unsupported public board size {board_size}; expected 0, 3, 4, or 5")


def _legal_actions(*, to_call: float, stack_remaining: float) -> tuple[str, ...]:
    actions: list[str] = []
    if to_call > 1e-12:
        actions.append("fold")
    actions.append("check_call")
    if stack_remaining > to_call + 1e-12:
        actions.append("bet_raise")
    return tuple(actions)


def _effective_stack(actor: int, remaining: list[float], active: list[bool]) -> float:
    actor_remaining = max(0.0, remaining[actor])
    opponent_remaining = [
        max(0.0, stack)
        for index, stack in enumerate(remaining)
        if index != actor and active[index]
    ]
    if not opponent_remaining:
        return 0.0
    # In a multiway pot, this is the largest one-opponent amount the actor can
    # still contest. It reduces to the standard heads-up effective stack.
    return min(actor_remaining, max(opponent_remaining))


def _contribute(
    actor: int,
    amount: float,
    *,
    street_contrib: list[float],
    total_contrib: list[float],
    remaining: list[float],
) -> float:
    amount = max(0.0, min(float(amount), remaining[actor]))
    street_contrib[actor] += amount
    total_contrib[actor] += amount
    remaining[actor] -= amount
    if abs(remaining[actor]) < 1e-12:
        remaining[actor] = 0.0
    return amount


def reconstruct_public_states(hand: SanitizedHand) -> tuple[PublicDecisionState, ...]:
    """Reconstruct pre-decision public states from a sanitized hand.

    Private cards are already absent from ``SanitizedHand`` action payloads.
    This function additionally omits outcomes and never includes the focal or
    future action in either history sequence.
    """

    n_players = len(hand.player_ids)
    if n_players < 2:
        raise ValueError("at least two players are required")

    remaining = [float(stack) for stack in hand.starting_stacks]
    total_contrib = [0.0] * n_players
    street_contrib = [0.0] * n_players
    active = [True] * n_players

    # Antes enter the pot but are not treated as live street commitments.
    for actor, ante in enumerate(hand.antes):
        _contribute(
            actor,
            ante,
            street_contrib=[0.0] * n_players,  # deliberately discard street effect
            total_contrib=total_contrib,
            remaining=remaining,
        )

    # Blinds/straddles are live preflop commitments.
    for actor, blind in enumerate(hand.blinds_or_straddles):
        _contribute(
            actor,
            blind,
            street_contrib=street_contrib,
            total_contrib=total_contrib,
            remaining=remaining,
        )

    public_board: list[str] = []
    full_history: list[tuple[str, int | None, float | None]] = []
    street_history: list[tuple[str, int | None, float | None]] = []
    raises_this_street = 0
    states: list[PublicDecisionState] = []
    decision_index = 0

    for event in hand.actions:
        if event.kind == "private_deal":
            continue

        if event.kind == "public_deal":
            public_board.extend(event.public_cards)
            _street_from_board(len(public_board))  # validate canonical Hold'em board sizes
            full_history.append(("public_deal", None, None))
            street_history = []
            street_contrib = [0.0] * n_players
            raises_this_street = 0
            continue

        if event.actor is None:
            continue
        actor = event.actor
        if not active[actor]:
            raise ValueError("folded player acts later in the same hand")

        street, street_index = _street_from_board(len(public_board))
        current_bet = max(
            (street_contrib[i] for i in range(n_players) if active[i]),
            default=0.0,
        )
        to_call = max(0.0, current_bet - street_contrib[actor])
        stack_remaining = max(0.0, remaining[actor])

        states.append(
            PublicDecisionState(
                study_hand_id=hand.study_hand_id,
                decision_index=decision_index,
                actor_position=actor,
                actor_seat=hand.seats[actor],
                actor_player_id=hand.player_ids[actor],
                observed_action=event.kind,
                observed_amount=event.amount,
                street=street,
                street_index=street_index,
                public_board=tuple(public_board),
                prior_action_sequence=tuple(full_history),
                street_action_sequence=tuple(street_history),
                pot_size=float(sum(total_contrib)),
                current_bet=float(current_bet),
                to_call=float(to_call),
                actor_street_contribution=float(street_contrib[actor]),
                actor_total_contribution=float(total_contrib[actor]),
                actor_stack_remaining=float(stack_remaining),
                effective_stack=float(_effective_stack(actor, remaining, active)),
                active_players=sum(active),
                raises_this_street=raises_this_street,
                legal_actions=_legal_actions(to_call=to_call, stack_remaining=stack_remaining),
            )
        )
        decision_index += 1

        # Apply the focal action only after its feature state has been emitted.
        if event.kind == "fold":
            active[actor] = False
        elif event.kind == "check_call":
            _contribute(
                actor,
                to_call,
                street_contrib=street_contrib,
                total_contrib=total_contrib,
                remaining=remaining,
            )
        elif event.kind == "bet_raise":
            if event.amount is None:
                raise ValueError("bet_raise event lacks amount")
            target = float(event.amount)
            if target + 1e-12 < street_contrib[actor]:
                raise ValueError("cbr target is below actor's existing street contribution")
            delta = target - street_contrib[actor]
            if delta > remaining[actor] + 1e-12:
                # Treat oversized targets as an all-in rather than inventing chips.
                delta = remaining[actor]
            before_bet = current_bet
            _contribute(
                actor,
                delta,
                street_contrib=street_contrib,
                total_contrib=total_contrib,
                remaining=remaining,
            )
            if street_contrib[actor] > before_bet + 1e-12:
                raises_this_street += 1
        else:
            raise ValueError(f"unsupported decision event {event.kind!r}")

        history_event = (event.kind, actor, event.amount)
        full_history.append(history_event)
        street_history.append(history_event)

    return tuple(states)


def modeling_features(state: PublicDecisionState) -> dict[str, object]:
    """Return a label-free feature dictionary suitable for later PCC models.

    ``observed_action``, ``observed_amount``, the study/player identifiers, and
    any outcome are deliberately excluded. This separates predictor features
    from the action label and prevents identity memorization.
    """

    return {
        "street": state.street,
        "street_index": state.street_index,
        "board_size": len(state.public_board),
        "pot_size": state.pot_size,
        "current_bet": state.current_bet,
        "to_call": state.to_call,
        "actor_street_contribution": state.actor_street_contribution,
        "actor_total_contribution": state.actor_total_contribution,
        "actor_stack_remaining": state.actor_stack_remaining,
        "effective_stack": state.effective_stack,
        "active_players": state.active_players,
        "raises_this_street": state.raises_this_street,
        "can_fold": "fold" in state.legal_actions,
        "can_check_call": "check_call" in state.legal_actions,
        "can_bet_raise": "bet_raise" in state.legal_actions,
        # Public action history is retained as structural information but has
        # no player IDs beyond within-hand position indices.
        "prior_action_sequence": state.prior_action_sequence,
        "street_action_sequence": state.street_action_sequence,
    }


def assert_feature_boundary(states: Iterable[PublicDecisionState]) -> None:
    """Fail if modeling features accidentally include labels or identifiers."""

    forbidden_keys = {
        "study_hand_id",
        "actor_player_id",
        "actor_seat",
        "observed_action",
        "observed_amount",
        "outcome",
        "winnings",
        "venue",
        "public_board",
    }
    for state in states:
        features = modeling_features(state)
        leaked = forbidden_keys.intersection(features)
        if leaked:
            raise AssertionError(f"forbidden modeling feature keys: {sorted(leaked)}")
        serialized = repr(features)
        if state.actor_player_id in serialized or state.study_hand_id in serialized:
            raise AssertionError("study/player identifier leaked into modeling features")
