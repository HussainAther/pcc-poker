"""Safe ingestion helpers for future HandHQ/PHH human-data work.

This module is intentionally testable without any human dataset.  It parses the
PHHS-style record structure from text, immediately minimizes source metadata,
and emits decision rows that contain only information available at the decision
point.

The repository's policy is that confirmatory human-data analysis must not begin
until the appropriate institutional determination is in place.  Tests therefore
use synthetic/mock records only.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import hmac
import re
from typing import Any, Iterable, Mapping, Sequence


_BLOCK_RE = re.compile(r"^\s*\[(\d+)\]\s*$")
_PLAYER_RE = re.compile(r"^p(\d+)$")

# Metadata that must never survive into sanitized analytical records.
FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "players",
        "table",
        "hand",
        "time",
        "day",
        "month",
        "year",
        "time_zone_abbreviation",
        "currency_symbol",
    }
)

# Outcome is useful for some analyses, but is deliberately kept out of
# decision-time feature rows to prevent post-outcome leakage.
OUTCOME_FIELDS = frozenset({"winnings"})


@dataclass(frozen=True)
class ActionEvent:
    """A normalized PHH action/event."""

    kind: str
    actor: int | None = None
    amount: float | None = None
    public_cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class SanitizedHand:
    """A privacy-minimized hand record safe for downstream feature building."""

    study_hand_id: str
    variant: str
    venue: str | None
    antes: tuple[float, ...]
    blinds_or_straddles: tuple[float, ...]
    min_bet: float
    starting_stacks: tuple[float, ...]
    seats: tuple[int, ...]
    player_ids: tuple[str, ...]
    actions: tuple[ActionEvent, ...]
    outcome: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DecisionRow:
    """One decision with strictly pre-decision public-history features."""

    study_hand_id: str
    decision_index: int
    actor_position: int
    actor_seat: int
    actor_player_id: str
    action: str
    amount: float | None
    public_history: tuple[tuple[str, int | None, float | None], ...]
    pot_contribution_proxy: tuple[float, ...]
    starting_stacks: tuple[float, ...]
    blinds_or_straddles: tuple[float, ...]
    min_bet: float
    venue: str | None
    public_board: tuple[str, ...]


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        # PHHS examples may contain unquoted values such as 00:00:01.
        return raw


def _parse_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _BLOCK_RE.match(stripped)
        if match:
            current = {"__record_number__": int(match.group(1))}
            blocks.append(current)
            continue
        if current is None:
            raise ValueError(f"content before first record header at line {line_number}")
        if "=" not in line:
            raise ValueError(f"invalid assignment at line {line_number}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key at line {line_number}")
        current[key] = _parse_scalar(raw_value)
    return blocks


def _float_tuple(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains a non-numeric value") from exc


def _int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains a non-integer value") from exc


def _study_player_id(raw_identifier: str, key: bytes) -> str:
    if not key:
        raise ValueError("pseudonymization_key must be non-empty")
    digest = hmac.new(key, raw_identifier.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"P{digest[:16]}"


def parse_action(action: str) -> ActionEvent:
    """Normalize one PHH action while discarding private-card contents.

    Supported betting codes follow the Poker Hand History convention used by
    HandHQ-derived PHH data: ``f`` (fold), ``cc`` (check/call), and ``cbr``
    (complete/bet/raise). Deal events are retained only when they reveal public
    board cards; private-hole-card contents are discarded.
    """

    tokens = action.split()
    if not tokens:
        raise ValueError("empty action")

    if tokens[0] == "d":
        if len(tokens) < 2:
            raise ValueError(f"malformed deal action: {action!r}")
        deal_kind = tokens[1]
        if deal_kind == "dh":
            return ActionEvent(kind="private_deal")
        if deal_kind == "db":
            cards = tuple(token for token in tokens[2:] if not _PLAYER_RE.match(token))
            return ActionEvent(kind="public_deal", public_cards=cards)
        return ActionEvent(kind="deal")

    player_match = _PLAYER_RE.match(tokens[0])
    if player_match is None or len(tokens) < 2:
        raise ValueError(f"malformed player action: {action!r}")
    actor = int(player_match.group(1)) - 1
    code = tokens[1]
    if code == "f":
        return ActionEvent(kind="fold", actor=actor)
    if code == "cc":
        return ActionEvent(kind="check_call", actor=actor)
    if code == "cbr":
        if len(tokens) < 3:
            raise ValueError(f"missing amount in cbr action: {action!r}")
        return ActionEvent(kind="bet_raise", actor=actor, amount=float(tokens[2]))
    raise ValueError(f"unsupported player action code {code!r}")


def sanitize_record(
    raw: Mapping[str, Any],
    *,
    study_index: int,
    pseudonymization_key: bytes,
    retain_outcome: bool = False,
) -> SanitizedHand:
    """Convert one parsed source record into the minimized analytical schema."""

    required = {
        "variant",
        "antes",
        "blinds_or_straddles",
        "min_bet",
        "starting_stacks",
        "actions",
        "seats",
        "players",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    players = raw["players"]
    if not isinstance(players, (list, tuple)):
        raise ValueError("players must be a sequence")
    seats = _int_tuple(raw["seats"], "seats")
    stacks = _float_tuple(raw["starting_stacks"], "starting_stacks")
    antes = _float_tuple(raw["antes"], "antes")
    blinds = _float_tuple(raw["blinds_or_straddles"], "blinds_or_straddles")
    n_players = len(players)
    for field, seq in (
        ("seats", seats),
        ("starting_stacks", stacks),
        ("antes", antes),
        ("blinds_or_straddles", blinds),
    ):
        if len(seq) != n_players:
            raise ValueError(f"{field} length does not match players")
    if len(set(seats)) != len(seats):
        raise ValueError("seat numbers must be unique within a hand")

    raw_actions = raw["actions"]
    if not isinstance(raw_actions, (list, tuple)):
        raise ValueError("actions must be a sequence")
    actions = tuple(parse_action(str(action)) for action in raw_actions)
    for event in actions:
        if event.actor is not None and not 0 <= event.actor < n_players:
            raise ValueError("action references player outside record")

    player_ids = tuple(
        _study_player_id(str(identifier), pseudonymization_key) for identifier in players
    )

    outcome: tuple[float, ...] | None = None
    if retain_outcome and "winnings" in raw:
        outcome = _float_tuple(raw["winnings"], "winnings")
        if len(outcome) != n_players:
            raise ValueError("winnings length does not match players")

    return SanitizedHand(
        study_hand_id=f"H{study_index:08d}",
        variant=str(raw["variant"]),
        venue=str(raw["venue"]) if "venue" in raw else None,
        antes=antes,
        blinds_or_straddles=blinds,
        min_bet=float(raw["min_bet"]),
        starting_stacks=stacks,
        seats=seats,
        player_ids=player_ids,
        actions=actions,
        outcome=outcome,
    )


def ingest_phhs_text(
    text: str,
    *,
    pseudonymization_key: bytes,
    retain_outcome: bool = False,
) -> tuple[SanitizedHand, ...]:
    """Parse PHHS text and immediately return privacy-minimized hands."""

    blocks = _parse_blocks(text)
    return tuple(
        sanitize_record(
            block,
            study_index=index,
            pseudonymization_key=pseudonymization_key,
            retain_outcome=retain_outcome,
        )
        for index, block in enumerate(blocks, start=1)
    )


def decision_rows(hand: SanitizedHand) -> tuple[DecisionRow, ...]:
    """Build pre-decision rows without private cards, outcomes, or future actions."""

    history: list[tuple[str, int | None, float | None]] = []
    public_board: list[str] = []
    contributions = [float(value) for value in hand.antes]
    for index, blind in enumerate(hand.blinds_or_straddles):
        contributions[index] += blind

    rows: list[DecisionRow] = []
    decision_index = 0
    for event in hand.actions:
        if event.kind == "private_deal":
            # Deliberately absent from public history.
            continue
        if event.kind == "public_deal":
            public_board.extend(event.public_cards)
            history.append(("public_deal", None, None))
            continue
        if event.actor is None:
            continue

        actor = event.actor
        rows.append(
            DecisionRow(
                study_hand_id=hand.study_hand_id,
                decision_index=decision_index,
                actor_position=actor,
                actor_seat=hand.seats[actor],
                actor_player_id=hand.player_ids[actor],
                action=event.kind,
                amount=event.amount,
                public_history=tuple(history),
                pot_contribution_proxy=tuple(contributions),
                starting_stacks=hand.starting_stacks,
                blinds_or_straddles=hand.blinds_or_straddles,
                min_bet=hand.min_bet,
                venue=hand.venue,
                public_board=tuple(public_board),
            )
        )
        decision_index += 1

        history.append((event.kind, actor, event.amount))
        if event.kind == "bet_raise" and event.amount is not None:
            # PHH cbr amounts are treated conservatively as the actor's stated
            # contribution level for a simple public-state proxy.  Detailed
            # pot accounting belongs in the later game-state reconstruction layer.
            contributions[actor] = max(contributions[actor], event.amount)

    return tuple(rows)


def assert_no_forbidden_values(
    source_record: Mapping[str, Any],
    sanitized_objects: Iterable[Any],
) -> None:
    """Test helper: fail if forbidden source values survive in object reprs."""

    serialized = "\n".join(repr(obj) for obj in sanitized_objects)
    value_fields = {"players", "table", "hand", "time", "time_zone_abbreviation", "currency_symbol"}
    for field in value_fields:
        value = source_record.get(field)
        if value is None:
            continue
        values: Sequence[Any] = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            text = str(item)
            if text and text in serialized:
                raise AssertionError(f"forbidden source value from {field!r} survived")
