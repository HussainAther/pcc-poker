import unittest

from pcc_poker.handhq import (
    FORBIDDEN_SOURCE_FIELDS,
    assert_no_forbidden_values,
    decision_rows,
    ingest_phhs_text,
    parse_action,
)


# Entirely synthetic fixture: identifiers, table, hand number, amounts, and
# actions are invented for unit testing and do not come from the human dataset.
MOCK_PHHS = """
[1]
variant = 'NT'
ante_trimming_status = false
antes = [0, 0, 0]
blinds_or_straddles = [0.25, 0.50, 0]
min_bet = 0.50
starting_stacks = [40.00, 55.00, 60.00]
actions = ['d dh p1 AsKd', 'd dh p2 QhQs', 'd dh p3 7c2d', 'p3 f', 'p1 cbr 1.50', 'p2 cc', 'd db 2c 8d Jh', 'p2 cc', 'p1 cbr 3.00', 'p2 f']
venue = 'Synthetic Poker Lab'
time = 12:34:56
day = 14
month = 8
year = 2026
hand = 999000111
table = 'MOCK TABLE ONLY'
seats = [2, 4, 6]
players = ['mock-alice-source-id', 'mock-bob-source-id', 'mock-carol-source-id']
winnings = [2.75, -1.50, -1.25]
currency_symbol = '$'
time_zone_abbreviation = 'ET'
"""

KEY = b"unit-test-only-secret-key"


class HandHQIngestionTests(unittest.TestCase):
    def test_action_parser_normalizes_supported_codes(self):
        self.assertEqual(parse_action("p2 f").kind, "fold")
        self.assertEqual(parse_action("p2 cc").kind, "check_call")
        raise_event = parse_action("p2 cbr 4.25")
        self.assertEqual(raise_event.kind, "bet_raise")
        self.assertEqual(raise_event.actor, 1)
        self.assertEqual(raise_event.amount, 4.25)

    def test_private_card_contents_are_discarded(self):
        event = parse_action("d dh p1 AsKd")
        self.assertEqual(event.kind, "private_deal")
        self.assertEqual(event.public_cards, ())
        self.assertNotIn("As", repr(event))
        self.assertNotIn("Kd", repr(event))

    def test_seat_reconstruction_and_pseudonyms(self):
        (hand,) = ingest_phhs_text(MOCK_PHHS, pseudonymization_key=KEY)
        self.assertEqual(hand.seats, (2, 4, 6))
        self.assertEqual(len(hand.player_ids), 3)
        self.assertTrue(all(identifier.startswith("P") for identifier in hand.player_ids))
        self.assertNotIn("mock-alice-source-id", repr(hand))
        rows = decision_rows(hand)
        # First player action is p3 f -> zero-based actor position 2 -> seat 6.
        self.assertEqual(rows[0].actor_position, 2)
        self.assertEqual(rows[0].actor_seat, 6)
        self.assertEqual(rows[0].actor_player_id, hand.player_ids[2])

    def test_source_metadata_is_not_in_sanitized_schema(self):
        (hand,) = ingest_phhs_text(MOCK_PHHS, pseudonymization_key=KEY)
        for field in FORBIDDEN_SOURCE_FIELDS:
            self.assertFalse(hasattr(hand, field), field)
        self.assertNotIn("MOCK TABLE ONLY", repr(hand))
        self.assertNotIn("999000111", repr(hand))
        self.assertNotIn("12:34:56", repr(hand))

    def test_outcome_is_opt_in_and_never_in_decision_rows(self):
        (without_outcome,) = ingest_phhs_text(MOCK_PHHS, pseudonymization_key=KEY)
        self.assertIsNone(without_outcome.outcome)
        (with_outcome,) = ingest_phhs_text(
            MOCK_PHHS, pseudonymization_key=KEY, retain_outcome=True
        )
        self.assertEqual(with_outcome.outcome, (2.75, -1.50, -1.25))
        self.assertNotIn("2.75", repr(decision_rows(with_outcome)))

    def test_decision_rows_use_only_prior_public_history(self):
        (hand,) = ingest_phhs_text(MOCK_PHHS, pseudonymization_key=KEY)
        rows = decision_rows(hand)
        self.assertEqual(rows[0].public_history, ())
        # The second decision can see the first fold, but not its own action or future actions.
        self.assertEqual(rows[1].public_history, (("fold", 2, None),))
        history_text = repr(rows[1].public_history)
        self.assertNotIn("check_call", history_text)
        self.assertNotIn("3.0", history_text)
        # Public cards become available only after the public deal.
        self.assertEqual(rows[2].public_board, ())
        self.assertEqual(rows[3].public_board, ("2c", "8d", "Jh"))
        # Private mock hole cards never enter any feature row.
        rows_text = repr(rows)
        for private_card_text in ("AsKd", "QhQs", "7c2d"):
            self.assertNotIn(private_card_text, rows_text)

    def test_forbidden_source_values_do_not_survive(self):
        source_record = {
            "players": ["mock-alice-source-id", "mock-bob-source-id", "mock-carol-source-id"],
            "table": "MOCK TABLE ONLY",
            "hand": 999000111,
            "time": "12:34:56",
            "day": 14,
            "month": 8,
            "year": 2026,
            "time_zone_abbreviation": "ET",
            "currency_symbol": "$",
        }
        (hand,) = ingest_phhs_text(MOCK_PHHS, pseudonymization_key=KEY)
        assert_no_forbidden_values(source_record, (hand, *decision_rows(hand)))

    def test_mismatched_player_vectors_are_rejected(self):
        bad = MOCK_PHHS.replace("seats = [2, 4, 6]", "seats = [2, 4]")
        with self.assertRaises(ValueError):
            ingest_phhs_text(bad, pseudonymization_key=KEY)


if __name__ == "__main__":
    unittest.main()
