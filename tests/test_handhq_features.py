import unittest

from pcc_poker.handhq import ingest_phhs_text
from pcc_poker.handhq_features import (
    assert_feature_boundary,
    modeling_features,
    reconstruct_public_states,
)


MOCK_PHHS = """
[1]
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [0.25, 0.50, 0]
min_bet = 0.50
starting_stacks = [40.00, 55.00, 60.00]
actions = ['d dh p1 AsKd', 'd dh p2 QhQs', 'd dh p3 7c2d', 'p3 f', 'p1 cbr 1.50', 'p2 cc', 'd db 2c 8d Jh', 'p2 cc', 'p1 cbr 3.00', 'p2 cc', 'd db 9s', 'p2 cc', 'p1 cbr 6.00', 'p2 f']
venue = 'Synthetic Poker Lab'
seats = [2, 4, 6]
players = ['mock-alice-source-id', 'mock-bob-source-id', 'mock-carol-source-id']
winnings = [5.00, -4.75, -0.25]
"""
KEY = b"unit-test-only-secret-key"


class HandHQFeatureTests(unittest.TestCase):
    def setUp(self):
        (self.hand,) = ingest_phhs_text(MOCK_PHHS, pseudonymization_key=KEY, retain_outcome=True)
        self.states = reconstruct_public_states(self.hand)

    def test_preflop_pot_to_call_and_effective_stack(self):
        # First decision is p3 fold. Pot contains only the 0.25/0.50 blinds.
        s0 = self.states[0]
        self.assertEqual(s0.street, "preflop")
        self.assertAlmostEqual(s0.pot_size, 0.75)
        self.assertAlmostEqual(s0.current_bet, 0.50)
        self.assertAlmostEqual(s0.to_call, 0.50)
        self.assertAlmostEqual(s0.actor_stack_remaining, 60.00)
        self.assertAlmostEqual(s0.effective_stack, 54.50)
        self.assertEqual(s0.active_players, 3)
        self.assertEqual(s0.legal_actions, ("fold", "check_call", "bet_raise"))

    def test_state_is_emitted_before_focal_action(self):
        # p1's raise to 1.50 is not already included in its own state.
        raise_state = self.states[1]
        self.assertAlmostEqual(raise_state.pot_size, 0.75)
        self.assertAlmostEqual(raise_state.actor_street_contribution, 0.25)
        self.assertAlmostEqual(raise_state.to_call, 0.25)
        self.assertNotIn(("bet_raise", 0, 1.5), raise_state.prior_action_sequence)

        # p2 acts next and now sees p1's raise and the enlarged pot.
        call_state = self.states[2]
        self.assertAlmostEqual(call_state.pot_size, 2.00)
        self.assertAlmostEqual(call_state.current_bet, 1.50)
        self.assertAlmostEqual(call_state.to_call, 1.00)
        self.assertIn(("bet_raise", 0, 1.5), call_state.prior_action_sequence)

    def test_public_deal_advances_street_and_resets_commitments(self):
        flop_state = self.states[3]
        self.assertEqual(flop_state.street, "flop")
        self.assertEqual(flop_state.public_board, ("2c", "8d", "Jh"))
        self.assertAlmostEqual(flop_state.pot_size, 3.00)
        self.assertAlmostEqual(flop_state.current_bet, 0.0)
        self.assertAlmostEqual(flop_state.to_call, 0.0)
        self.assertEqual(flop_state.street_action_sequence, ())
        self.assertEqual(flop_state.raises_this_street, 0)
        self.assertEqual(flop_state.legal_actions, ("check_call", "bet_raise"))

        turn_state = self.states[6]
        self.assertEqual(turn_state.street, "turn")
        self.assertEqual(turn_state.public_board, ("2c", "8d", "Jh", "9s"))
        self.assertAlmostEqual(turn_state.current_bet, 0.0)
        self.assertEqual(turn_state.street_action_sequence, ())

    def test_pot_and_stack_accounting_across_streets(self):
        # After p1 bets 3 on flop, p2 faces a 3-chip call with pot 6.
        flop_call = self.states[5]
        self.assertEqual(flop_call.street, "flop")
        self.assertAlmostEqual(flop_call.pot_size, 6.00)
        self.assertAlmostEqual(flop_call.to_call, 3.00)
        self.assertAlmostEqual(flop_call.actor_stack_remaining, 53.50)
        self.assertEqual(flop_call.raises_this_street, 1)

        # Turn bet is again a street-total target, not cumulative across streets.
        turn_fold = self.states[8]
        self.assertEqual(turn_fold.street, "turn")
        self.assertAlmostEqual(turn_fold.pot_size, 15.00)
        self.assertAlmostEqual(turn_fold.to_call, 6.00)
        self.assertEqual(turn_fold.observed_action, "fold")

    def test_modeling_features_exclude_labels_identity_and_outcome(self):
        assert_feature_boundary(self.states)
        features = modeling_features(self.states[5])
        forbidden = {
            "study_hand_id", "actor_player_id", "actor_seat", "observed_action",
            "observed_amount", "outcome", "winnings", "venue", "public_board",
        }
        self.assertTrue(forbidden.isdisjoint(features))
        text = repr(features)
        self.assertNotIn(self.states[5].actor_player_id, text)
        self.assertNotIn(self.hand.study_hand_id, text)
        self.assertNotIn("5.0", repr({k: v for k, v in features.items() if k in ("outcome", "winnings")}))

    def test_future_actions_and_private_cards_do_not_leak(self):
        first = self.states[0]
        history = repr(first.prior_action_sequence)
        self.assertNotIn("bet_raise", history)
        features_text = repr(modeling_features(first))
        for card in ("As", "Kd", "Qh", "Qs", "7c", "2d"):
            self.assertNotIn(card, features_text)
        self.assertNotIn("6.0", features_text)

    def test_folded_player_cannot_act_again(self):
        bad = MOCK_PHHS.replace("'p2 f']", "'p2 f', 'p3 cc']")
        (hand,) = ingest_phhs_text(bad, pseudonymization_key=KEY)
        with self.assertRaises(ValueError):
            reconstruct_public_states(hand)


if __name__ == "__main__":
    unittest.main()
