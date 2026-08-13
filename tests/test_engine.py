import unittest

from pcc_poker.engine import apply_action, equity, initial_state, utility


class EngineTests(unittest.TestCase):
    def test_check_check_advances_round(self):
        state=initial_state([0,1,2,0,1,2]);state=apply_action(state,"check");state=apply_action(state,"check")
        self.assertEqual(state.round_index,1);self.assertEqual(state.public,2);self.assertFalse(state.terminal)

    def test_bet_call_advances_and_conserves_zero_sum(self):
        state=initial_state([0,1,2,0,1,2]);state=apply_action(state,"bet");state=apply_action(state,"call");state=apply_action(state,"check");state=apply_action(state,"check")
        self.assertTrue(state.terminal);self.assertAlmostEqual(utility(state,0)+utility(state,1),0)

    def test_fold_ends_hand(self):
        state=initial_state([0,1,2,0,1,2]);state=apply_action(state,"bet");state=apply_action(state,"fold")
        self.assertTrue(state.terminal);self.assertGreater(utility(state,0),0)

    def test_equity_is_bounded(self):
        state=initial_state([2,0,1,0,1,2]);self.assertGreaterEqual(equity(state,0),0);self.assertLessEqual(equity(state,0),1)


if __name__ == "__main__": unittest.main()
