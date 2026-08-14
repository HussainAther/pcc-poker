import unittest

from pcc_poker.engine import apply_action, initial_state
from pcc_poker.families import AdaptiveMixturePolicy
from pcc_poker.play import play_session


class AdaptiveGameTests(unittest.TestCase):
    def test_pressure_commits_more_with_stronger_information(self):
        policy = AdaptiveMixturePolicy((1.0, 0.0, 0.0), seed=1)
        weak = initial_state([0, 1, 2, 0, 1, 2])
        strong = initial_state([2, 1, 0, 0, 1, 2])
        self.assertGreater(
            policy._coercive_distribution(strong)["bet"],
            policy._coercive_distribution(weak)["bet"],
        )

    def test_chaos_keeps_every_legal_branch_possible(self):
        policy = AdaptiveMixturePolicy((0.0, 0.0, 1.0), seed=1)
        state = initial_state([0, 1, 2, 0, 1, 2])
        probabilities = policy._novelty_distribution(state)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertTrue(all(value > 0 for value in probabilities.values()))

    def test_control_times_aggression_to_observed_fold_rate(self):
        policy = AdaptiveMixturePolicy((0.0, 1.0, 0.0), seed=1)
        state = initial_state([0, 1, 2, 0, 1, 2])
        before = policy._adaptive_control_distribution(state)["bet"]
        response_state = apply_action(state, "bet")
        for _ in range(8):
            policy.opponent_model.observe(response_state, "fold")
        after = policy._adaptive_control_distribution(state)["bet"]
        self.assertGreater(after, before)

    def test_auto_session_is_reproducible_and_zero_sum(self):
        left_records, left_summary = play_session(
            2, "control", seed=17, auto_human=True, output_fn=lambda _: None
        )
        right_records, right_summary = play_session(
            2, "control", seed=17, auto_human=True, output_fn=lambda _: None
        )
        self.assertEqual(left_records, right_records)
        self.assertEqual(left_summary, right_summary)
        self.assertAlmostEqual(
            left_summary["human_total"] + left_summary["ai_total"], 0.0
        )
        self.assertTrue(left_records)
        self.assertTrue(all("terminal_payoff" in row for row in left_records))


if __name__ == "__main__":
    unittest.main()
