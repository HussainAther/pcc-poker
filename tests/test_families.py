import unittest
from unittest.mock import patch

from pcc_poker.engine import initial_state
from pcc_poker.families import IndependentMixturePolicy
from pcc_poker.simulate import generate_family_dataset
from pcc_poker.transfer import analyze_family_transfer, run_family_transfer_grid


class PolicyFamilyTests(unittest.TestCase):
    def test_independent_family_does_not_call_pcc_component_scores(self):
        policy = IndependentMixturePolicy((0.2, 0.5, 0.3), seed=4)
        state = initial_state([0, 1, 2, 0, 1, 2])
        with patch(
            "pcc_poker.policies.component_scores",
            side_effect=AssertionError("PCC score function was reused"),
        ):
            decision = policy.decide(state)
        self.assertIn(decision.action, state.legal_actions())
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0)

    def test_family_dataset_records_mechanism_and_balances_seats(self):
        records, summary = generate_family_dataset(
            "independent", mixtures=6, hands_per_seat=5, seed=7
        )
        self.assertEqual(summary["family"], "independent")
        focal = [record for record in records if record["is_focal_policy"]]
        self.assertTrue(focal)
        self.assertEqual({record["policy_family"] for record in focal}, {"independent"})
        seats = {}
        for record in focal:
            seats.setdefault(record["mixture_id"], set()).add(record["focal_seat"])
        self.assertTrue(all(value == {0, 1} for value in seats.values()))

    def test_adaptive_family_is_available(self):
        records, summary = generate_family_dataset(
            "adaptive", mixtures=5, hands_per_seat=2, seed=91
        )
        self.assertEqual(summary["family"], "adaptive")
        self.assertTrue(records)

    def test_transfer_uses_disjoint_mixture_groups(self):
        training, _ = generate_family_dataset(
            "score", mixtures=10, hands_per_seat=8, seed=11
        )
        transfer, _ = generate_family_dataset(
            "independent", mixtures=10, hands_per_seat=8, seed=17
        )
        report = analyze_family_transfer(
            training, transfer, shuffle_repetitions=3, seed=2
        )
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["mixture_id_overlap"])
        self.assertEqual(report["training_policy_families"], ["score"])
        self.assertEqual(report["transfer_policy_families"], ["independent"])
        self.assertLessEqual(report["contextual_history_model"]["mae"], 1.0)

    def test_bidirectional_grid_reports_each_direction(self):
        report = run_family_transfer_grid(
            seed_pairs=((2, 3),),
            mixtures=6,
            hands_per_seat=4,
            shuffle_repetitions=2,
        )
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(
            set(report["aggregate_by_direction"]),
            {"score_to_independent", "independent_to_score"},
        )


if __name__ == "__main__":
    unittest.main()
