import unittest

from pcc_poker.analyze import analyze_mode_recovery
from pcc_poker.simulate import (
    adaptive_pairwise_sweep,
    balanced_cycle_confirmation,
    generate_recovery_dataset,
    pairwise_sweep,
    simulate_match,
)


class SimulationTests(unittest.TestCase):
    def test_match_is_reproducible(self):
        left,summary1=simulate_match(20,seed=5);right,summary2=simulate_match(20,seed=5)
        self.assertEqual(left,right);self.assertEqual(summary1,summary2)

    def test_pairwise_sweep_reports_without_forcing_cycle(self):
        report=pairwise_sweep(10,seed=2);self.assertIn("complete_cycle_observed",report);self.assertEqual(len(report["mean_payoff_focal_policy"]),9);self.assertIn("both seats",report["design"])

    def test_seat_balanced_recovery_dataset(self):
        records, summary = generate_recovery_dataset(12, seed=7)
        self.assertEqual(summary["total_hands"], 72)
        report = analyze_mode_recovery(records)
        self.assertEqual(set(report["labels"]), {"pressure", "control", "chaos"})
        self.assertGreaterEqual(report["accuracy"], 0.0)
        self.assertLessEqual(report["accuracy"], 1.0)

    def test_adaptive_sweep_is_seat_balanced_and_antisymmetric(self):
        report = adaptive_pairwise_sweep(hands_per_seat_order=10, seed=5)
        matrix = report["mean_payoff_focal_policy"]
        self.assertAlmostEqual(
            matrix["pressure_vs_control"], -matrix["control_vs_pressure"]
        )
        self.assertIn(report["balance_status"], {"unbalanced", "candidate_cycle"})

    def test_adaptive_sweep_accepts_temperature_and_purity(self):
        report = adaptive_pairwise_sweep(
            hands_per_seat_order=10,
            seed=7,
            temperature=0.5,
            mode_purity=0.9,
        )
        self.assertEqual(report["temperature"], 0.5)
        self.assertEqual(report["mode_purity"], 0.9)

    def test_balanced_cycle_confirmation_uses_replicate_level_edges(self):
        report = balanced_cycle_confirmation(
            replicates=2, hands_per_seat_order=10, seed=501
        )
        self.assertEqual(
            set(report["edges"]),
            {
                "control_over_pressure",
                "chaos_over_control",
                "pressure_over_chaos",
            },
        )
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(report["design"]["hands_per_replicate"], 60)


if __name__ == "__main__": unittest.main()
