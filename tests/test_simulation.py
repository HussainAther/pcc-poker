import unittest

from pcc_poker.analyze import analyze_mode_recovery
from pcc_poker.simulate import generate_recovery_dataset, pairwise_sweep, simulate_match


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


if __name__ == "__main__": unittest.main()
