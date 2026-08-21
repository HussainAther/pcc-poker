import unittest

from pcc_poker.control_structural_recovery import (
    run_control_structural_recovery,
    summarize_control_structural_recovery,
)
from pcc_poker.contextual_control_observable import FrozenAlignedYokedHistoryModel
from pcc_poker.behavioral import CounterfactualOracle, PublicActionModel
from pcc_poker.simulate import generate_family_dataset


class ControlStructuralRecoveryTests(unittest.TestCase):
    def test_small_run_is_two_family_and_preserves_yoke_margins(self):
        report = run_control_structural_recovery(
            calibration_mixtures=4,
            calibration_hands_per_seat=4,
            evaluation_mixtures=6,
            evaluation_hands_per_seat=6,
        )
        self.assertEqual(set(report["families"]), {"score", "adaptive"})
        self.assertTrue(report["prespecified_checks"]["matched_yoke_margins_preserved"])
        self.assertFalse(report["design"]["human_data_accessed"])
        self.assertFalse(report["design"]["frozen_v0.8_human_panel_modified"])

    def test_summary_uses_no_hidden_generator_diagnostics_as_measurement_inputs(self):
        calibration = []
        for family, seed in (("score", 91), ("adaptive", 99)):
            records, _ = generate_family_dataset(family, 4, 4, seed)
            calibration.extend(records)
        history = FrozenAlignedYokedHistoryModel.from_records(calibration, seed=107)
        oracle = CounterfactualOracle(PublicActionModel.from_records(calibration))
        evaluation = []
        for family, seed in (("score", 111), ("adaptive", 119)):
            records, _ = generate_family_dataset(family, 5, 5, seed, measurement_oracle=oracle)
            for record in records:
                record["component_scores"] = {"secret": {"bet": 999}}
                record["action_probabilities"] = {"bet": 1.0}
                record["terminal_payoff"] = 999
            evaluation.extend(records)
        report = summarize_control_structural_recovery(evaluation, history)
        self.assertEqual(report["trajectory_groups"], 20)
        self.assertIn("value_sensitive_intervention", report["stage_replication"])


if __name__ == "__main__":
    unittest.main()
