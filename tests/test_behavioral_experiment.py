import unittest

from pcc_poker.behavioral_experiment import (
    run_behavioral_validation,
    run_opponent_adaptation_confirmation,
    run_predictive_control_confirmation,
)


class BehavioralExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_behavioral_validation(
            calibration_mixtures=5,
            calibration_hands_per_seat=2,
            evaluation_mixtures=5,
            evaluation_hands_per_seat=3,
        )

    def test_calibration_and_evaluation_are_separate(self):
        self.assertFalse(
            self.report["design"]["calibration_evaluation_overlap"]
        )

    def test_both_policy_families_are_reported(self):
        self.assertEqual(
            set(self.report["families"]), {"score", "independent"}
        )

    def test_hidden_generator_weights_are_validation_outcomes_only(self):
        self.assertFalse(self.report["generator_weights_used_as_predictors"])
        self.assertIn("cross_family_construct_result", self.report)

    def test_control_confirmation_uses_new_seeds_and_frozen_candidate(self):
        report = run_predictive_control_confirmation(
            calibration_mixtures=5,
            calibration_hands_per_seat=2,
            evaluation_mixtures=5,
            evaluation_hands_per_seat=3,
        )
        self.assertTrue(
            report["prospective_test"]["candidate_frozen_before_seed_results"]
        )
        self.assertFalse(report["prospective_test"]["old_validation_seeds_reused"])
        self.assertEqual(
            report["prospective_test"]["primary_control_measure"],
            "predictive_control",
        )

    def test_adaptation_confirmation_has_discriminant_check(self):
        report = run_opponent_adaptation_confirmation(
            calibration_mixtures=5,
            calibration_hands_per_seat=2,
            evaluation_mixtures=5,
            evaluation_hands_per_seat=3,
        )
        self.assertTrue(
            report["prospective_test"][
                "candidate_frozen_before_confirmation_seeds"
            ]
        )
        self.assertEqual(
            set(report["prospective_test"]["discriminant_check"]),
            {"score", "independent"},
        )
        for result in report["prospective_test"]["discriminant_check"].values():
            self.assertEqual(
                len(result["control_correlation_approximate_95pct_ci"]), 2
            )


if __name__ == "__main__":
    unittest.main()
