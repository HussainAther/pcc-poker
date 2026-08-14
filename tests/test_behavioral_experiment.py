import unittest

from pcc_poker.behavioral_experiment import run_behavioral_validation


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


if __name__ == "__main__":
    unittest.main()
