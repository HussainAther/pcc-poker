import unittest

from pcc_poker.behavioral import PublicActionModel
from pcc_poker.effective_chaos_validation import (
    IndependentChaosOracle,
    run_effective_chaos_validation,
)
from pcc_poker.engine import initial_state


class EffectiveChaosValidationTests(unittest.TestCase):
    def test_oracle_returns_independent_floor_fields(self):
        state = initial_state([0, 1, 0, 1, 2, 2])
        model = PublicActionModel(smoothing=1.0)
        oracle = IndependentChaosOracle(model)
        result = oracle.measure(state, state.legal_actions()[0]).as_dict()
        self.assertIn("raw_normalized_surprisal", result)
        self.assertIn("performance_adequacy", result)
        self.assertIn("independent_effective_surprisal", result)
        self.assertLessEqual(result["independent_effective_surprisal"], result["raw_normalized_surprisal"] + 1e-12)

    def test_small_fresh_seed_run_has_full_falsification_report(self):
        report = run_effective_chaos_validation(
            calibration_mixtures=3,
            calibration_hands_per_seat=4,
            evaluation_mixtures=5,
            evaluation_hands_per_seat=6,
            score_calibration_seed=2401,
            independent_calibration_seed=2409,
            score_evaluation_seed=2601,
            independent_evaluation_seed=2609,
            shuffle_seed=2901,
        )
        self.assertEqual(set(report["families"]), {"score", "independent"})
        for result in report["families"].values():
            self.assertEqual(set(result["effective_surprisal_weight_correlations"]), {"pressure", "control", "chaos"})
            self.assertIn("shuffled_chaos_weight_correlation", result)
            self.assertIn("value_floor_margin_gain", result)
        self.assertTrue(report["design"]["candidate_frozen_before_evaluation"])
        self.assertFalse(report["design"]["human_data_used"])


if __name__ == "__main__":
    unittest.main()
