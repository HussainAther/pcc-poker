import copy
import unittest

import numpy as np

from pcc_poker.simulate import generate_family_dataset
from pcc_poker.temporal_control import (
    run_temporal_control_validation,
    temporal_examples,
)


class TemporalControlTests(unittest.TestCase):
    def test_predictors_ignore_generator_diagnostics_and_outcomes(self):
        records, _ = generate_family_dataset(
            "adaptive", mixtures=5, hands_per_seat=3, seed=801
        )
        altered = copy.deepcopy(records)
        for record in altered:
            record["hidden_pcc_weights"] = {"pressure": 99, "control": -1}
            record["component_scores"] = {"secret": {"bet": 1000}}
            record["action_probabilities"] = {"bet": 1.0}
            record["terminal_payoff"] = 999
        original_examples = temporal_examples(records)
        altered_examples = temporal_examples(altered)
        self.assertEqual(len(original_examples), len(altered_examples))
        for original, changed in zip(original_examples, altered_examples):
            np.testing.assert_array_equal(original["static"], changed["static"])
            np.testing.assert_array_equal(original["temporal"], changed["temporal"])
            self.assertEqual(original["target"], changed["target"])

    def test_temporal_features_are_prior_only(self):
        records, _ = generate_family_dataset(
            "adaptive", mixtures=5, hands_per_seat=3, seed=803
        )
        examples = temporal_examples(records)
        first_by_simulation = {}
        for example in examples:
            first_by_simulation.setdefault(example["simulation_seed"], example)
        self.assertTrue(first_by_simulation)
        lengths = {len(example["temporal"]) for example in examples}
        self.assertEqual(len(lengths), 1)
        self.assertTrue(all(np.isfinite(example["temporal"]).all() for example in examples))

    def test_validation_uses_disjoint_mixture_groups(self):
        report = run_temporal_control_validation(
            training_mixtures=6,
            evaluation_mixtures=6,
            hands_per_seat=8,
            training_seed=811,
            evaluation_seed=821,
            shuffle_repetitions=2,
        )
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["mixture_id_overlap"])
        self.assertEqual(report["training_mixtures"], 6)
        self.assertEqual(report["evaluation_mixtures"], 6)
        self.assertIn("adaptive_log_loss_below_static", report["prespecified_checks"])
        self.assertIn("control", report["trajectory_control_score"]["weight_correlations"])


if __name__ == "__main__":
    unittest.main()
