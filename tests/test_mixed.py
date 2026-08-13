import random
import unittest

import numpy as np

from pcc_poker.mixed import analyze_mixed_recovery, mixture_examples, run_mixed_grid
from pcc_poker.simulate import generate_mixed_dataset, sample_simplex


class MixedRecoveryTests(unittest.TestCase):
    def test_simplex_sample_is_valid_and_reproducible(self):
        left = sample_simplex(random.Random(4), alpha=0.7)
        right = sample_simplex(random.Random(4), alpha=0.7)
        self.assertEqual(left, right)
        self.assertAlmostEqual(sum(left), 1.0)
        self.assertTrue(all(value >= 0 for value in left))

    def test_dataset_balances_each_mixture_across_seats(self):
        records, summary = generate_mixed_dataset(
            mixtures=10, hands_per_seat=8, seed=5
        )
        self.assertEqual(summary["total_hands"], 160)
        examples = mixture_examples(records)
        seats_by_mixture = {}
        for example in examples:
            seats_by_mixture.setdefault(example["mixture_id"], set()).add(
                example["focal_seat"]
            )
        self.assertEqual(len(seats_by_mixture), 10)
        self.assertTrue(all(seats == {0, 1} for seats in seats_by_mixture.values()))

    def test_analysis_predicts_continuous_weights_with_grouped_split(self):
        records, _ = generate_mixed_dataset(
            mixtures=20, hands_per_seat=20, seed=11
        )
        report = analyze_mixed_recovery(records, shuffle_repetitions=3, seed=2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(
            report["prediction_target"],
            "continuous_pressure_control_chaos_weights",
        )
        self.assertGreater(report["train_mixtures"], 0)
        self.assertGreater(report["test_mixtures"], 0)
        self.assertFalse(
            set(report["train_mixture_ids"]) & set(report["test_mixture_ids"])
        )
        self.assertLessEqual(report["contextual_history_model"]["mae"], 1.0)

    def test_hidden_fields_do_not_change_predictors(self):
        records, _ = generate_mixed_dataset(
            mixtures=5, hands_per_seat=5, seed=13
        )
        before = mixture_examples(records)
        for record in records:
            record["private_rank"] = 999
            record["hidden_pcc_weights"] = {"pressure": 999}
            record["component_scores"] = {"leak": 999}
            record["action_probabilities"] = {"leak": 1.0}
        after = mixture_examples(records)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(
                left["action_features"], right["action_features"]
            )
            np.testing.assert_array_equal(
                left["contextual_features"], right["contextual_features"]
            )

    def test_replication_grid_crosses_seeds_and_temperatures(self):
        report = run_mixed_grid(
            seeds=(3, 4),
            temperatures=(0.25, 0.5),
            mixtures=10,
            hands_per_seat=5,
            shuffle_repetitions=2,
        )
        self.assertEqual(report["aggregate"]["runs"], 4)
        self.assertEqual(len(report["runs"]), 4)


if __name__ == "__main__":
    unittest.main()
