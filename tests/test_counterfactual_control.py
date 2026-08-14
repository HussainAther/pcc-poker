import tempfile
from pathlib import Path
import unittest

from pcc_poker.counterfactual_control import (
    FrozenOpponentModel,
    run_counterfactual_control_validation,
    write_counterfactual_control_validation,
)
from pcc_poker.engine import initial_state
from pcc_poker.policies import OpponentModel


class CounterfactualControlTests(unittest.TestCase):
    def test_frozen_model_ignores_evaluation_observations(self):
        source = OpponentModel()
        state = initial_state([0, 1, 0, 1, 2, 2])
        source.observe(state, "check")
        frozen = FrozenOpponentModel(source)
        before = dict(frozen.context_actions[frozen.context(state)])
        frozen.observe(state, "bet")
        self.assertEqual(before, dict(frozen.context_actions[frozen.context(state)]))

    def test_intervention_is_seat_balanced_and_uses_common_seeds(self):
        report = run_counterfactual_control_validation(
            replicates=2,
            calibration_hands_per_seat=5,
            evaluation_hands_per_seat=8,
        )
        self.assertTrue(report["design"]["seat_balanced"])
        self.assertTrue(report["design"]["common_random_numbers_within_comparison"])
        self.assertEqual(
            set(report["design"]["conditions"]),
            {"aligned", "swapped", "prior"},
        )
        self.assertEqual(len(report["condition_rows"]), 18)
        self.assertFalse(report["design"]["policies_modified"])

    def test_writer_records_prespecified_result_without_forcing_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual.json"
            report = write_counterfactual_control_validation(
                path,
                replicates=2,
                calibration_hands_per_seat=4,
                evaluation_hands_per_seat=6,
            )
            self.assertTrue(path.exists())
            self.assertEqual(
                report["counterfactual_control_confirmed"],
                all(report["prespecified_checks"].values()),
            )
            self.assertIn("control_specificity", report)


if __name__ == "__main__":
    unittest.main()
