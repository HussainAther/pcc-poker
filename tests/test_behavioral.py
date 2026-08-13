import unittest

from pcc_poker.behavioral import (
    CounterfactualOracle,
    PublicActionModel,
    information_states,
)
from pcc_poker.engine import initial_state


class BehavioralMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.oracle = CounterfactualOracle(PublicActionModel())

    def test_information_states_are_normalized(self):
        state = initial_state([0, 1, 2, 0, 1, 2])
        possibilities = information_states(state, observer=0)
        self.assertAlmostEqual(sum(weight for _, weight in possibilities), 1.0)
        self.assertTrue(all(concrete.private[0] == 0 for concrete, _ in possibilities))

    def test_measurements_are_bounded_and_regret_is_nonnegative(self):
        state = initial_state([0, 1, 2, 0, 1, 2])
        measurement = self.oracle.measure(state, "bet")
        self.assertGreaterEqual(measurement.regret, 0.0)
        self.assertGreater(measurement.control_efficiency, 0.0)
        self.assertLessEqual(measurement.control_efficiency, 1.0)
        self.assertGreaterEqual(measurement.pressure_index, 0.0)
        self.assertLessEqual(measurement.pressure_index, 1.0)
        self.assertLessEqual(
            measurement.effective_surprisal, measurement.action_surprisal
        )

    def test_best_counterfactual_action_has_full_control_efficiency(self):
        state = initial_state([0, 1, 2, 0, 1, 2])
        values = self.oracle.action_values(state)
        best_action = max(values, key=values.get)
        measurement = self.oracle.measure(state, best_action)
        self.assertAlmostEqual(measurement.regret, 0.0)
        self.assertAlmostEqual(measurement.control_efficiency, 1.0)

    def test_check_is_not_mislabeled_as_pressure(self):
        state = initial_state([0, 1, 2, 0, 1, 2])
        measurement = self.oracle.measure(state, "check")
        self.assertEqual(measurement.pressure_index, 0.0)
        self.assertEqual(measurement.commitment_ratio, 0.0)

    def test_actual_hidden_opponent_card_does_not_change_measurement(self):
        left = initial_state([0, 1, 2, 0, 1, 2])
        right = initial_state([0, 2, 1, 0, 1, 2])
        left_measurement = self.oracle.measure(left, "bet")
        right_measurement = self.oracle.measure(right, "bet")
        self.assertEqual(left_measurement, right_measurement)

    def test_public_model_ignores_hidden_fields(self):
        base = {
            "actor": 0,
            "round_index": 0,
            "to_call": 0,
            "pot": 2,
            "legal_actions": ["check", "bet"],
            "action": "bet",
        }
        altered = {
            **base,
            "private_rank": 999,
            "hidden_pcc_weights": {"pressure": 999},
            "component_scores": {"leak": 999},
        }
        model = PublicActionModel.from_records([base, altered])
        probabilities = model.probabilities(initial_state([0, 1, 2, 0, 1, 2]))
        self.assertGreater(probabilities["bet"], probabilities["check"])


if __name__ == "__main__":
    unittest.main()
