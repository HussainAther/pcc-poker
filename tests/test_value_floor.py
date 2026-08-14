import unittest

from pcc_poker.engine import initial_state
from pcc_poker.value_floor import (
    UniformContinuationValueModel,
    effective_chaos_candidate,
    measure_synthetic_effective_chaos,
    performance_floor,
)


class ValueFloorTests(unittest.TestCase):
    def test_uniform_reference_is_independent_and_returns_all_legal_actions(self):
        state = initial_state([0, 1, 0, 1, 2, 2])
        values = UniformContinuationValueModel().action_values(state)
        self.assertEqual(set(values), set(state.legal_actions()))
        self.assertTrue(all(isinstance(value, float) for value in values.values()))

    def test_best_action_gets_full_adequacy(self):
        floor = performance_floor({'safe': 1.0, 'bad': -1.0}, 'safe', tolerance=1.0)
        self.assertEqual(floor.regret, 0.0)
        self.assertEqual(floor.adequacy, 1.0)

    def test_value_destroying_action_fails_floor(self):
        floor = performance_floor({'safe': 1.0, 'bad': -1.0}, 'bad', tolerance=1.0)
        self.assertEqual(floor.adequacy, 0.0)

    def test_random_but_bad_is_not_rewarded_as_chaos(self):
        score = effective_chaos_candidate(
            1.0, {'strong': 1.0, 'bad': -1.0}, 'bad', tolerance=1.0
        )
        self.assertEqual(score.effective_surprisal, 0.0)

    def test_strong_deterministic_is_low_effective_chaos(self):
        score = effective_chaos_candidate(
            0.05, {'strong': 1.0, 'bad': -1.0}, 'strong', tolerance=1.0
        )
        self.assertAlmostEqual(score.adequacy, 1.0)
        self.assertAlmostEqual(score.effective_surprisal, 0.05)

    def test_strong_mixed_is_high_effective_chaos(self):
        deterministic = effective_chaos_candidate(
            0.05, {'a': 1.0, 'b': 0.9}, 'a', tolerance=0.5
        )
        mixed = effective_chaos_candidate(
            0.90, {'a': 1.0, 'b': 0.9}, 'b', tolerance=0.5
        )
        self.assertGreater(mixed.adequacy, 0.0)
        self.assertGreater(mixed.effective_surprisal, deterministic.effective_surprisal)

    def test_synthetic_measurement_uses_information_set_value_model(self):
        state = initial_state([0, 1, 0, 1, 2, 2])
        action = max(UniformContinuationValueModel().action_values(state), key=UniformContinuationValueModel().action_values(state).get)
        score = measure_synthetic_effective_chaos(state, action, 0.8)
        self.assertEqual(score.regret, 0.0)
        self.assertAlmostEqual(score.effective_surprisal, 0.8)

    def test_invalid_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            performance_floor({'a': 1.0}, 'a', tolerance=0.0)


if __name__ == '__main__':
    unittest.main()
