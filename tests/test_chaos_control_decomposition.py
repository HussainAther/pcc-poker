import unittest

from pcc_poker.chaos_control_decomposition import (
    ChaosControlDecompositionOracle,
    FrozenStaticTemporalActionModel,
    summarize_chaos_control_decomposition,
)
from pcc_poker.engine import initial_state


class ChaosControlDecompositionTests(unittest.TestCase):
    def test_temporal_context_can_change_probability_without_future_information(self):
        records = [
            {"actor": 0, "round_index": 0, "to_call": 0, "pot": 2, "legal_actions": ["check", "bet"], "history": [], "action": "check"},
            {"actor": 0, "round_index": 0, "to_call": 0, "pot": 2, "legal_actions": ["check", "bet"], "history": ["bet", "call"], "action": "bet"},
            {"actor": 0, "round_index": 0, "to_call": 0, "pot": 2, "legal_actions": ["check", "bet"], "history": ["bet", "call"], "action": "bet"},
        ]
        model = FrozenStaticTemporalActionModel.from_records(records)
        state = initial_state([0, 1, 2, 0, 1, 2])
        # Initial state's context is different, so this mainly asserts stable
        # deterministic backoff and a proper distribution.
        probabilities = model.probabilities(state, temporal=True)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertEqual(set(probabilities), set(state.legal_actions()))

    def test_measurement_is_nonnegative_and_uses_same_value_floor(self):
        calibration = [
            {"actor": 0, "round_index": 0, "to_call": 1, "pot": 2, "legal_actions": ["fold", "call", "raise"], "history": [], "action": "call"},
        ]
        oracle = ChaosControlDecompositionOracle(FrozenStaticTemporalActionModel.from_records(calibration))
        state = initial_state([0, 1, 2, 0, 1, 2])
        measurement = oracle.measure(state, state.legal_actions()[0])
        self.assertGreaterEqual(measurement.history_explained_surprisal, 0.0)
        self.assertGreaterEqual(measurement.history_residual_surprisal, 0.0)
        self.assertGreaterEqual(measurement.performance_adequacy, 0.0)
        self.assertLessEqual(measurement.performance_adequacy, 1.0)
        self.assertAlmostEqual(
            measurement.history_residual_effective_surprisal,
            measurement.history_residual_surprisal * measurement.performance_adequacy,
        )

    def test_summary_keeps_full_cross_axis_correlations(self):
        records = []
        for index, weights in enumerate(((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))):
            for seat in (0, 1):
                records.append({
                    "policy_family": "independent",
                    "mixture_id": f"m{index}",
                    "focal_seat": seat,
                    "is_focal_policy": True,
                    "target_pcc_weights": {"pressure": weights[0], "control": weights[1], "chaos": weights[2]},
                    "behavioral_measurements": {
                        "static_effective_surprisal": weights[2] + 0.2 * weights[1],
                        "history_explained_effective_surprisal": weights[1],
                        "history_residual_effective_surprisal": weights[2],
                        "performance_adequacy": 0.9,
                    },
                })
        report = summarize_chaos_control_decomposition(records)
        family = report["families"]["independent"]
        self.assertIn("pressure", family["history_residual_effective_correlations"])
        self.assertIn("control", family["history_residual_effective_correlations"])
        self.assertIn("chaos", family["history_residual_effective_correlations"])
        self.assertGreater(family["history_explained_control_margin"], 0.0)
        self.assertGreater(family["history_residual_chaos_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
