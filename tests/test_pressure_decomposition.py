import unittest

from pcc_poker.engine import State
from pcc_poker.pressure_decomposition import (
    PressureDecompositionPolicy,
    run_pressure_decomposition,
)


class PressureDecompositionTests(unittest.TestCase):
    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            PressureDecompositionPolicy((0.8, 0.1, 0.1), pressure_variant="unknown")

    def test_ablation_changes_only_pressure_distribution_path(self):
        state = State(private=(0, 2), public=None, deck=(0, 1, 1, 2), actor=0)
        full = PressureDecompositionPolicy((0.8, 0.1, 0.1), seed=1, pressure_variant="full")
        ablated = PressureDecompositionPolicy((0.8, 0.1, 0.1), seed=1, pressure_variant="no_strength_selectivity")
        self.assertNotEqual(
            full._coercive_distribution(state),
            ablated._coercive_distribution(state),
        )
        self.assertEqual(
            full._novelty_distribution(state),
            ablated._novelty_distribution(state),
        )

    def test_smoke_report_has_prespecified_structure(self):
        report = run_pressure_decomposition(
            replicates=2,
            calibration_hands_per_seat=2,
            evaluation_hands_per_seat=3,
            calibration_seed=130001,
            evaluation_seed=140001,
        )
        self.assertEqual(
            set(report["variant_summary"]),
            {"full", "no_fold_leverage", "no_strength_selectivity"},
        )
        self.assertIn("full_pressure_alignment_effect_positive", report["prespecified_checks"])
        self.assertFalse(report["design"]["control_policy_modified"])


if __name__ == "__main__":
    unittest.main()
