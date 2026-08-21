import json
import unittest
from pathlib import Path

from pcc_poker.score_control_value_decomposition import run_score_control_value_decomposition


class ScoreControlValueDecompositionTests(unittest.TestCase):
    def test_small_run_is_well_formed_and_read_only(self):
        report = run_score_control_value_decomposition(
            calibration_mixtures=3,
            calibration_hands_per_seat=5,
            evaluation_mixtures=4,
            evaluation_hands_per_seat=8,
        )
        self.assertIn(report["status"], {"confirmed", "partial"})
        self.assertFalse(report["policy_modified"])
        self.assertFalse(report["human_data_accessed"])
        self.assertEqual(set(report["families"]), {"score", "adaptive"})

    def test_frozen_default_result_records_value_guardrail_bottleneck(self):
        path = Path(__file__).resolve().parents[1] / "validation" / "score-control-value-decomposition.json"
        report = json.loads(path.read_text())
        self.assertTrue(report["value_guardrail_bottleneck_supported"])
        self.assertTrue(all(report["prespecified_checks"].values()))
        score = report["families"]["score"]
        self.assertLessEqual(score["weight_correlations"]["control_efficiency"]["control"], -0.20)
        self.assertGreaterEqual(score["weight_correlations"]["regret"]["control"], 0.20)
        self.assertGreaterEqual(score["positive_gain_to_value_product_control_attenuation"], 0.05)
