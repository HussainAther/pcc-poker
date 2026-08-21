import json
import unittest
from pathlib import Path

from pcc_poker.score_control_intervention import CONTEXT_RESPONSE_GAIN
from pcc_poker.score_control_value_decomposition import LOW_EFFICIENCY_THRESHOLD
from pcc_poker.score_control_value_intervention import (
    ValueAwareContextualScorePolicy,
    run_score_control_value_intervention,
)


class ScoreControlValueInterventionTests(unittest.TestCase):
    def test_protocol_reuses_frozen_gain_and_efficiency_threshold(self):
        self.assertEqual(CONTEXT_RESPONSE_GAIN, 3.35)
        self.assertEqual(LOW_EFFICIENCY_THRESHOLD, 0.80)

    def test_small_run_is_well_formed_and_human_safe(self):
        report = run_score_control_value_intervention(
            oracle_mixtures=3,
            oracle_hands_per_seat=5,
            calibration_mixtures=3,
            calibration_hands_per_seat=5,
            evaluation_mixtures=4,
            evaluation_hands_per_seat=8,
        )
        self.assertIn(report["status"], {"confirmed", "partial", "failed"})
        self.assertFalse(report["intervention"]["human_data_accessed"])
        self.assertFalse(report["intervention"]["frozen_v0.8_human_panel_modified"])
        self.assertEqual(
            report["intervention"]["minimum_aggressive_counterfactual_efficiency"],
            0.80,
        )

    def test_frozen_default_result_matches_committed_status(self):
        path = Path(__file__).resolve().parents[1] / "validation" / "score-control-value-intervention.json"
        report = json.loads(path.read_text())
        self.assertEqual(
            report["control_structural_recovery_confirmed"],
            all(report["prespecified_checks"].values()),
        )
        self.assertEqual(
            report["intervention"]["minimum_aggressive_counterfactual_efficiency"],
            0.80,
        )
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["control_structural_recovery_confirmed"])
        self.assertFalse(report["stage_replication"]["information_uptake"])
        self.assertTrue(report["stage_replication"]["context_alignment"])
        self.assertFalse(report["stage_replication"]["value_sensitive_intervention"])
