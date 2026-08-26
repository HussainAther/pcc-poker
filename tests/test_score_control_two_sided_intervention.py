from pcc_poker.score_control_two_sided_intervention import (
    PASSIVE_OPTIONALITY_GAIN,
    TwoSidedContextualScorePolicy,
    run_score_control_two_sided_intervention,
)
from pcc_poker.score_control_intervention import CONTEXT_RESPONSE_GAIN


def test_passive_gain_is_frozen_from_matched_state_result():
    assert CONTEXT_RESPONSE_GAIN == 3.35
    assert PASSIVE_OPTIONALITY_GAIN == 1.24


def test_two_sided_intervention_small_run_keeps_gate_shape():
    report = run_score_control_two_sided_intervention(
        calibration_mixtures=2,
        calibration_hands_per_seat=2,
        evaluation_mixtures=2,
        evaluation_hands_per_seat=2,
    )
    assert set(report["families"]) == {"score", "adaptive"}
    assert set(report["stage_replication"]) == {
        "information_uptake",
        "context_alignment",
        "value_sensitive_intervention",
    }
    assert report["intervention"]["human_data_accessed"] is False
    assert report["intervention"]["frozen_v0.8_human_panel_modified"] is False
