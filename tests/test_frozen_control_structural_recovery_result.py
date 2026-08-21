import json
from pathlib import Path


def test_frozen_post_v08_control_structural_recovery_result_is_retained():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "validation/control-structural-recovery.json").read_text())
    assert report["status"] == "partial"
    assert report["control_structural_recovery_confirmed"] is False
    assert report["families"]["adaptive"]["all_three_stages_recovered"] is True
    assert report["families"]["score"]["all_three_stages_recovered"] is False
    assert not any(report["stage_replication"].values())
    assert report["margin_checks"]["static_context_action_margins_preserved"] is True
    assert report["margin_checks"]["global_action_margins_preserved"] is True
    assert report["design"]["human_data_accessed"] is False
    assert report["design"]["frozen_v0.8_human_panel_modified"] is False
