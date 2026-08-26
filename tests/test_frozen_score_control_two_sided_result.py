import json
from pathlib import Path


def test_frozen_two_sided_result_is_partial_and_not_overclaimed():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'validation/score-control-two-sided-intervention.json').read_text())
    assert report['control_structural_recovery_confirmed'] is False
    assert report['status'] == 'partial'
    assert report['families']['score']['stages']['information_uptake']['stage_recovered'] is True
    assert report['families']['score']['stages']['context_alignment']['stage_recovered'] is True
    assert report['families']['score']['stages']['value_sensitive_intervention']['stage_recovered'] is False
    assert report['families']['adaptive']['all_three_stages_recovered'] is True
    assert report['intervention']['human_data_accessed'] is False
    assert report['intervention']['frozen_v0.8_human_panel_modified'] is False
