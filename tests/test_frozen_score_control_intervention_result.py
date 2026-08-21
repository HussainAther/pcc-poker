import json
from pathlib import Path


def test_frozen_score_control_intervention_result():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'validation/score-control-intervention.json').read_text())
    assert report['status'] == 'partial'
    assert report['control_structural_recovery_confirmed'] is False
    assert report['stage_replication'] == {
        'information_uptake': True,
        'context_alignment': True,
        'value_sensitive_intervention': False,
    }
    score = report['families']['score']['stages']
    assert score['information_uptake']['stage_recovered'] is True
    assert score['context_alignment']['stage_recovered'] is True
    assert score['value_sensitive_intervention']['stage_recovered'] is False
    assert abs(score['value_sensitive_intervention']['control_correlation'] - 0.1323798) < 1e-5
    assert report['intervention']['human_data_accessed'] is False
    assert report['intervention']['frozen_v0.8_human_panel_modified'] is False
