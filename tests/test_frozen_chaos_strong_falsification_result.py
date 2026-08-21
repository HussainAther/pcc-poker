import json
from pathlib import Path


def test_frozen_poker_chaos_strong_falsification_result():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "validation/chaos-strong-falsification.json").read_text())
    assert report["poker_chaos_strong_falsification_confirmed"] is True
    assert report["exploiter_calibration"]["selected"]["label"] == "adaptive-pressure-cold"
    assert report["exploiter_calibration"]["selection_used_only_predictable_baseline"] is True
    assert report["design"]["human_data_used"] is False
    assert report["families"]["score"]["all_checks_pass"] is True
    assert report["families"]["independent"]["all_checks_pass"] is True
    assert report["baselines"]["uniform_random"]["mean_normalized_policy_entropy"] > report["families"]["score"]["metrics"]["mean_normalized_policy_entropy"]
    assert report["baselines"]["uniform_random"]["mean_payoff_vs_neutral"] < report["families"]["score"]["metrics"]["mean_payoff_vs_neutral"]
