from collections import Counter, defaultdict

import numpy as np

from pcc_poker.mixed_ood_ablation import _nested_features
from pcc_poker.mixed_ood_history_destruction import (
    _shuffled_m3_features,
    analyze_mixed_ood_history_destruction,
)


def _row(mixture, hand, decision, action, target, region=None):
    return {
        "mixture_id": mixture,
        "hand_id": hand,
        "decision_index": decision,
        "action": action,
        "round_index": decision % 2,
        "to_call": 1 if decision % 2 else 0,
        "pot": 2 + decision,
        "legal_actions": ["check", "bet", "call"],
        "focal_seat": 0,
        "is_focal_policy": True,
        "target_pcc_weights": target,
        "ood_region": region,
    }


def test_shuffle_rebuilds_only_transition_block_and_preserves_M2_exactly():
    rows = [
        _row("m1", "h1", 0, "check", {"pressure": 0.5, "control": 0.3, "chaos": 0.2}),
        _row("m1", "h1", 1, "bet", {"pressure": 0.5, "control": 0.3, "chaos": 0.2}),
        _row("m1", "h1", 2, "call", {"pressure": 0.5, "control": 0.3, "chaos": 0.2}),
        _row("m1", "h2", 0, "bet", {"pressure": 0.5, "control": 0.3, "chaos": 0.2}),
        _row("m1", "h2", 1, "check", {"pressure": 0.5, "control": 0.3, "chaos": 0.2}),
    ]
    before = _nested_features(rows)
    shuffled_m3 = _shuffled_m3_features(rows, np.random.default_rng(12))
    np.testing.assert_allclose(before["M2"], shuffled_m3[: len(before["M2"])], rtol=0.0, atol=0.0)
    assert len(shuffled_m3) == len(before["M3"])


def test_history_destruction_report_runs_on_small_fixture():
    targets = [
        {"pressure": 0.70, "control": 0.20, "chaos": 0.10},
        {"pressure": 0.20, "control": 0.70, "chaos": 0.10},
        {"pressure": 0.15, "control": 0.15, "chaos": 0.70},
    ]
    training = []
    ood = []
    actions = ["check", "bet", "call", "bet"]
    for i, target in enumerate(targets):
        for hand in range(3):
            for decision, action in enumerate(actions):
                training.append(_row(f"train-{i}", f"t{i}-{hand}", decision, action, target))
                ood.append(_row(
                    f"test-{i}", f"o{i}-{hand}", decision, actions[(decision + i) % len(actions)], target,
                    region=("pressure_heavy", "control_heavy", "chaos_heavy")[i],
                ))

    report = analyze_mixed_ood_history_destruction(training, ood, permutation_seeds=(101, 102, 103))
    assert report["status"] == "completed"
    assert report["permutation_replicates"] == 3
    assert report["preservation_contract"]["M0_M1_M2_features"] is True
    assert "gain_attenuation_fraction" in report["overall"]
