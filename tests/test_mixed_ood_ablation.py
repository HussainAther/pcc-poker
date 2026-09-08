import json
from pathlib import Path

import numpy as np

from pcc_poker.mixed import _features
from pcc_poker.mixed_ood_ablation import (
    FORBIDDEN_PREDICTOR_FIELDS,
    MODEL_ORDER,
    _nested_features,
    analyze_mixed_ood_ablation,
)


def _row(hand, decision, action, **overrides):
    row = {
        "hand_id": hand,
        "decision_index": decision,
        "action": action,
        "round_index": 0,
        "to_call": 0,
        "pot": 2,
        "legal_actions": ["check", "bet"],
        "focal_seat": 0,
        "private_rank": 3,
        "showdown_equity": 0.99,
        "terminal_payoff": 999,
        "component_scores": {"pressure": {"bet": 100}},
    }
    row.update(overrides)
    return row


def test_nested_features_are_strictly_nested_and_full_matches_existing_contextual_vector():
    rows = [
        _row("h1", 0, "check"),
        _row("h1", 1, "bet", round_index=1, pot=4),
        _row("h2", 0, "call", to_call=2, legal_actions=["fold", "call", "raise"]),
    ]
    nested = _nested_features(rows)
    action, contextual = _features(rows)

    assert tuple(nested) == MODEL_ORDER
    assert len(nested["M0"]) < len(nested["M1"]) < len(nested["M2"]) < len(nested["M3"])
    np.testing.assert_allclose(nested["M0"], action)

    # Existing contextual vector is action + betting context + transitions + summaries;
    # M3 contains the same values with summaries before transitions. Reorder to compare.
    m3 = nested["M3"]
    action_and_context = m3[:25]
    summaries = m3[25:31]
    transitions = m3[31:56]
    reconstructed_existing = np.concatenate([action_and_context, transitions, summaries])
    np.testing.assert_allclose(reconstructed_existing, contextual)


def test_private_or_outcome_fields_do_not_change_nested_features():
    base = [
        _row("h1", 0, "check"),
        _row("h2", 0, "bet", round_index=1, pot=5, to_call=1),
    ]
    changed = [dict(row) for row in base]
    changed[0].update(private_rank=1, showdown_equity=0.01, terminal_payoff=-1000)
    changed[1].update(private_rank=2, showdown_equity=0.50, terminal_payoff=5000)

    first = _nested_features(base)
    second = _nested_features(changed)
    for model in MODEL_ORDER:
        np.testing.assert_allclose(first[model], second[model])

    assert {"private_rank", "showdown_equity", "terminal_payoff"} <= FORBIDDEN_PREDICTOR_FIELDS


def test_ablation_runs_on_repository_mixed_datasets():
    root = Path(__file__).resolve().parents[1]
    training = [json.loads(line) for line in (root / "outputs/mixed-recovery-data.jsonl").read_text().splitlines()]
    ood = [json.loads(line) for line in (root / "outputs/mixed-ood-data.jsonl").read_text().splitlines()]

    report = analyze_mixed_ood_ablation(training, ood)

    assert report["status"] == "completed"
    assert report["train_examples"] == 120
    assert report["ood_test_examples"] == 200
    assert report["prespecified_checks"]["balanced_region_retained_without_tuning"]
    assert set(report["overall"]) == set(MODEL_ORDER)
    assert set(report["by_region"]) == {
        "pressure_heavy", "control_heavy", "chaos_heavy", "balanced", "boundary"
    }
