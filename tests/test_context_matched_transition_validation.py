import numpy as np

from pcc_poker.context_matched_transition_validation import (
    _aggregate_seed_results,
    generate_control_heavy_evaluation,
    mixed_pair_block,
)
from pcc_poker.history_structure_probe import MIXED_PAIR_CATEGORIES


def test_mixed_pair_block_assigns_transition_to_second_action_context():
    rows = [
        {"hand_id": 1, "decision_index": 0, "action": "bet", "round_index": 0, "to_call": 0},
        {"hand_id": 1, "decision_index": 1, "action": "call", "round_index": 0, "to_call": 1},
        {"hand_id": 1, "decision_index": 2, "action": "check", "round_index": 1, "to_call": 0},
    ]
    block = mixed_pair_block(rows, facing_bet=True)
    expected = np.zeros(len(MIXED_PAIR_CATEGORIES))
    expected[MIXED_PAIR_CATEGORIES.index(("bet", "call"))] = 1.0
    assert np.allclose(block, expected)


def test_generate_control_heavy_evaluation_is_fresh_seed_and_both_seats():
    records, design = generate_control_heavy_evaluation(seed=1101, mixtures=1, hands_per_seat=1)
    assert design["seed"] == 1101
    assert design["region"] == "control_heavy"
    assert design["total_hands"] == 2
    assert {row["focal_seat"] for row in records} == {0, 1}
    assert all(row["ood_region"] == "control_heavy" for row in records)
    assert all(row["mixture_id"].startswith("fresh-1101-control_heavy-") for row in records)


def test_aggregate_requires_positive_mean_and_seed_fraction():
    seed_reports = []
    for seed, delta in enumerate((0.01, 0.02, 0.03, 0.04, -0.01), start=1):
        hypotheses = {}
        for name in ("facing_bet_bet_call", "round1_open_bet_check"):
            hypotheses[name] = {
                "context_matched": {"delta_mae": delta, "absolute_seat_effect_gap": 0.01},
                "all_context_comparator": {"absolute_seat_effect_gap": 0.02},
            }
        seed_reports.append({"seed": seed, "analysis": {"hypotheses": hypotheses}})

    aggregate = _aggregate_seed_results(seed_reports, min_positive_seed_fraction=0.8)
    assert aggregate["all_primary_replications_passed"] is True
    assert aggregate["seat_asymmetry"]["attenuated_after_context_matching"] is True
