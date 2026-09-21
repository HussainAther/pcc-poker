import numpy as np

from pcc_poker.seat_yoked_transition_validation import (
    aggregate_seed_results,
    attach_bet_call_features,
    generate_seat_yoked_evaluation,
)
from pcc_poker.context_matched_transition_validation import (
    make_examples,
)


def test_same_mixture_is_used_for_both_focal_seats():
    records, design = generate_seat_yoked_evaluation(
        seed=1201,
        mixtures=1,
        hands_per_seat=2,
    )

    assert design["mixtures"] == 1
    assert design["total_hands"] == 4

    focal_rows = [
        row
        for row in records
        if row["is_focal_policy"]
    ]

    weights_by_seat = {}

    for row in focal_rows:
        seat = row["focal_seat"]

        weights_by_seat.setdefault(
            seat,
            row["target_pcc_weights"],
        )

    assert set(weights_by_seat) == {0, 1}

    assert (
        weights_by_seat[0]
        == weights_by_seat[1]
    )


def test_mixture_produces_both_seat_examples():
    records, _ = generate_seat_yoked_evaluation(
        seed=1201,
        mixtures=1,
        hands_per_seat=2,
    )

    examples = make_examples(records)

    assert len(examples) == 2

    assert {
        example["focal_seat"]
        for example in examples
    } == {0, 1}

    assert (
        examples[0]["mixture_id"]
        == examples[1]["mixture_id"]
    )

    assert np.allclose(
        examples[0]["target"],
        examples[1]["target"],
    )


def test_aggregate_preserves_signed_seat_difference():
    reports = []

    for seed in range(5):
        reports.append(
            {
                "analysis": {
                    "effect": {
                        "by_seat": {
                            "0": {
                                "delta_mae": 0.010,
                            },
                            "1": {
                                "delta_mae": 0.004,
                            },
                        }
                    }
                }
            }
        )

    aggregate = aggregate_seed_results(
        reports
    )

    assert np.isclose(
        aggregate[
            "seat_difference"
        ]["mean_signed_difference"],
        0.006,
    )

    assert np.isclose(
        aggregate[
            "seat_difference"
        ]["mean_absolute_gap"],
        0.006,
    )


def test_attach_bet_call_features_zeros_only_target_coordinate():
    records, _ = generate_seat_yoked_evaluation(
        seed=1201,
        mixtures=1,
        hands_per_seat=4,
    )

    examples = make_examples(records)
    attach_bet_call_features(examples)

    pair_index = 0
    from pcc_poker.history_structure_probe import MIXED_PAIR_CATEGORIES
    pair_index = MIXED_PAIR_CATEGORIES.index(("bet", "call"))

    for example in examples:
        full = example["bet_call_features"]
        ablated = example["bet_call_ablated"]
        offset = len(example["M2_features"])
        differences = np.flatnonzero(~np.isclose(full, ablated))

        if np.isclose(full[offset + pair_index], 0.0):
            assert differences.size == 0
        else:
            assert differences.tolist() == [offset + pair_index]
            assert ablated[offset + pair_index] == 0.0
