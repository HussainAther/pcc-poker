import numpy as np
import pytest

from pcc_poker.context_matched_transition_validation import make_examples
from pcc_poker.paired_matched_opportunity_validation import (
    PAIR_INDEX,
    _pair_examples,
    attach_paired_features,
)
from pcc_poker.seat_yoked_transition_validation import generate_seat_yoked_evaluation


def _examples(seed=1401, mixtures=2, hands=60):
    records, _ = generate_seat_yoked_evaluation(
        seed=seed,
        mixtures=mixtures,
        hands_per_seat=hands,
    )
    return make_examples(records)


def test_pair_examples_requires_both_seats():
    examples = _examples(mixtures=1)
    with pytest.raises(ValueError, match="not paired"):
        _pair_examples(examples[:1])


def test_paired_matching_selects_exact_same_count_by_round():
    examples = _examples()
    attach_paired_features(examples)
    for seat0, seat1 in _pair_examples(examples):
        d0 = seat0["paired_opportunity_diagnostics"]
        d1 = seat1["paired_opportunity_diagnostics"]
        for round_index in ("0", "1"):
            expected = min(
                d0["eligible_by_round"][round_index],
                d1["eligible_by_round"][round_index],
            )
            assert d0["matched_budgets"][round_index] == expected
            assert d1["matched_budgets"][round_index] == expected
            assert d0["selected_by_round"][round_index] == expected
            assert d1["selected_by_round"][round_index] == expected


def test_paired_feature_ablation_zeros_only_bet_call_coordinate():
    examples = _examples(mixtures=1)
    attach_paired_features(examples)
    for example in examples:
        offset = len(example["M2_features"])
        for prefix in ("ordinary", "paired_matched"):
            full = example[f"{prefix}_features"]
            ablated = example[f"{prefix}_ablated"]
            differences = np.flatnonzero(~np.isclose(full, ablated))
            if np.isclose(full[offset + PAIR_INDEX], 0.0):
                assert differences.size == 0
            else:
                assert differences.tolist() == [offset + PAIR_INDEX]
                assert ablated[offset + PAIR_INDEX] == 0.0


def test_attach_paired_features_is_deterministic():
    first = _examples(mixtures=1)
    second = _examples(mixtures=1)
    attach_paired_features(first)
    attach_paired_features(second)
    for left, right in zip(first, second):
        assert np.allclose(left["paired_matched_features"], right["paired_matched_features"])
        assert left["paired_opportunity_diagnostics"] == right["paired_opportunity_diagnostics"]
