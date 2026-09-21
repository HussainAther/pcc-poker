import numpy as np

from pcc_poker.context_matched_transition_validation import make_examples
from pcc_poker.seat_opportunity_decomposition import (
    PAIR_INDEX,
    ROUND_BUDGETS,
    _stable_transition_key,
    attach_decomposition_features,
    opportunity_standardized_block,
)
from pcc_poker.seat_yoked_transition_validation import generate_seat_yoked_evaluation


def test_stable_transition_key_does_not_depend_on_pair_label():
    base = {
        "hand_id": "h1",
        "left_decision_index": 1,
        "right_decision_index": 3,
        "round_index": 1,
        "pair": ("bet", "call"),
    }
    changed = dict(base)
    changed["pair"] = ("bet", "check")
    assert _stable_transition_key(base) == _stable_transition_key(changed)


def test_standardized_block_respects_frozen_round_budgets_when_available():
    records, _ = generate_seat_yoked_evaluation(seed=1201, mixtures=1, hands_per_seat=100)
    examples = make_examples(records)
    for example in examples:
        _, diagnostics = opportunity_standardized_block(example["rows"])
        for round_index, budget in ROUND_BUDGETS.items():
            available = diagnostics["eligible_by_round"][str(round_index)]
            selected = diagnostics["selected_by_round"][str(round_index)]
            assert selected == min(available, budget)


def test_feature_ablation_zeros_only_bet_call_coordinate():
    records, _ = generate_seat_yoked_evaluation(seed=1201, mixtures=1, hands_per_seat=100)
    examples = make_examples(records)
    attach_decomposition_features(examples)
    for example in examples:
        offset = len(example["M2_features"])
        for prefix in ("ordinary", "standardized"):
            full = example[f"{prefix}_features"]
            ablated = example[f"{prefix}_ablated"]
            differences = np.flatnonzero(~np.isclose(full, ablated))
            if np.isclose(full[offset + PAIR_INDEX], 0.0):
                assert differences.size == 0
            else:
                assert differences.tolist() == [offset + PAIR_INDEX]
                assert ablated[offset + PAIR_INDEX] == 0.0
