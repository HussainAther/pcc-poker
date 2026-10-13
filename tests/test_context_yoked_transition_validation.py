import numpy as np
import pytest

from pcc_poker.context_matched_transition_validation import make_examples
from pcc_poker.context_yoked_transition_validation import (
    _bin, attach_context_features, public_context_transitions,
    run_context_yoked_transition_validation,
)
from pcc_poker.paired_matched_opportunity_validation import PAIR_INDEX, _pair_examples
from pcc_poker.seat_yoked_transition_validation import generate_seat_yoked_evaluation


def examples():
    records, _ = generate_seat_yoked_evaluation(seed=1501, mixtures=2, hands_per_seat=30)
    return make_examples(records)


def test_bins_are_fixed_and_boundary_inclusive():
    assert [_bin(x, (2, 5, 10)) for x in (0, 2, 3, 5, 10, 11)] == [0, 0, 1, 1, 2, 3]


def test_exact_context_count_yoking_and_ablation():
    data = examples()
    attach_context_features(data)
    for a, b in _pair_examples(data):
        da, db = (e['context_matching_diagnostics'] for e in (a, b))
        assert da['matched_context_counts'] == db['matched_context_counts']
        assert da['selected'] == db['selected']
        for ex in (a, b):
            full = ex['context_matched_features']
            ablated = ex['context_matched_ablated']
            idx = len(ex['M2_features']) + PAIR_INDEX
            assert np.allclose(np.delete(full, idx), np.delete(ablated, idx))
            assert ablated[idx] == 0


def test_deterministic_and_does_not_use_private_fields():
    first, second = examples(), examples()
    for ex in second:
        for row in ex['rows']:
            row['private_rank'] = -999
            row['target_pcc_weights'] = {'pressure': 999}
    attach_context_features(first)
    attach_context_features(second)
    for a, b in zip(first, second):
        assert np.array_equal(a['context_matched_features'], b['context_matched_features'])
        assert a['context_matching_diagnostics'] == b['context_matching_diagnostics']


def test_rejects_duplicate_or_empty_seeds():
    with pytest.raises(ValueError, match='seeds'):
        run_context_yoked_transition_validation([], seeds=[1501, 1501])
    with pytest.raises(ValueError, match='seeds'):
        run_context_yoked_transition_validation([], seeds=[])
