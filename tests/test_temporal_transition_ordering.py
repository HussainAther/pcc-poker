import json

import pytest

from pcc_poker.temporal_transition_ordering import (
    compare_pair, directed_transitions, run_temporal_transition_ordering, summarize,
)


def example(directions, seat):
    rows = []
    for i, direction in enumerate(directions):
        for j, action in enumerate(direction):
            rows.append({'hand_id': f'{seat}-{i}', 'decision_index': j,
                         'action': action, 'round_index': 0,
                         'to_call': 2 if j else 0, 'pot': 6})
    return {'rows': rows}


def test_reversed_order_detected_after_matching_pair_and_context():
    a = example([('bet', 'call')] * 4, 0)
    b = example([('call', 'bet')] * 4, 1)
    result = compare_pair(a, b, permutations=19)
    assert result['overlap'] == 4
    assert result['orientation_gap'] == pytest.approx(1)
    assert result['signed_orientation_gap'] == pytest.approx(1)


def test_identical_direction_has_zero_gap():
    result = compare_pair(example([('bet', 'call')], 0),
                          example([('bet', 'call')], 1), permutations=19)
    assert result['orientation_gap'] == 0
    assert result['permutation_p_value'] == 1


def test_composition_matching_excludes_nonoverlapping_pair():
    result = compare_pair(example([('bet', 'call')], 0),
                          example([('check', 'bet')], 1))
    assert result['orientation_gap'] is None
    assert result['overlap'] == 0
    assert summarize([result])['overlap_weighted_orientation_gap'] is None


def test_private_fields_do_not_affect_ordering():
    a = example([('call', 'bet')], 0)
    b = example([('bet', 'call')], 1)
    before = compare_pair(a, b, permutations=19, seed=2)
    for row in a['rows'] + b['rows']:
        row['private_rank'] = 99
        row['target_pcc_weights'] = {'chaos': 9}
    assert compare_pair(a, b, permutations=19, seed=2) == before
    assert directed_transitions(a['rows'])[0]['directed_pair'] == ('call', 'bet')


def test_permutation_is_reproducible():
    a = example([('bet', 'call')] * 3 + [('call', 'bet')] * 3, 0)
    b = example([('bet', 'call')] * 2 + [('call', 'bet')] * 4, 1)
    assert compare_pair(a, b, 50, 17) == compare_pair(a, b, 50, 17)


def test_small_run_is_serializable_and_deterministic():
    kwargs = dict(seeds=[1501], mixtures_per_seed=1, hands_per_seat=12, permutations=9)
    a = run_temporal_transition_ordering(**kwargs)
    b = run_temporal_transition_ordering(**kwargs)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        run_temporal_transition_ordering(seeds=[1501, 1501])
    with pytest.raises(ValueError):
        run_temporal_transition_ordering(permutations=-1)
    with pytest.raises(ValueError):
        compare_pair(example([], 0), example([], 1), -1)
