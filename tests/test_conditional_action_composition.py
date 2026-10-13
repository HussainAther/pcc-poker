import pytest

from pcc_poker.conditional_action_composition import (
    _summarize, compare_pair, run_conditional_action_composition,
)


def _example(pairs, seat):
    rows = []
    for index, pair in enumerate(pairs):
        rows.extend([
            {'hand_id': f'{seat}-{index}', 'decision_index': 0,
             'action': pair[0], 'round_index': 0, 'to_call': 0, 'pot': 3},
            {'hand_id': f'{seat}-{index}', 'decision_index': 1,
             'action': pair[1], 'round_index': 0, 'to_call': 2, 'pot': 6},
        ])
    return {'rows': rows}


def test_identical_conditional_composition_has_zero_gap():
    first = _example([('bet', 'call'), ('check', 'bet')], 0)
    second = _example([('bet', 'call'), ('check', 'bet')], 1)
    result = compare_pair(first, second)
    assert result['overlap'] == 2
    assert result['conditional_tv'] == 0
    assert result['conditional_bet_call_gap'] == 0


def test_distinct_composition_with_same_public_context_is_detected():
    result = compare_pair(_example([('bet', 'call')], 0),
                          _example([('check', 'bet')], 1))
    assert result['overlap'] == 1
    assert result['conditional_tv'] == pytest.approx(1)
    assert result['conditional_bet_call_gap'] == pytest.approx(1)
    assert result['conditional_bet_call_signed_gap'] == pytest.approx(1)


def test_no_common_support_is_not_reported_as_zero():
    a = _example([('bet', 'call')], 0)
    b = _example([('bet', 'call')], 1)
    b['rows'][1]['to_call'] = 20
    result = compare_pair(a, b)
    assert result['conditional_tv'] is None
    assert _summarize([result])['overlap_weighted_conditional_tv'] is None


def test_private_values_and_labels_do_not_affect_composition():
    a = _example([('bet', 'call')], 0)
    b = _example([('check', 'bet')], 1)
    before = compare_pair(a, b)
    for row in a['rows'] + b['rows']:
        row['private_rank'] = -100
        row['target_pcc_weights'] = {'control': 900}
    assert compare_pair(a, b) == before


def test_run_is_deterministic_and_serializable():
    import json
    kwargs = {'seeds': [1501], 'mixtures_per_seed': 2, 'hands_per_seat': 25}
    a = run_conditional_action_composition(**kwargs)
    b = run_conditional_action_composition(**kwargs)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a['aggregate']['supported_seeds'] <= 1


def test_invalid_seeds_rejected():
    with pytest.raises(ValueError, match='seeds'):
        run_conditional_action_composition(seeds=[1501, 1501])
