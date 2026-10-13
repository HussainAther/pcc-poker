import numpy as np
import pytest
from pcc_poker.mechanistic_identifiability import pairwise_audit, knn_predict, evaluate, cfr_readiness, run


def item(group, seat, x, y):
    return {'mixture_id': group, 'focal_seat': seat, 'M3': np.array([x], float), 'target': np.array(y, float)}


def test_pairwise_detects_distinct_targets_close_features():
    rows = [item('a', 0, 0, [1, 0, 0]), item('b', 0, 0, [0, 1, 0]), item('c', 0, 3, [0, 0, 1])]
    assert pairwise_audit(rows)['closest_pairs_target_far_fraction'] == 1.0


def test_pairwise_does_not_compare_same_group():
    rows = [item('a', 0, 0, [1, 0, 0]), item('a', 0, 0, [0, 1, 0]), item('b', 0, 1, [0, 0, 1])]
    assert pairwise_audit(rows)['comparisons'] == 2


def test_pairwise_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        pairwise_audit([], near_feature_quantile=0)


def test_knn_exact_neighbor():
    train = [item('a', 0, 0, [1, 0, 0]), item('b', 0, 10, [0, 1, 0])]
    assert np.allclose(knn_predict(train, [item('c', 0, 0, [0,0,1])], k=1), [[1,0,0]])


def test_knn_rejects_k():
    with pytest.raises(ValueError):
        knn_predict([item('a', 0, 0, [1,0,0])], [item('b', 0, 1, [1,0,0])], k=2)


def test_knn_does_not_use_test_target():
    train = [item('a', 0, 0, [1,0,0]), item('b', 0, 1, [0,1,0])]
    a = item('c', 0, 0, [0,0,1]); b = item('c', 0, 0, [0,0,1])
    assert np.array_equal(knn_predict(train, [a], k=1), knn_predict(train, [b], k=1))


def test_leakage_rejected():
    with pytest.raises(ValueError, match='overlap'):
        evaluate([item('a',0,0,[1,0,0])], [item('a',0,0,[1,0,0])], k=1)


def test_seed_validation():
    with pytest.raises(ValueError):
        run(train_seed=1800, seeds=(1800,))


def test_cfr_readiness_is_not_solver():
    out = cfr_readiness()
    assert out['zero_sum_terminal_utilities']
    assert out['consistent_information_set_legal_actions']
    assert out['solver_status'] == 'not_implemented'
