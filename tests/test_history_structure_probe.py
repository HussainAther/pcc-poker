import numpy as np

from pcc_poker import history_structure_probe as probe


def test_pair_partition_is_complete_and_disjoint():
    assert set(probe.SAME_PAIR_CATEGORIES).isdisjoint(probe.MIXED_PAIR_CATEGORIES)
    assert set(probe.SAME_PAIR_CATEGORIES) | set(probe.MIXED_PAIR_CATEGORIES) == set(probe.PAIR_CATEGORIES)


def test_persistence_features_simple_sequence():
    rows = [
        {"hand_id": 1, "action": "bet"},
        {"hand_id": 1, "action": "bet"},
        {"hand_id": 1, "action": "call"},
    ]
    features = probe.persistence_features(rows)
    assert np.allclose(features, [0.5, 1.5, 2.0])
