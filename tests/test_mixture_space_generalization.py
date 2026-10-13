import numpy as np
import pytest

from pcc_poker.mixture_space_generalization import run, region_metrics, summarize
from pcc_poker.mixed_ood import REGIONS, sample_ood_weights
from pcc_poker.game_theoretic_baselines import MODEL_KEYS
import random


def test_prespecified_five_regions():
    assert set(REGIONS) == {'pressure_heavy', 'control_heavy', 'chaos_heavy', 'balanced', 'boundary'}


def test_all_regions_simplex():
    for region in REGIONS:
        for seed in range(10):
            weights = sample_ood_weights(region, random.Random(seed))
            assert sum(weights) == pytest.approx(1)
            assert min(weights) >= 0


def test_heavy_regions_target_expected_component():
    for i, region in enumerate(REGIONS[:3]):
        weights = sample_ood_weights(region, random.Random(42))
        assert weights[i] >= 0.8


def test_region_metrics_refuses_empty():
    with pytest.raises(ValueError, match='empty|contain'):
        region_metrics([], {})


def test_region_metrics_target_means():
    result = region_metrics([{'target': np.array([.1, .8, .1])}], {})
    assert result['target_mean']['control'] == pytest.approx(.8)


def test_duplicate_seeds_fail_without_simulation():
    with pytest.raises(ValueError, match='unique'):
        run([], seeds=[1701, 1701])


def test_invalid_design_fails_without_simulation():
    with pytest.raises(ValueError, match='positive'):
        run([], seeds=[1701], mixtures_per_region=0)


def test_models_match_patch05():
    assert 'M3' in MODEL_KEYS and 'M2_plus_matched_transitions' in MODEL_KEYS


def test_empty_training_is_rejected():
    with pytest.raises(ValueError, match='empty'):
        run([], seeds=[1701], mixtures_per_region=1, hands_per_seat=1)
