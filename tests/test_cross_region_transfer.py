import numpy as np
import pytest

from pcc_poker.cross_region_transfer import evaluate_fold, run, summarize, main
from pcc_poker.mixed_ood import REGIONS


def _sample(name, group, target):
    return {'mixture_id': group, 'target': np.asarray(target, dtype=float), 'M2': np.array([0., 0.]),
            'M0': np.array([0.]), 'M1': np.array([0.]), 'M3': np.array([0.]),
            'public_decision_proxy': np.array([0.]), 'M2_plus_matched_transitions': np.array([0.])}


def test_train_region_excluded_and_prior_only():
    data = {r: [_sample(r, r+'-1', [1., 0., 0.]), _sample(r, r+'-2', [0., 1., 0.])] for r in REGIONS}
    test = [_sample('x', 'unique-target', [0., 0., 1.])]
    outcome = evaluate_fold(data, test, 'balanced')
    assert 'balanced' not in outcome['training_regions']
    assert outcome['training_groups'] == 8
    assert outcome['evaluation_groups'] == 1
    assert outcome['prior_only']['mae'] > 0


def test_mixture_overlap_rejected():
    data = {r: [_sample(r, 'overlap', [1., 0., 0.])] for r in REGIONS}
    with pytest.raises(ValueError, match='leakage'):
        evaluate_fold(data, [_sample('x', 'overlap', [1., 0., 0.])], 'balanced')


def test_unknown_region_rejected():
    with pytest.raises(ValueError, match='invalid'):
        evaluate_fold({}, [_sample('x', 'id', [1., 0., 0.])], 'other')


def test_empty_test_rejected():
    with pytest.raises(ValueError, match='empty'):
        evaluate_fold({}, [], 'balanced')


@pytest.mark.parametrize('seeds,train_seed', [([1801, 1801], 1800), ([1800], 1800), ([], 1800)])
def test_seed_validation(seeds, train_seed):
    with pytest.raises(ValueError, match='seeds'):
        run(train_seed=train_seed, seeds=seeds, mixtures_per_region=2, hands_per_seat=1)


def test_small_simulated_run():
    result = run(train_seed=1800, seeds=[1801], mixtures_per_region=2, hands_per_seat=3)
    assert set(result['summary']) == set(REGIONS)
    assert result['provenance']['region_excluded_from_training']
    assert result['seed_results'][0]['regions']['balanced']['evaluation_groups'] == 2


def test_determinism():
    first = run(train_seed=1800, seeds=[1801], mixtures_per_region=2, hands_per_seat=2)
    again = run(train_seed=1800, seeds=[1801], mixtures_per_region=2, hands_per_seat=2)
    assert first == again


def test_output_collision(tmp_path):
    existing = tmp_path / 'existing.json'
    existing.write_text('keep')
    with pytest.raises(FileExistsError):
        main(['--output', str(existing)])
    assert existing.read_text() == 'keep'
