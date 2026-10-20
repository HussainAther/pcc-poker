import json

import numpy as np
import pytest

from pcc_poker import prospective_representation_intervention as p


def rows(n=12, shift=0.):
    result = []
    for i in range(n):
        v = float(i + 1)
        result.append({'baseline': [0.]*5 + [v, v*.2, v*.05],
                       'candidate': [v*.4 + shift, v*.3 + shift, v*.1 + shift],
                       'outcomes': [v*.5 + v*v*.01, v*.1 + v*v*.02, v*.2 + v*v*.03]})
    return result


def test_seed_uniqueness_and_fixed_primary():
    ids = [p.seed(f, r, i) for r in range(p.PROTOCOL['replications'])
           for f in ('A', 'B') for i in range(p.PROTOCOL['policies_per_family'])]
    assert len(ids) == len(set(ids))
    assert p.PROTOCOL['primary_endpoint'] == 'value_vs_loose'


def test_freeze_and_existing_artifacts_protected(tmp_path):
    directory = tmp_path/'experiment'
    p.freeze(directory)
    p.verify(directory)
    with pytest.raises(FileExistsError):
        p.freeze(directory)
    (directory/'measurements.json').write_text('[]')
    with pytest.raises(FileExistsError):
        p.verify(directory)


def test_protocol_mutation_rejected(tmp_path):
    directory = tmp_path/'experiment'
    p.freeze(directory)
    obj = json.loads((directory/'protocol.json').read_text())
    obj['protocol']['ridge_penalty'] = 1.0
    (directory/'protocol.json').write_text(json.dumps(obj))
    with pytest.raises(RuntimeError):
        p.verify(directory)


def test_clipping_uses_training_bounds():
    train = np.array([[-1., 0.], [1., 3.]])
    test = np.array([[-3., 4.], [.5, 2.]])
    assert p._clamp_to_training(train, test).tolist() == [[-1., 3.], [.5, 2.]]


def test_evaluate_is_deterministic_and_nonmutating():
    source, target = rows(), rows(shift=3.)
    before = json.dumps([source, target])
    a = p.evaluate(source, target)
    assert a == p.evaluate(source, target)
    assert json.dumps([source, target]) == before
    assert set(a['mse']) == {'baseline','unchanged','clip_pcc_to_training_range',
                            'clip_all_to_training_range'}


def test_summary_fixed_primary_and_deterministic():
    def item(val):
        return {'normalized_improvement_vs_unchanged': {
            name: {endpoint: val for endpoint in p.ENDPOINTS}
            for name in ('clip_pcc_to_training_range', 'clip_all_to_training_range')}}
    cohort = [{'A_to_B': item(i), 'B_to_A': item(-i)} for i in (1.,2.,3.,4.)]
    a = p.summarize(cohort)
    assert a == p.summarize(cohort)
    assert a['primary_point_estimate_criterion_met']
    assert a['primary_bootstrap_interval_positive']
