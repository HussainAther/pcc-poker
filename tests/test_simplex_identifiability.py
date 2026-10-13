import numpy as np
import pytest

from pcc_poker.mixed_ood import REGIONS
from pcc_poker.simplex_identifiability import simplex_bin, audit_predictions, audit_fold, run, main


def _example(key, target):
    return {'mixture_id': key, 'target': np.array(target, dtype=float),
            'M2': np.array([0., 1.]), 'M3': np.array([0., 1.]),
            'M2_plus_matched_transitions': np.array([0., 1.])}


def test_bins():
    assert simplex_bin([0., .5, .5]) == 'edge_0_05'
    assert simplex_bin([.9, .06, .04]) == 'edge_0_05'
    assert simplex_bin([.82, .09, .09]) == 'corner_0_80'
    assert simplex_bin([.34, .33, .33]) == 'near_center'
    assert simplex_bin([.2, .3, .5]) == 'interior_other'


def test_invalid_bin():
    with pytest.raises(ValueError):
        simplex_bin([.3, .3, .3])


def test_zero_bias_and_per_mode():
    truth = np.array([[.3, .3, .4], [.1, .1, .8]])
    r = audit_predictions(truth, truth.copy())
    assert r['metrics']['mae'] == 0
    assert all(v == 0 for v in r['mean_signed_bias'].values())
    assert sum(x['n'] for x in r['simplex_bins'].values()) == 2


def test_signed_bias_direction():
    r = audit_predictions([[.2, .3, .5]], [[.3, .2, .5]])
    assert r['mean_signed_bias']['pressure'] == pytest.approx(.1)
    assert r['mean_signed_bias']['control'] == pytest.approx(-.1)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        audit_predictions([[.3, .3, .4]], [[np.nan, .3, .4]])


def test_group_leakage():
    data = {r: [_example('same', [.3, .3, .4])] for r in REGIONS}
    with pytest.raises(ValueError, match='leakage'):
        audit_fold(data, [_example('same', [.3, .3, .4])], 'balanced')


def test_invalid_penalty():
    data = {r: [_example(r, [.3, .3, .4])] for r in REGIONS}
    with pytest.raises(ValueError, match='penalties'):
        audit_fold(data, [_example('test', [.3, .3, .4])], 'balanced', [0])


def test_small_run():
    result = run(seeds=[1801], mixtures_per_region=2, hands_per_seat=2, penalties=[3.])
    assert set(result['summary']) == set(REGIONS)
    assert result['provenance']['heldout_region_excluded_from_fit']
    assert 'mean_signed_bias' in result['summary']['balanced']['models']['M3']['3.0']


def test_deterministic():
    kwargs = dict(seeds=[1801], mixtures_per_region=2, hands_per_seat=2, penalties=[3.])
    assert run(**kwargs) == run(**kwargs)


def test_collision(tmp_path):
    p = tmp_path / 'report.json'
    p.write_text('untouched')
    with pytest.raises(FileExistsError):
        main(['--output', str(p)])
    assert p.read_text() == 'untouched'
