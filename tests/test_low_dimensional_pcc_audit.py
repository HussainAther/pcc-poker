"""Regression checks for post-hoc low-dimensional PCC diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pcc_poker.low_dimensional_pcc_audit import (
    audit, main, residualize, ridge_fit_predict, standardize,
)

ROOT = Path(__file__).resolve().parents[1]/'validation'/'beyond-strategy-blind-v1'


def test_audit_shapes_and_finite_values():
    result = audit(ROOT)
    assert set(result['directions']) == {'A_to_B', 'B_to_A'}
    assert result['baseline_indices'] == [5, 6, 7]
    for direction in result['directions'].values():
        assert direction['combined_condition']['features'] == 6
        assert direction['n_train'] == direction['n_test'] == 32
        for models in direction['improvement_vs_low_dim_baseline'].values():
            assert set(models) == {'baseline', 'pressure', 'control', 'chaos', 'all', 'residualized_all'}
            assert all(v == 0. for v in models['baseline'].values())
            assert all(np.isfinite(v) for values in models.values() for v in values.values())


def test_train_only_scaling_and_residual_projection():
    train = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
    test = np.array([[100., 200.]])
    z, tz = standardize(train, test)
    assert np.allclose(z.mean(0), 0.)
    assert np.all(tz > 10.)
    c = np.c_[train[:, 0]*2, train[:, 1]*3]
    tc = np.array([[7., 9.]])
    r, rt = residualize(z, c, tz, tc, 10.)
    r2, rt2 = residualize(z, c, tz, tc+100, 10.)
    assert np.array_equal(r, r2)
    assert not np.allclose(rt, rt2)


def test_no_test_label_dependency():
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(18, dtype=float).reshape(6, 3)
    t = np.array([[2., 4.]])
    assert np.array_equal(ridge_fit_predict(x, y, t, 10.), ridge_fit_predict(x, y, t, 10.))
    with pytest.raises(ValueError, match='penalty'):
        ridge_fit_predict(x, y, t, 0.)


def test_read_only_cli_and_invalid_penalties(capsys):
    paths = [ROOT/'measurements.json', ROOT/'predictions_frozen.json']
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert main([str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)['frozen_evidence_modified'] is False
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    with pytest.raises(ValueError, match='penalties'):
        audit(ROOT, penalties=(0.,))
