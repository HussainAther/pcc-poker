"""Patch 26 exploratory ablation regression checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pcc_poker.feature_ablation_audit import audit, _residualize, main

ROOT = Path(__file__).resolve().parents[1] / 'validation' / 'beyond-strategy-blind-v1'


def test_all_models_and_directions():
    result = audit(ROOT)
    assert set(result['directions']) == {'A_to_B', 'B_to_A'}
    for direction in result['directions'].values():
        assert direction['n_train'] == direction['n_test'] == 32
        for fits in direction['improvement_vs_refit_baseline'].values():
            assert len(fits) == 9
            assert all(abs(v) < 1e-10 for v in fits['baseline'].values())
            assert all(len(metrics) == 3 for metrics in fits.values())


def test_read_only_and_cli(capsys):
    paths = (ROOT/'measurements.json', ROOT/'predictions_frozen.json')
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert main([str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)['frozen_evidence_modified'] is False
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def test_residualization_ignores_test_targets():
    rng = np.random.default_rng(123)
    x, c = rng.normal(size=(12, 4)), rng.normal(size=(12, 3))
    tx, tc = rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
    r, tr = _residualize(x, c, tx, tc)
    assert r.shape == (12, 3) and tr.shape == (5, 3)
    assert np.allclose(_residualize(x, c, tx, tc + 999)[0], r)
    assert not np.allclose(_residualize(x, c, tx, tc + 999)[1], tr)


def test_invalid_penalties_rejected():
    with pytest.raises(ValueError, match='penalties'):
        audit(ROOT, penalties=(0,))
