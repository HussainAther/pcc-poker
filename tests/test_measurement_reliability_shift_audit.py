"""Patch 28: deterministic, read-only shift diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pcc_poker.measurement_reliability_shift_audit import (
    audit, dispersion_bootstrap, shift_diagnostics, error_shift_diagnostics, main,
)

ROOT = Path(__file__).resolve().parents[1] / 'validation' / 'beyond-strategy-blind-v1'


def test_bootstrap_determinism_and_validation():
    x = np.arange(24., dtype=float).reshape(8, 3)
    assert dispersion_bootstrap(x, repetitions=40) == dispersion_bootstrap(x, repetitions=40)
    with pytest.raises(ValueError):
        dispersion_bootstrap(x, repetitions=5)


def test_shift_is_train_defined():
    train = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
    test = np.array([[99., 100.], [100., 101.]])
    report = shift_diagnostics(train, test)
    assert report['any_feature_outside_training_range_fraction'] == 1.
    assert report['any_feature_abs_z_gt_threshold_fraction'] == 1.
    with pytest.raises(ValueError):
        shift_diagnostics(train, test[:, :1])


def test_error_strata_do_not_train_on_heldout_labels():
    train = np.arange(12., dtype=float).reshape(6, 2)
    test = np.array([[0., 1.], [50., 51.]])
    y = np.arange(18., dtype=float).reshape(6, 3)
    target = np.ones((2, 3))
    a = error_shift_diagnostics(train, test, y, target)
    b = error_shift_diagnostics(train, test, y, target + 1)
    assert a['outside_training_range']['n_flagged'] == 1
    assert a['outside_training_range']['n_unflagged'] == 1
    assert a != b


def test_cli_and_frozen_evidence_unmodified(capsys):
    paths = [ROOT/'measurements.json', ROOT/'predictions_frozen.json']
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert main([str(ROOT), '--bootstrap-repetitions', '40']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['frozen_evidence_modified'] is False
    assert set(result['directions']) == {'A_to_B', 'B_to_A'}
    assert all(len(v['shift']['abs_test_mean_z']) == 6 for v in result['directions'].values())
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
