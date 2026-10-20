"""Regression checks for the read-only post-hoc failure diagnostics."""
import json
from pathlib import Path
import shutil

import pytest

from pcc_poker.cross_family_failure_audit import audit, main

ROOT = Path(__file__).resolve().parents[1] / 'validation' / 'beyond-strategy-blind-v1'


def test_frozen_results_reproduced():
    result = audit(ROOT)
    assert result['frozen_evidence_modified'] is False
    assert set(result['directions']) == {'A_to_B', 'B_to_A'}
    for d in result['directions'].values():
        assert d['n_train'] == d['n_test'] == 32
        assert len(d['candidate_shift_pooled_sd_units']) == 3
        assert len(d['endpoints']) == 3
        for e in d['endpoints'].values():
            assert 0 <= e['fraction_test_policies_helped'] <= 1


def test_no_mutation(tmp_path):
    dst = tmp_path / 'frozen'
    dst.mkdir()
    names = ['measurements.json', 'predictions_frozen.json']
    for name in names:
        shutil.copyfile(ROOT / name, dst / name)
    before = {name: (dst/name).read_bytes() for name in names}
    audit(dst)
    assert {name: (dst/name).read_bytes() for name in names} == before


def test_misaligned_prediction_is_rejected(tmp_path):
    dst = tmp_path / 'frozen'
    dst.mkdir()
    for name in ['measurements.json', 'predictions_frozen.json']:
        shutil.copyfile(ROOT / name, dst / name)
    path = dst / 'predictions_frozen.json'
    pred = json.loads(path.read_text())
    pred['A_to_B']['base_predictions'][0][0] += 100
    path.write_text(json.dumps(pred))
    with pytest.raises(ValueError, match='alignment mismatch'):
        audit(dst)


def test_cli(capsys):
    assert main([str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)['frozen_evidence_modified'] is False
