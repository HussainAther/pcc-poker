"""Patch 32 evidence-audit regression tests using synthetic frozen data."""
import json

import pytest

from pcc_poker.prospective_pcc_replication import sha, write_exclusive
from pcc_poker.prospective_transfer_asymmetry import (PROTOCOL, frozen_object,
                                                       summarize)
from pcc_poker.transfer_asymmetry_evidence_audit import _same_numbers, audit


def test_numeric_comparison_detects_changes():
    _same_numbers({'x': [1.0, 2.0]}, {'x': [1.0, 2.0]})
    with pytest.raises(ValueError, match='numerical mismatch'):
        _same_numbers({'x': [1.0]}, {'x': [2.0]})


def test_numeric_comparison_rejects_boolean_and_nonfinite():
    with pytest.raises(ValueError):
        _same_numbers({'x': 1}, {'x': True})
    with pytest.raises(ValueError):
        _same_numbers({'x': 1.}, {'x': float('nan')})


def test_missing_experiment_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)


def test_protocol_commitment_checked_first(tmp_path):
    write_exclusive(tmp_path / 'protocol.json', {'wrong': True})
    write_exclusive(tmp_path / 'measurements.json', [])
    write_exclusive(tmp_path / 'results.json', {})
    with pytest.raises(ValueError, match='protocol/source commitment mismatch'):
        audit(tmp_path)


def test_measurements_hash_check(tmp_path):
    write_exclusive(tmp_path / 'protocol.json', frozen_object())
    write_exclusive(tmp_path / 'measurements.json', [])
    write_exclusive(tmp_path / 'results.json', {'protocol_sha256': sha((tmp_path / 'protocol.json').read_bytes()), 'measurements_sha256': 'bad'})
    with pytest.raises(ValueError, match='measurements file hash mismatch'):
        audit(tmp_path)


def test_summary_rejects_modified_aggregate():
    fake = [{'A_to_B': {'improvement': {e: -2.0 for e in PROTOCOL['endpoints']}},
             'B_to_A': {'improvement': {e: -1.0 for e in PROTOCOL['endpoints']}}} for _ in range(8)]
    expected = summarize(fake)
    damaged = json.loads(json.dumps(expected))
    damaged['endpoints'][PROTOCOL['primary_endpoint']]['paired_asymmetry_mean'] = -10.0
    with pytest.raises(ValueError, match='numerical mismatch'):
        _same_numbers(expected, damaged)
