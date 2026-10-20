"""Patch 23 integrity tests: no rerun or modifications to frozen experiment."""
import json
from pathlib import Path
import shutil
from pcc_poker.blind_evidence_audit import audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'validation' / 'beyond-strategy-blind-v1'


def test_existing_frozen_evidence_is_consistent():
    result = audit(SOURCE)
    assert result['status'] == 'PASS', result['errors']
    assert result['events_checked'] == 6


def test_tampering_is_detected(tmp_path):
    target = tmp_path / 'validation' / SOURCE.name
    shutil.copytree(SOURCE, target)
    (target / 'measurements.json').write_text('{}')
    result = audit(target)
    assert result['status'] == 'FAIL'
    assert any('commitment mismatch' in e for e in result['errors'])


def test_report_tampering_is_detected(tmp_path):
    target = tmp_path / 'validation' / SOURCE.name
    shutil.copytree(SOURCE, target)
    report = json.loads((target / 'report.json').read_text())
    report['results']['A_to_B']['improvement'][0] = 999
    (target / 'report.json').write_text(json.dumps(report))
    errors = audit(target)['errors']
    assert any('report results differ' in e for e in errors)
