import json

import pytest

from pcc_poker import prospective_evidence_synthesis as s
from pcc_poker import prospective_pcc_replication as p


def _fixture(tmp_path):
    folder = tmp_path / 'experiment'
    p.freeze(folder)
    rows = []
    for family in ('A', 'B'):
        for i, seed in enumerate(p.PROTOCOL[f'family_{family}_seeds']):
            t = (i + 1) / 32
            rows.append({'id': f'{family}-{i:02d}', 'family': family, 'seed': seed,
                         'baseline': [t + j * 0.03 + (family == 'B') * 0.1 for j in range(8)],
                         'candidate': [t, t * t, t ** 3],
                         'outcomes': [t + (family == 'B') * .1, t * t + .1, t ** 3 + .2]})
    p.write_exclusive(folder / 'measurements.json', rows)
    groups = {fam: [r for r in rows if r['family'] == fam] for fam in ('A', 'B')}
    results = {f'{a}_to_{b}': p.scores(groups[a], groups[b], p.PROTOCOL['bootstrap_seed'] + j, 60)
               for j, (a, b) in enumerate((('A', 'B'), ('B', 'A')))}
    report = {'protocol_sha256': p.sha((folder/'protocol.json').read_bytes()),
              'measurements_sha256': p.sha((folder/'measurements.json').read_bytes()),
              'results': results,
              'primary_point_estimate_criterion_met': all(
                  v > 0 for data in results.values() for v in data['improvement'].values())}
    p.write_exclusive(folder / 'results.json', report)
    return folder


def test_hashes_and_results(tmp_path, monkeypatch):
    folder = _fixture(tmp_path)
    # Keep test fast without changing the frozen protocol itself.
    original_scores = p.scores
    monkeypatch.setattr(s.p, 'scores', lambda a,b,seed,repetitions: original_scores(a,b,seed,60))
    result = s.audit(folder)
    assert result['status'] == 'verified'
    assert result['integrity']['freeze_cli_canonical_sha256'] != result['integrity']['protocol_file_byte_sha256']
    assert result['integrity']['reported_results_recomputed']


def test_measurements_tamper_rejected(tmp_path):
    folder = _fixture(tmp_path)
    with (folder/'measurements.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='measurement file-byte hash'):
        s.audit(folder)


def test_protocol_tamper_rejected(tmp_path):
    folder = _fixture(tmp_path)
    path = folder/'protocol.json'
    contents = json.loads(path.read_text())
    contents['protocol']['ridge_penalty'] = 9
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='protocol content'):
        s.audit(folder)


def test_report_tamper_rejected(tmp_path, monkeypatch):
    folder = _fixture(tmp_path)
    path = folder/'results.json'
    report = json.loads(path.read_text())
    report['results']['A_to_B']['improvement']['value_vs_tight'] += 0.1
    path.write_text(json.dumps(report))
    original_scores = p.scores
    monkeypatch.setattr(s.p, 'scores', lambda a,b,seed,repetitions: original_scores(a,b,seed,60))
    with pytest.raises(ValueError, match='reported values disagree'):
        s.audit(folder)
