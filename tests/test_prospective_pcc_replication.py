import json

import numpy as np
import pytest

from pcc_poker import prospective_pcc_replication as p


def sample_rows(n=8):
    rows = []
    for i in range(n):
        t = i / n
        b = [0.] * 8
        b[5:8] = [t, t*t, np.sin(3*t)]
        rows.append({'baseline': b, 'candidate': [t*t, t + .01, np.cos(t)],
                     'outcomes': [t + t*t, t - t*t, np.sin(t)]})
    return rows


def test_fixed_protocol_seeds_are_distinct():
    a = p.PROTOCOL['family_A_seeds']
    b = p.PROTOCOL['family_B_seeds']
    assert len(a) == len(b) == 32
    assert len(set(a + b)) == 64
    assert p.PROTOCOL['baseline_columns'] == [5, 6, 7]


def test_freeze_and_reject_overwrite(tmp_path):
    output = tmp_path / 'fresh'
    p.freeze(output)
    p.verify(output)
    with pytest.raises(FileExistsError):
        p.freeze(output)
    (output / 'measurements.json').write_text('[]')
    with pytest.raises(FileExistsError):
        p.verify(output)


def test_freeze_detects_protocol_mutation(tmp_path):
    out = tmp_path / 'fresh'
    p.freeze(out)
    obj = json.loads((out / 'protocol.json').read_text())
    obj['protocol']['ridge_penalty'] = 1
    (out / 'protocol.json').write_text(json.dumps(obj))
    with pytest.raises(RuntimeError):
        p.verify(out)


def test_scores_deterministic_and_not_mutating_inputs():
    train, test = sample_rows(), sample_rows()
    snapshot = json.dumps(train)
    a = p.scores(train, test, 9, 50)
    b = p.scores(train, test, 9, 50)
    assert a == b
    assert json.dumps(train) == snapshot
    assert set(a['improvement']) == set(p.ENDPOINTS)
    assert all(np.isfinite(x) for x in a['improvement'].values())
