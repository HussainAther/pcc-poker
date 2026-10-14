"""Patch 15 report-only audit tests; never train CFR or run tournaments."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from pcc_poker.adaptation_reliability_audit import audit, main


def fixture():
    design = {'seeds': [11, 12, 13], 'primary_cfr_iterations': 2000, 'secondary_cfr_iterations': 500}
    digest = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    results = {}
    for budget in ('500', '2000'):
        reps = []
        for i, seed in enumerate(design['seeds']):
            conditions = {'adaptive': -0.5 + 0.1*i, 'frozen': -0.8, 'opponent_only': -0.7, 'history_only': -0.6}
            reps.append({'seed': seed, 'conditions': conditions,
                         'primary_interaction': conditions['adaptive']-conditions['opponent_only']-conditions['history_only']+conditions['frozen'],
                         'full_vs_frozen': conditions['adaptive']-conditions['frozen']})
        def summary(vals):
            return {'mean': sum(vals)/len(vals), 'seed_values': vals}
        results[budget] = {'role': 'primary' if budget == '2000' else 'secondary', 'replicates': reps,
                           'interaction': summary([r['primary_interaction'] for r in reps]),
                           'full_vs_frozen': summary([r['full_vs_frozen'] for r in reps]),
                           'absolute_adaptive_payoff': summary([r['conditions']['adaptive'] for r in reps])}
    return {'analysis_status': 'prospective_balanced_adaptation_replication', 'design': design,
            'design_sha256': digest, 'results': results}


def test_contrasts_and_replicate_count():
    x = audit(fixture(), bootstrap_samples=40)
    case = x['results']['2000']
    assert case['replicate_count'] == 3
    assert case['contrasts']['full_vs_frozen']['mean'] == pytest.approx(0.4)
    assert case['contrasts']['factorial_interaction']['mean'] == pytest.approx(0.1)
    assert case['contrasts']['full_vs_frozen']['positive_replicates'] == 3


def test_deterministic_bootstrap():
    assert audit(fixture(), bootstrap_samples=40) == audit(fixture(), bootstrap_samples=40)


def test_rejects_fingerprint_mismatch():
    d = fixture(); d['design_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='fingerprint'):
        audit(d)


def test_rejects_out_of_order_seeds():
    d = fixture(); d['results']['2000']['replicates'].reverse()
    with pytest.raises(ValueError, match='seeds'):
        audit(d)


def test_rejects_inconsistent_contrast():
    d = fixture(); d['results']['500']['replicates'][0]['full_vs_frozen'] += 1
    with pytest.raises(ValueError, match='inconsistent'):
        audit(d)


def test_rejects_missing_condition():
    d = fixture(); del d['results']['500']['replicates'][0]['conditions']['frozen']
    with pytest.raises(ValueError, match='condition'):
        audit(d)


def test_rejects_nonfinite_payoff():
    d = fixture(); d['results']['500']['replicates'][0]['conditions']['adaptive'] = float('nan')
    with pytest.raises(ValueError, match='payoff'):
        audit(d)


def test_missing_seat_and_curves_marked_unavailable():
    d = audit(fixture(), bootstrap_samples=20)
    assert 'seat_specific_payoffs' in d['unavailable_from_source']
    assert 'within_tournament_learning_curves' in d['unavailable_from_source']
    assert d['provenance']['simulation_rerun'] is False


def test_cli_read_only_and_no_overwrite(tmp_path):
    inp, out = tmp_path/'input.json', tmp_path/'out.json'
    inp.write_text(json.dumps(fixture()))
    assert main(['--input', str(inp), '--output', str(out), '--bootstrap-samples', '10']) == 0
    assert inp.read_text() == json.dumps(fixture())
    with pytest.raises(SystemExit):
        main(['--input', str(inp), '--output', str(out), '--bootstrap-samples', '10'])


def test_invalid_bootstrap():
    with pytest.raises(ValueError, match='positive'):
        audit(fixture(), bootstrap_samples=0)
