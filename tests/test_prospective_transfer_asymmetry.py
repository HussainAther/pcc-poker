import json

import pytest

from pcc_poker import prospective_transfer_asymmetry as p


def mock_cohorts(gaps):
    return [{d: {'improvement': {k: (0. if d == 'A_to_B' else gap)
                               for k in p.ENDPOINTS}}
             for d in ('A_to_B', 'B_to_A')} for gap in gaps]


def test_unique_cohort_seeds_and_protocol():
    seeds = [p.cohort_seed(f, r, i)
             for r in range(p.PROTOCOL['replications'])
             for f in ('A', 'B')
             for i in range(p.PROTOCOL['policies_per_family_per_replication'])]
    assert len(seeds) == len(set(seeds))
    assert p.PROTOCOL['primary_endpoint'] == 'value_vs_loose'
    assert p.PROTOCOL['ridge_penalty'] == 10.0


def test_freeze_rejects_overwrite_and_tampering(tmp_path):
    root = tmp_path / 'new'
    p.freeze(root)
    p.verify(root)
    with pytest.raises(FileExistsError):
        p.freeze(root)
    frozen = json.loads((root / 'protocol.json').read_text())
    frozen['protocol']['primary_endpoint'] = 'value_vs_tight'
    (root / 'protocol.json').write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError):
        p.verify(root)


def test_existing_results_refuse_new_run(tmp_path):
    root = tmp_path / 'new'
    p.freeze(root)
    (root / 'measurements.json').write_text('[]')
    with pytest.raises(FileExistsError):
        p.verify(root)


def test_paired_cohort_bootstrap_is_deterministic():
    data = mock_cohorts([1, 2, -1, 4, 2, 3, 1, 0])
    first = p.summarize(data, 200)
    assert first == p.summarize(data, 200)
    assert first['primary_point_estimate_criterion_met']
    result = first['endpoints']['value_vs_loose']
    assert result['paired_asymmetry_mean'] == 1.5
    assert result['fraction_cohorts_positive'] == .75
    assert len(result['cohort_bootstrap_ci95']) == 2


def test_no_positive_primary_asymmetry():
    report = p.summarize(mock_cohorts([-1, -2, -3]), 30)
    assert not report['primary_point_estimate_criterion_met']
    assert not report['primary_interval_excludes_zero_positive']
