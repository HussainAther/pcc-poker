import copy
import json

import pytest

from pcc_poker.integrated_seat_asymmetry import integrate, main, orientation_sensitivity


def reports():
    out = {}
    for source in ('context', 'composition', 'temporal'):
        items = []
        for seed in (1501, 1502):
            if source == 'context':
                items.append({'seed': seed, 'analysis': {
                    'ordinary': {'absolute_seat_gap': .5},
                    'paired_matched': {'absolute_seat_gap': .4},
                    'context_matched': {'absolute_seat_gap': .3}}})
            else:
                key = 'overlap_weighted_conditional_tv' if source == 'composition' else 'overlap_weighted_orientation_gap'
                summary = {'zero_support_mixtures': 0, key: .2 if source == 'composition' else 0.0}
                if source == 'composition':
                    summary['overlap_weighted_bet_call_absolute_gap'] = .1
                items.append({'seed': seed, 'summary': summary})
        out[source] = {'status': 'completed', 'seeds': [1501, 1502],
                       'mixtures_per_seed': 2, 'hands_per_seat': 10,
                       'seed_results': items,
                       'provenance': {'human_data_accessed': False}}
    return out


def test_sensitivity_controls():
    assert orientation_sensitivity()['status'] == 'passed'
    assert orientation_sensitivity()['injected_reversal_gap'] == 1


def test_combined_metrics_without_additive_attribution():
    result = integrate(reports(), bootstrap_iterations=50)
    assert result['summary']['context_vs_count_gap_change']['mean'] == pytest.approx(.1)
    assert result['summary']['temporal_orientation_gap']['mean'] == 0
    assert result['status'] == 'completed'
    assert 'NOT be summed' in result['limitations']
    assert json.dumps(result)


def test_mismatched_seeds_rejected():
    data = reports()
    data['temporal']['seeds'] = [1502, 1501]
    with pytest.raises(ValueError, match='mismatched seeds'):
        integrate(data)


def test_missing_human_provenance_rejected():
    data = reports()
    data['composition']['provenance'] = {}
    with pytest.raises(ValueError, match='provenance'):
        integrate(data)


def test_missing_support_does_not_become_zero():
    data = reports()
    data['temporal']['seed_results'][1]['summary']['overlap_weighted_orientation_gap'] = None
    result = integrate(data, bootstrap_iterations=25)
    assert result['summary']['temporal_orientation_gap']['supported_seeds'] == 1
    assert result['summary']['temporal_orientation_gap']['seed_resampling_interval_95'] is None


def test_deterministic_bootstrap():
    assert integrate(reports(), 50) == integrate(reports(), 50)


def test_cli_reads_reports_without_modifying_them(tmp_path):
    data = reports()
    paths = {}
    for source, report in data.items():
        path = tmp_path / f'{source}.json'
        path.write_text(json.dumps(report))
        paths[source] = path
    target = tmp_path / 'out.json'
    assert main(['--context', str(paths['context']), '--composition', str(paths['composition']),
                 '--temporal', str(paths['temporal']), '--bootstrap-iterations', '20',
                 '--output', str(target)]) == 0
    assert json.loads(target.read_text())['status'] == 'completed'
    for source, path in paths.items():
        assert json.loads(path.read_text()) == data[source]


def test_cli_refuses_overwrite(tmp_path):
    data = reports()
    paths = {}
    for source, report in data.items():
        path = tmp_path / f'{source}.json'
        path.write_text(json.dumps(report))
        paths[source] = path
    with pytest.raises(SystemExit):
        main(['--context', str(paths['context']), '--composition', str(paths['composition']),
              '--temporal', str(paths['temporal']), '--output', str(paths['temporal'])])
