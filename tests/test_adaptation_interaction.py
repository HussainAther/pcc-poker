import copy
import json

import pytest

from pcc_poker.adaptation_interaction import CONTRASTS, analyze, main, paired_interval


def sample():
    entries = []
    for seed, x in ((11, 1.0), (12, 2.0), (13, 3.0)):
        means = {'frozen': 0, 'opponent_only': x, 'history_only': 2*x, 'adaptive': 4*x}
        entries.append({'seed': seed, 'conditions': {name: {'seat0_mean': v, 'seat1_mean': v,
            'seat_average_payoff': v, 'cumulative_quarter_half_full': [v, v, v]} for name, v in means.items()}})
    return {'status': 'completed', 'design': {'seeds': [11, 12, 13], 'matched_deals_across_conditions': True,
        'independent_learning_state_per_condition_and_seat': True},
        'ablations': {'frozen': [False, False], 'opponent_only': [True, False],
                      'history_only': [False, True], 'adaptive': [True, True]},
        'budgets': {'100': {'cases': {'balanced': {'replicates': entries}}}}}


def test_factorial_contrasts():
    d = analyze(sample(), bootstrap_samples=100)
    c = d['budgets']['100']['cases']['balanced']['contrasts']
    assert c['factorial_interaction']['seed_values'] == [1, 2, 3]
    assert c['factorial_interaction']['mean'] == 2
    assert c['full_vs_frozen']['mean'] == 8
    assert c['history_at_adaptive_opponent']['mean'] == 6
    assert d['analysis_status'].startswith('posthoc')


def test_interval_is_paired_and_deterministic():
    assert paired_interval([1, 2, 3], seed=3, samples=200) == paired_interval([1, 2, 3], seed=3, samples=200)
    assert paired_interval([2, 2, 2], samples=100) == [2, 2]


def test_interval_validation():
    for vals in ([], [1], [float('nan'), 3]):
        with pytest.raises(ValueError):
            paired_interval(vals)
    with pytest.raises(ValueError):
        paired_interval([1, 2], samples=0)


@pytest.mark.parametrize('change', [
    lambda d: d['design'].update(seeds=[11, 11, 13]),
    lambda d: d['design'].update(matched_deals_across_conditions=False),
    lambda d: d['ablations'].update(adaptive=[False, True]),
    lambda d: d['budgets']['100']['cases']['balanced']['replicates'][0]['conditions'].pop('frozen'),
    lambda d: d['budgets']['100']['cases']['balanced']['replicates'][0].update(seed=12),
    lambda d: d['budgets']['100']['cases']['balanced']['replicates'][0]['conditions']['frozen'].update(seat0_mean=10),
])
def test_invalid_input(change):
    d = sample()
    change(d)
    with pytest.raises(ValueError):
        analyze(d)


def test_curve_is_descriptive():
    d = sample()
    vals = d['budgets']['100']['cases']['balanced']['replicates'][0]['conditions']['adaptive']
    vals['cumulative_quarter_half_full'] = [1, 2, 4]
    out = analyze(d)['budgets']['100']['cases']['balanced']['learning_curve_descriptive']['adaptive']
    assert out['cumulative_quarter_half_full'][0] == pytest.approx(7)


def test_cli_read_only_and_no_overwrite(tmp_path):
    inp, out = tmp_path / 'source.json', tmp_path / 'result.json'
    inp.write_text(json.dumps(sample()))
    assert main(['--input', str(inp), '--output', str(out), '--bootstrap-samples', '20']) == 0
    assert inp.read_text() == json.dumps(sample())
    with pytest.raises(SystemExit):
        main(['--input', str(inp), '--output', str(out)])
    with pytest.raises(SystemExit):
        main(['--input', str(inp), '--output', str(inp)])


def test_six_distinct_contrasts():
    assert len(CONTRASTS) == 6
    assert sum(CONTRASTS['factorial_interaction'].values()) == 0
