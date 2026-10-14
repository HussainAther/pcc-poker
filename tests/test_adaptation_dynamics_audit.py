import copy
import json

import pytest

from pcc_poker.adaptation_dynamics_audit import _validate, audit, main


def fixture_report():
    reps = []
    for seed in (2601, 2602, 2603):
        shift = (seed - 2601) * 0.02
        frozen = {'seat0_mean': -1., 'seat1_mean': -1., 'seat_average_payoff': -1.,
                  'cumulative_quarter_half_full': [-1., -1., -1.]}
        adaptive = {'seat0_mean': -0.8 + shift, 'seat1_mean': -0.6 + shift,
                    'seat_average_payoff': -0.7 + shift,
                    'cumulative_quarter_half_full': [-0.9 + shift, -0.8 + shift, -0.7 + shift]}
        reps.append({'seed': seed, 'conditions': {'adaptive': adaptive, 'frozen': frozen},
                     'seat0_adaptation_gain': .2 + shift, 'seat1_adaptation_gain': .4 + shift,
                     'paired_adaptation_gain': .3 + shift})
    contrasts = [r['paired_adaptation_gain'] for r in reps]
    entry = {'replicates': reps, 'full_vs_frozen': {'mean': sum(contrasts) / len(contrasts), 'seed_values': contrasts}}
    return {'status': 'completed', 'analysis_status': 'prospective_balanced_adaptation_benefit_replication',
            'design_sha256': 'test-fingerprint',
            'design': {'seeds': [2601, 2602, 2603], 'hands_per_seat': 200,
                       'paired_deals': True, 'independent_agent_state_by_condition_and_seat': True},
            'results': {'500': {'role': 'secondary', **copy.deepcopy(entry)},
                        '2000': {'role': 'primary', **copy.deepcopy(entry)}}}


def test_valid_fixture():
    assert _validate(fixture_report()) == ([2601, 2602, 2603], 200)


def test_computes_seat_gains():
    x = audit(fixture_report(), bootstrap_samples=50)['results']['2000']
    assert x['seat0_gain']['mean'] == pytest.approx(.22)
    assert x['seat1_gain']['mean'] == pytest.approx(.42)
    assert x['seat0_minus_seat1_gain']['mean'] == pytest.approx(-.2)


def test_recovers_disjoint_segments():
    x = audit(fixture_report(), bootstrap_samples=50)['results']['2000']['segments']
    assert x['first_quarter']['mean'] == pytest.approx(.12)
    assert x['second_quarter']['mean'] == pytest.approx(.32)
    assert x['second_half']['mean'] == pytest.approx(.42)
    assert x['second_half_minus_first_quarter']['mean'] == pytest.approx(.3)


def test_leave_one_out():
    x = audit(fixture_report(), bootstrap_samples=50)['results']['2000']['leave_one_seed_out']
    assert x['min_mean'] == pytest.approx(.31)
    assert x['max_mean'] == pytest.approx(.33)


def test_deterministic():
    assert audit(fixture_report(), bootstrap_samples=50) == audit(fixture_report(), bootstrap_samples=50)


@pytest.mark.parametrize('change', [
    lambda r: r['design'].update(paired_deals=False),
    lambda r: r['design'].update(seeds=[2601, 2601, 2603]),
    lambda r: r['design'].update(hands_per_seat=2),
    lambda r: r['results']['2000']['replicates'][0].update(paired_adaptation_gain=5.),
    lambda r: r['results']['2000']['replicates'][0]['conditions']['adaptive'].update(seat_average_payoff=3.),
    lambda r: r['results']['2000']['replicates'][0]['conditions']['adaptive'].update(cumulative_quarter_half_full=[0, 0]),
    lambda r: r['results']['500'].update(role='primary'),
    lambda r: r['results']['2000']['full_vs_frozen'].update(mean=5.),
    lambda r: r['results']['2000']['replicates'][0]['conditions']['adaptive'].update(seat0_mean=float('nan')),
])
def test_rejects_inconsistent_input(change):
    r = fixture_report()
    change(r)
    with pytest.raises(ValueError):
        audit(r, bootstrap_samples=50)


def test_bad_bootstrap():
    with pytest.raises(ValueError):
        audit(fixture_report(), bootstrap_samples=0)


def test_refuses_overwrite(tmp_path):
    source = tmp_path / 'source.json'
    target = tmp_path / 'result.json'
    source.write_text(json.dumps(fixture_report()))
    target.write_text('existing')
    with pytest.raises(SystemExit):
        main(['--input', str(source), '--output', str(target)])
    assert target.read_text() == 'existing'


def test_cli(tmp_path):
    source = tmp_path / 'source.json'
    target = tmp_path / 'result.json'
    source.write_text(json.dumps(fixture_report()))
    assert main(['--input', str(source), '--output', str(target), '--bootstrap-samples', '30']) == 0
    assert json.loads(target.read_text())['analysis_status'] == 'post_hoc_patch16_adaptation_dynamics_audit'
