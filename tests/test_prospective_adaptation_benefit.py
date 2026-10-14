import json

import pytest

from pcc_poker import prospective_adaptation_benefit as m


def test_default_design_is_fresh_and_primary_is_explicit():
    assert len(m.DEFAULT_SEEDS) == 30
    assert not set(m.DEFAULT_SEEDS) & set(range(2401, 2421))
    assert m.PRIMARY_BUDGET == 2000
    assert m.PRIMARY_ENDPOINT.startswith('paired_adaptive_minus_frozen')


@pytest.mark.parametrize('seeds', [(), (1,), (7, 7), (-1, 2), (True, 2), (1.5, 2), (2401, 2500), (2201, 2500)])
def test_rejects_invalid_or_reused_seeds(seeds):
    with pytest.raises(ValueError):
        m.experiment(seeds=seeds)


@pytest.mark.parametrize('kw', [{'hands_per_seat': 0}, {'bootstrap_samples': 0}, {'bootstrap_seed': True}, {'hands_per_seat': 1.2}])
def test_rejects_invalid_sampling(kw):
    with pytest.raises(ValueError):
        m.experiment(seeds=(2601, 2602), **kw)


def test_bootstrap_reproducibility():
    a = m._summary([0.2, -0.1, 0.4], 73, 100)
    assert a == m._summary([0.2, -0.1, 0.4], 73, 100)
    assert a['mean'] == pytest.approx(1 / 6)


def test_end_to_end_with_stub_preserves_seat_values_and_primary_decision(monkeypatch):
    class Stub:
        def __init__(self): self.n = 0
        def train(self, n): self.n += n
        def average_strategy(self): return {}
    def replicate(weights, strategy, decks, seed, temperature):
        x = (seed % 3) / 10
        return {name: {'seat0_mean': x + gain, 'seat1_mean': x + gain + 0.2,
                       'seat_average_payoff': x + gain + 0.1,
                       'cumulative_quarter_half_full': [x, x + 0.05, x + gain + 0.1]}
                for name, gain in [('adaptive', 0.4), ('frozen', 0.0)]}
    monkeypatch.setattr(m, 'LeducCFR', Stub)
    monkeypatch.setattr(m, 'ablation_replicate', replicate)
    r = m.experiment(seeds=(2601, 2602, 2603), hands_per_seat=2, bootstrap_samples=100)
    assert r['primary_decision']['criterion_met'] is True
    assert r['results']['2000']['full_vs_frozen']['mean'] == pytest.approx(0.4)
    assert len(r['results']['500']['replicates']) == 3
    assert r['results']['2000']['replicates'][0]['seat0_adaptation_gain'] == pytest.approx(0.4)
    assert r['design_sha256'] == m.experiment(seeds=(2601, 2602, 2603), hands_per_seat=2, bootstrap_samples=100)['design_sha256']


def test_no_fabricated_positive_conclusion(monkeypatch):
    class Stub:
        def train(self, n): pass
        def average_strategy(self): return {}
    def replicate(weights, strategy, decks, seed, temperature):
        return {name: {'seat0_mean': gain, 'seat1_mean': gain, 'seat_average_payoff': gain,
                       'cumulative_quarter_half_full': [gain] * 3}
                for name, gain in [('adaptive', -0.3), ('frozen', 0.0)]}
    monkeypatch.setattr(m, 'LeducCFR', Stub)
    monkeypatch.setattr(m, 'ablation_replicate', replicate)
    r = m.experiment(seeds=(2601, 2602), hands_per_seat=1, bootstrap_samples=25)
    assert r['primary_decision']['criterion_met'] is False


def test_cli_refuses_existing_report(tmp_path):
    output = tmp_path / 'already.json'
    output.write_text('{}')
    with pytest.raises(SystemExit):
        m.main(['--output', str(output)])
