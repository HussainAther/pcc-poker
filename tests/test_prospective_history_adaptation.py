"""Patch 22 prospective history-only primary endpoint tests."""
import math

import pytest

from pcc_poker import prospective_history_adaptation as m


def test_prespecified_design():
    assert m.SEEDS == tuple(range(3601, 3631))
    assert m.CFR_BUDGET == 2000
    assert m.PRIMARY_ARMS == ('history_only', 'frozen')
    assert m.HANDS_PER_SEAT == 200


@pytest.mark.parametrize('seeds', [[], [1], [1, 1], [True, 3601], [-1, 3601],
                                  [3401, 3601], [3201, 3601], [2601, 3601]])
def test_bad_seeds(seeds):
    with pytest.raises(ValueError):
        m.validate(seeds, 200, 10, 1)


@pytest.mark.parametrize('options', [(True, 10, 1), (2, 10, 1), (4, 0, 1),
                                     (4, True, 1), (4, 10, True)])
def test_bad_settings(options):
    with pytest.raises(ValueError):
        m.validate((3601, 3602), *options)


def test_bootstrap_summary():
    x = m.summarize([1.0, 1.0, 1.0], 100, 7)
    assert x['mean'] == 1 and x['paired_replicate_interval_95'] == [1, 1]
    with pytest.raises(ValueError):
        m.summarize([math.nan, 1.0], 5, 2)


def test_stubbed_prospective_primary(monkeypatch):
    class Solver:
        def train(self, iterations):
            assert iterations == 2000
        def average_strategy(self):
            return {'frozen': True}
    monkeypatch.setattr(m, 'LeducCFR', Solver)
    monkeypatch.setattr(m, 'all_deals', lambda: [0, 1, 2])
    calls = []
    def fake(weights, strategy, deals, seed, temperature):
        calls.append((seed, tuple(deals)))
        values = {'frozen': 0., 'opponent_only': -1., 'history_only': 2., 'adaptive': 1.}
        return {name: {'seat0_mean': v, 'seat1_mean': v, 'seat_average_payoff': v}
                for name, v in values.items()}
    monkeypatch.setattr(m, 'ablation_replicate', fake)
    result = m.experiment(seeds=(3601, 3602), hands_per_seat=4, bootstrap_samples=30)
    assert result['primary']['mean'] == 2
    assert result['primary_decision']['criterion_met'] is True
    assert result['secondary']['full_vs_frozen']['mean'] == 1
    assert result['secondary']['opponent_only_vs_frozen']['mean'] == -1
    assert result['secondary']['full_vs_history_only']['mean'] == -1
    assert result['primary_by_seat']['0']['mean'] == 2
    assert len(result['design_sha256']) == 64
    assert [s for s, _ in calls] == [3601, 3602]
    assert result == m.experiment(seeds=(3601, 3602), hands_per_seat=4, bootstrap_samples=30)


def test_primary_negative_is_not_confirmed(monkeypatch):
    class Solver:
        def train(self, n): pass
        def average_strategy(self): return {}
    monkeypatch.setattr(m, 'LeducCFR', Solver)
    monkeypatch.setattr(m, 'all_deals', lambda: [0])
    def fake(*args):
        values = {'frozen': 1., 'history_only': 0., 'opponent_only': 1., 'adaptive': 0.}
        return {k: {'seat0_mean': v, 'seat1_mean': v, 'seat_average_payoff': v}
                for k, v in values.items()}
    monkeypatch.setattr(m, 'ablation_replicate', fake)
    outcome = m.experiment(seeds=(3601, 3602), hands_per_seat=4, bootstrap_samples=30)
    assert not outcome['primary_decision']['criterion_met']


def test_bad_seat_accounting(monkeypatch):
    class Solver:
        def train(self, n): pass
        def average_strategy(self): return {}
    monkeypatch.setattr(m, 'LeducCFR', Solver)
    monkeypatch.setattr(m, 'all_deals', lambda: [0])
    monkeypatch.setattr(m, 'ablation_replicate', lambda *a: {
        k: {'seat0_mean': 1., 'seat1_mean': 1., 'seat_average_payoff': 8.}
        for k in m.ABLATIONS})
    with pytest.raises(ValueError):
        m.experiment(seeds=(3601, 3602), hands_per_seat=4)


def test_no_overwrite(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text('{}')
    with pytest.raises(SystemExit):
        m.main(['--output', str(report)])
