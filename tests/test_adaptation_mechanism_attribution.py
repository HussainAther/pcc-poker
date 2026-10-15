"""Patch 21 component attribution contracts."""
import json
import math

import pytest
from pcc_poker import adaptation_mechanism_attribution as m


def test_fixed_design():
    assert m.SEEDS == tuple(range(3401, 3431))
    assert m.CFR_BUDGET == 2000
    assert m.HANDS_PER_SEAT == 200


@pytest.mark.parametrize('seeds', [[], [1], [1, 1], [True, 2], [-1, 2], [3201, 3401], [2601, 3401]])
def test_bad_seeds(seeds):
    with pytest.raises(ValueError):
        m.validate(seeds, 200, .35, 10, 1)


@pytest.mark.parametrize('kwargs', [{'hands_per_seat': 2}, {'hands_per_seat': True},
                                    {'temperature': math.inf}, {'temperature': 0},
                                    {'bootstrap_samples': 0}, {'bootstrap_samples': True},
                                    {'bootstrap_seed': True}])
def test_invalid_settings(kwargs):
    values = dict(seeds=(3401, 3402), hands_per_seat=4, temperature=.35,
                  bootstrap_samples=10, bootstrap_seed=2)
    values.update(kwargs)
    with pytest.raises(ValueError):
        m.validate(**values)


def test_stubbed_matched_factorial(monkeypatch):
    class CFR:
        def train(self, n):
            assert n == 2000
        def average_strategy(self):
            return {'fixed': True}
    calls = []
    monkeypatch.setattr(m, 'LeducCFR', CFR)
    monkeypatch.setattr(m, 'all_deals', lambda: [1, 2, 3])
    def fake(weights, strategy, schedule, seed, temperature):
        calls.append((seed, tuple(schedule)))
        values = {'frozen': 0., 'opponent_only': 1., 'history_only': 2., 'adaptive': 4.}
        return {k: {'seat0_mean': v, 'seat1_mean': v, 'seat_average_payoff': v}
                for k, v in values.items()}
    monkeypatch.setattr(m, 'ablation_replicate', fake)
    r = m.experiment(seeds=(3401, 3402), hands_per_seat=4, bootstrap_samples=20)
    assert r['contrasts']['full_vs_frozen']['mean'] == 4
    assert r['contrasts']['opponent_at_frozen_history']['mean'] == 1
    assert r['contrasts']['history_at_frozen_opponent']['mean'] == 2
    assert r['contrasts']['factorial_interaction']['mean'] == 1
    assert r['seat_full_vs_frozen']['0']['mean'] == 4
    assert len(r['design_sha256']) == 64
    assert calls == [(3401, calls[0][1]), (3402, calls[1][1])]
    assert r == m.experiment(seeds=(3401, 3402), hands_per_seat=4, bootstrap_samples=20)


def test_reject_inconsistent_payoff(monkeypatch):
    class CFR:
        def train(self, n): pass
        def average_strategy(self): return {}
    monkeypatch.setattr(m, 'LeducCFR', CFR)
    monkeypatch.setattr(m, 'all_deals', lambda: [1])
    monkeypatch.setattr(m, 'ablation_replicate', lambda *args: {
        k: {'seat0_mean': 1., 'seat1_mean': 1., 'seat_average_payoff': math.nan}
        for k in m.ABLATIONS})
    with pytest.raises(ValueError):
        m.experiment(seeds=(3401, 3402), hands_per_seat=4)


def test_report_not_overwritten(tmp_path):
    out = tmp_path / 'report.json'
    out.write_text('{}')
    with pytest.raises(SystemExit):
        m.main(['--output', str(out)])
    assert json.loads(out.read_text()) == {}


def test_intervals_constant():
    assert m.summarize([2., 2., 2.], samples=20, seed=3)['paired_replicate_interval_95'] == [2., 2.]
