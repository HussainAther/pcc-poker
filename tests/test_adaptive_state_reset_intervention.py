"""Patch 18 controlled reset protocol tests."""
import json

import pytest

from pcc_poker import adaptive_state_reset_intervention as m


def test_default_seeds_are_fresh():
    assert len(m.DEFAULT_SEEDS) == 30
    assert not (set(m.DEFAULT_SEEDS) & set(range(2601, 2631)))


def test_invalid_designs(monkeypatch):
    for kwargs in ({'seeds': [1, 1]}, {'seeds': [2601, 2801]},
                   {'hands_per_seat': 3}, {'reset_period': 1},
                   {'reset_period': 200}, {'bootstrap_samples': 0},
                   {'bootstrap_seed': True}):
        with pytest.raises(ValueError):
            m.experiment(**kwargs)


def test_seed_reset_count_and_first_epoch_identity(monkeypatch):
    constructed = []
    class FakeRNG:
        def __init__(self):
            self.value = 0
        def getstate(self):
            return self.value
        def setstate(self, value):
            self.value = value
    class FakePCC:
        def __init__(self, weights, seed, temperature):
            constructed.append(seed)
            self.rng = FakeRNG()
    def fake_hand(deck, pcc, seat, strategy, rng):
        pcc.rng.value += 1
        return float(pcc.rng.value)
    monkeypatch.setattr(m, 'PCCPolicy', FakePCC)
    monkeypatch.setattr(m, 'play_reset_hand', fake_hand)
    weights = m.PURE_MIXTURES['balanced']
    a = m.seat_arm(weights, {}, [None] * 10, 2801, 0, None)
    b = m.seat_arm(weights, {}, [None] * 10, 2801, 0, 4)
    assert a == b == [float(i) for i in range(1, 11)]
    assert len(set(constructed)) == 1
    assert len(constructed) == 4


def test_replicate_pairing_with_stub(monkeypatch):
    monkeypatch.setattr(m, 'seat_arm', lambda weights, strategy, decks, seed, seat, reset_period, temperature=m.TEMPERATURE:
                        [float(seat + (0 if reset_period is None else 1))] * len(decks))
    result = m.replicate({}, [None] * 8, 2801)
    assert result['continuous']['seat_average_payoff'] == 0.5
    assert result['periodic_reset']['seat_average_payoff'] == 1.5
    assert result['continuous']['segments']['second_half'] == 0.5


def test_experiment_smoke_with_stub(monkeypatch):
    class FakeSolver:
        def train(self, n):
            assert n == 2000
        def average_strategy(self):
            return {}
    monkeypatch.setattr(m, 'LeducCFR', FakeSolver)
    monkeypatch.setattr(m, 'all_deals', lambda: [('deck',)])
    def stub(strategy, decks, seed, reset_period):
        return {k: {'seat0_mean': v, 'seat1_mean': v, 'seat_average_payoff': v,
                    'segments': {'first_quarter': v, 'second_quarter': v, 'second_half': v}}
                for k, v in [('continuous', 1.0), ('periodic_reset', 0.0), ('reset_each_hand', -1.0)]}
    monkeypatch.setattr(m, 'replicate', stub)
    report = m.experiment(seeds=[2801, 2802, 2803], hands_per_seat=8, reset_period=4, bootstrap_samples=100)
    assert report['primary']['mean'] == 1.0
    assert report['primary_decision']['criterion_met']
    assert report['secondary_continuous_minus_reset_each_hand']['mean'] == 2.0
    assert len(report['replicates']) == 3
    assert report['design_sha256'] == m.experiment(seeds=[2801, 2802, 2803], hands_per_seat=8, reset_period=4, bootstrap_samples=100)['design_sha256']


def test_cli_writes_and_refuses_overwrite(monkeypatch, tmp_path):
    output = tmp_path / 'report.json'
    monkeypatch.setattr(m, 'experiment', lambda **kwargs: {
        'status': 'completed', 'primary': {'mean': 1.0},
        'primary_decision': {'criterion_met': True},
        'secondary_continuous_minus_reset_each_hand': {'mean': 2.0}})
    assert m.main(['--output', str(output)]) == 0
    assert json.loads(output.read_text())['status'] == 'completed'
    with pytest.raises(SystemExit):
        m.main(['--output', str(output)])
