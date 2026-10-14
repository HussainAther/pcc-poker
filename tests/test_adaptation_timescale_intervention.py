"""Tests for Patch 19 exploratory reset-timescale experiment."""
import json
import random

import pytest

from pcc_poker import adaptation_timescale_intervention as m
from pcc_poker.policies import OpponentModel, PCCPolicy


def test_conditions_expected():
    assert m.conditions() == [('continuous', None, None), ('both_every_1', 1, 1),
        ('both_every_10', 10, 10), ('both_every_25', 25, 25),
        ('both_every_50', 50, 50), ('both_every_100', 100, 100),
        ('opponent_only_reset', 50, None), ('history_only_reset', None, 50)]


@pytest.mark.parametrize('periods,component', [([], 50), ([0], 50), ([1, 1], 50),
                                                   ([True], 50), ([10], 0), ([10], False)])
def test_invalid_conditions(periods, component):
    with pytest.raises(ValueError):
        m.conditions(periods, component)


def test_rng_preserved_during_opponent_reset(monkeypatch):
    observed = []
    def fake_hand(deck, pcc, seat, strategy, rng):
        observed.append((id(pcc), id(pcc.opponent_model), id(pcc.action_history), pcc.rng.random(), rng.random()))
        return float(deck)
    monkeypatch.setattr(m, 'play_hand', fake_hand)
    result = m.seat_payoffs({}, [1, 2, 3, 4], 7, 0, 2, None)
    assert result == [1, 2, 3, 4]
    assert len({x[0] for x in observed}) == 1
    assert observed[0][1] == observed[1][1] and observed[1][1] != observed[2][1]
    assert len({x[2] for x in observed}) == 1
    assert [x[3] for x in observed] == [random.Random(56).random()] + [x[3] for x in observed][1:]
    expected = random.Random(56)
    assert [x[3] for x in observed] == [expected.random() for _ in range(4)]


def test_only_history_resets(monkeypatch):
    seen = []
    def fake_hand(deck, pcc, seat, strategy, rng):
        seen.append((id(pcc.opponent_model), id(pcc.action_history)))
        return 1.0
    monkeypatch.setattr(m, 'play_hand', fake_hand)
    m.seat_payoffs({}, [1, 2, 3, 4], 6, 0, None, 2)
    assert len({v[0] for v in seen}) == 1
    assert seen[0][1] == seen[1][1] and seen[1][1] != seen[2][1]


def test_experiment_smoke_without_solver_training(monkeypatch):
    class FakeSolver:
        def train(self, count):
            assert count == 2000
        def average_strategy(self):
            return {}
    monkeypatch.setattr(m, 'LeducCFR', FakeSolver)
    monkeypatch.setattr(m, 'all_deals', lambda: [1, 2, 3])
    def fake_seat(strategy, decks, seed, seat, opponent_period, history_period, temperature=m.TEMPERATURE):
        return [float(2 + (opponent_period is None) + (history_period is None) + seat) for _ in decks]
    monkeypatch.setattr(m, 'seat_payoffs', fake_seat)
    result = m.experiment(seeds=[3001, 3002], hands_per_seat=8, periods=[1, 4],
                          component_period=4, bootstrap_samples=20)
    assert result['status'] == 'completed'
    assert len(result['replicates']) == 2
    assert result['summaries']['both_every_1']['continuous_minus_condition']['mean'] == 2
    assert result['summaries']['opponent_only_reset']['continuous_minus_condition']['mean'] == 1
    assert result['summaries']['history_only_reset']['continuous_minus_condition']['mean'] == 1
    assert result['replicates'][0]['arms']['continuous']['seat_average_payoff'] == 4.5
    assert len(result['design_sha256']) == 64
    assert result == m.experiment(seeds=[3001, 3002], hands_per_seat=8, periods=[1, 4],
                                  component_period=4, bootstrap_samples=20)


@pytest.mark.parametrize('kwargs', [{'seeds': [2801, 3001]}, {'seeds': [3001, 3001]},
    {'hands_per_seat': 3}, {'hands_per_seat': 8, 'periods': [10]},
    {'bootstrap_samples': 0}, {'bootstrap_seed': True}])
def test_invalid_experiment(kwargs):
    with pytest.raises(ValueError):
        m.experiment(**kwargs)


def test_output_refuses_overwrite(tmp_path, monkeypatch):
    dest = tmp_path / 'existing.json'
    dest.write_text('{}')
    with pytest.raises(SystemExit):
        m.main(['--output', str(dest)])
    assert json.loads(dest.read_text()) == {}


def test_seat_payoffs_invalid():
    with pytest.raises(ValueError):
        m.seat_payoffs({}, [], 2, 0, None, None)
    with pytest.raises(ValueError):
        m.seat_payoffs({}, [1], 2, 2, None, None)
    with pytest.raises(ValueError):
        m.seat_payoffs({}, [1], 2, 0, 0, None)
