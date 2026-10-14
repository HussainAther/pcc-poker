import random

import pytest

from pcc_poker.adaptive_cfr_ablations import ABLATIONS, ablation_replicate, experiment, play_ablation_hand
from pcc_poker.engine import all_deals
from pcc_poker.leduc_cfr import LeducCFR
from pcc_poker.policies import PCCPolicy


@pytest.fixture(scope='module')
def strategy():
    solver = LeducCFR()
    solver.train(2)
    return solver.average_strategy()


def test_modes_cover_independent_switches():
    assert set(ABLATIONS.values()) == {(True, True), (False, False), (True, False), (False, True)}


def test_frozen_never_updates_either_history(strategy):
    policy = PCCPolicy((.1, .8, .1), seed=5)
    for deck in all_deals()[:5]:
        play_ablation_hand(deck, policy, 0, strategy, random.Random(1), False, False)
    assert not any(policy.action_history.context_actions.values())
    assert not any(policy.opponent_model.context_actions.values())


def test_opponent_only_updates_only_opponent(strategy):
    policy = PCCPolicy((.1, .8, .1), seed=6)
    for deck in all_deals()[:10]:
        play_ablation_hand(deck, policy, 0, strategy, random.Random(2), True, False)
    assert not any(policy.action_history.context_actions.values())
    assert sum(sum(x.values()) for x in policy.opponent_model.context_actions.values()) > 0


def test_history_only_updates_only_history(strategy):
    policy = PCCPolicy((.1, .8, .1), seed=6)
    for deck in all_deals()[:10]:
        play_ablation_hand(deck, policy, 0, strategy, random.Random(2), False, True)
    assert not any(policy.opponent_model.context_actions.values())
    assert sum(sum(x.values()) for x in policy.action_history.context_actions.values()) > 0


def test_seat_pairing_and_determinism(strategy):
    args = ((.8, .1, .1), strategy, all_deals()[:6], 98)
    a = ablation_replicate(*args)
    assert a == ablation_replicate(*args)
    assert set(a) == set(ABLATIONS)
    for v in a.values():
        assert len(v['cumulative_quarter_half_full']) == 3
        assert v['seat_average_payoff'] == pytest.approx((v['seat0_mean'] + v['seat1_mean']) / 2)


@pytest.mark.parametrize('kwargs', [
    {'budgets': [0]}, {'budgets': [2, 2]}, {'seeds': [1]},
    {'seeds': [1, 1]}, {'hands_per_seat': 0}, {'temperature': 0},
    {'bootstrap_samples': 0},
])
def test_invalid_design(kwargs):
    with pytest.raises(ValueError):
        experiment(**kwargs)


def test_smoke_deterministic():
    kwargs = dict(budgets=[2, 3], seeds=[2201, 2202], hands_per_seat=4, bootstrap_samples=20)
    result = experiment(**kwargs)
    assert result == experiment(**kwargs)
    assert set(result['budgets']) == {'2', '3'}
    for budget in result['budgets'].values():
        for case in budget['cases'].values():
            assert case['conditions']['frozen']['delta_vs_frozen'] == 0
            assert len(case['replicates']) == 2
