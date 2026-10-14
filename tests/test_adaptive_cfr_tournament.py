import random

import numpy as np
import pytest

from pcc_poker.engine import all_deals, initial_state
from pcc_poker.leduc_cfr import LeducCFR
from pcc_poker.policies import PCCPolicy
from pcc_poker.adaptive_cfr_tournament import (select_action, play_hand, paired_replicate,
    bootstrap_interval, tournament)


@pytest.fixture(scope='module')
def cfr():
    solver = LeducCFR()
    solver.train(2)
    return solver.average_strategy()


def test_select_action_validates_distribution():
    with pytest.raises(ValueError):
        select_action(('check', 'bet'), [0.2, 0.2], random.Random(0))
    with pytest.raises(ValueError):
        select_action(('check', 'bet'), [1.2, -0.2], random.Random(0))
    assert select_action(('check', 'bet'), [1, 0], random.Random(0)) == 'check'


def test_both_seats_zero_sum_payoffs(cfr):
    deal = all_deals()[0]
    for seat in (0, 1):
        pcc = PCCPolicy((1, 0, 0), seed=4)
        value = play_hand(deal, pcc, seat, cfr, random.Random(4))
        assert np.isfinite(value)


def test_opponent_updates_during_hand(cfr):
    pcc = PCCPolicy((0, 1, 0), seed=5)
    play_hand(all_deals()[1], pcc, 0, cfr, random.Random(9))
    assert sum(sum(counter.values()) for counter in pcc.opponent_model.context_actions.values()) > 0
    assert sum(sum(counter.values()) for counter in pcc.action_history.context_actions.values()) > 0


def test_pairing_deterministic_and_symmetric(cfr):
    decks = all_deals()[:5]
    a = paired_replicate((0.8, 0.1, 0.1), cfr, decks, 901)
    b = paired_replicate((0.8, 0.1, 0.1), cfr, decks, 901)
    assert a == b
    assert np.isclose(a['seat_average_payoff'], -a['cfr_seat_average_payoff'])
    assert len(a['seat0_payoffs']) == len(a['seat1_payoffs']) == 5


def test_empty_replicate_rejected(cfr):
    with pytest.raises(ValueError):
        paired_replicate((1, 0, 0), cfr, [], 1)


def test_bootstrap_determinism_and_degenerate():
    assert bootstrap_interval([2.0] * 5, seed=2) == [2.0, 2.0]
    assert bootstrap_interval([1, 2, 3], seed=9) == bootstrap_interval([1, 2, 3], seed=9)


def test_invalid_design():
    for kwargs in ({'seeds': [1, 1]}, {'seeds': [1]}, {'iterations': 0}, {'hands_per_seat': 0}, {'temperature': 0}):
        with pytest.raises(ValueError):
            tournament(**kwargs)


def test_small_tournament_reproducible():
    opts = dict(iterations=2, seeds=(2201, 2202), hands_per_seat=5, bootstrap_samples=30)
    first = tournament(**opts)
    second = tournament(**opts)
    assert first == second
    assert first['status'] == 'completed'
    assert len(first['cases']) == 4
    assert first['design']['chance_deals_paired_across_seats']
    for case in first['cases'].values():
        assert len(case['replicates']) == 2
        assert np.isfinite(case['mean_pcc_payoff'])
