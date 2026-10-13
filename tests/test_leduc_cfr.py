"""Solver invariants and exact best-response sanity tests."""
import numpy as np
import pytest
from pcc_poker.leduc_cfr import LeducCFR, regret_matching, run


def test_regret_matching():
    assert np.allclose(regret_matching(np.array([-2., 0.])), [0.5, 0.5])
    assert np.allclose(regret_matching(np.array([1., 3.])), [0.25, 0.75])


def test_tree_consistency():
    solver = LeducCFR()
    assert len(solver.roots) == 90
    assert len(solver.infos) == 528
    assert len(solver.nodes) == 14490


def test_average_strategy_normalized():
    solver = LeducCFR()
    solver.train(3)
    for values in solver.average_strategy().values():
        assert np.all(values >= 0)
        assert np.isclose(values.sum(), 1)


def test_best_responses_dominate_profile():
    solver = LeducCFR()
    solver.train(3)
    result = solver.evaluate()
    v = result['profile_value_player0']
    assert result['best_response_player0'] >= v - 1e-9
    assert result['best_response_player1'] >= -v - 1e-9
    assert np.isclose(result['nash_conv'], result['best_response_player0'] + result['best_response_player1'])
    assert np.isclose(result['exploitability'], result['nash_conv'] / 2)


def test_first_checkpoints_improve():
    checkpoints = run((1, 5, 20))['checkpoints']
    assert checkpoints[-1]['exploitability'] < checkpoints[0]['exploitability']


def test_iterations_validate():
    solver = LeducCFR()
    with pytest.raises(ValueError):
        solver.train(0)
    for budgets in [(0, 1), (5, 5), (8, 3), ()]:
        with pytest.raises(ValueError):
            run(budgets)


def test_determinism():
    a = LeducCFR(); b = LeducCFR()
    a.train(2); b.train(2)
    assert a.evaluate() == b.evaluate()
