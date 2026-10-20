"""Regression checks for Patch 24 independent exact evaluator."""
import pytest

from pcc_poker.independent_solver_audit import audit, independent_values, main
from pcc_poker.leduc_cfr import LeducCFR


@pytest.mark.parametrize('iterations', [0, 1, 2, 5])
def test_exact_crosscheck(iterations):
    result = audit(iterations)
    assert result['status'] == 'pass', result['deltas']
    assert result['independent']['nash_conv'] >= -1e-9


def test_pure_policy_crosscheck():
    solver = LeducCFR()
    policy = {k: [1.0] + [0.0] * (len(v) - 1) for k, v in solver.regrets.items()}
    independent = independent_values(policy)
    assert abs(independent['profile_value_player0'] - solver.profile_value(policy)) < 1e-8
    for player in (0, 1):
        assert abs(independent[f'best_response_player{player}'] - solver.best_response_value(policy, player)) < 1e-8


def test_invalid_probability_rejected():
    solver = LeducCFR()
    policy = solver.average_strategy()
    key = next(iter(policy))
    policy[key] = [0.0] * len(policy[key])
    with pytest.raises(ValueError, match='invalid policy'):
        independent_values(policy)


def test_non_overwrite(tmp_path):
    target = tmp_path / 'evidence.json'
    assert main(['--iterations', '0', '--output', str(target)]) == 0
    original = target.read_bytes()
    with pytest.raises(SystemExit):
        main(['--iterations', '0', '--output', str(target)])
    assert target.read_bytes() == original
