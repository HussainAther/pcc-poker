"""Patch 20 prospective memory-retention replication contracts."""
import json
import math

import pytest
from pcc_poker import prospective_memory_retention as m


def test_frozen_primary_design():
    assert m.SEEDS == tuple(range(3201, 3231))
    assert m.PRIMARY_PERIOD == 10
    assert m.SECONDARY_PERIODS == (25, 100)
    assert m.CFR_BUDGET == 2000


@pytest.mark.parametrize('seeds', [[], [1], [1, 1], [True, 2], [-1, 2],
                                   [3001, 3201], [2801, 3201], [2601, 3201], [2401, 3201]])
def test_bad_seeds(seeds):
    with pytest.raises(ValueError):
        m._validate(seeds, 200, 10, 1)


@pytest.mark.parametrize('kwargs', [dict(hands_per_seat=99), dict(hands_per_seat=True),
                                    dict(bootstrap_samples=0), dict(bootstrap_samples=True),
                                    dict(bootstrap_seed=True)])
def test_bad_settings(kwargs):
    params = dict(seeds=(3201, 3202), hands_per_seat=200,
                  bootstrap_samples=10, bootstrap_seed=1)
    params.update(kwargs)
    with pytest.raises(ValueError):
        m._validate(**params)


def test_smoke_matched_and_independent(monkeypatch):
    class Solver:
        def train(self, iterations):
            assert iterations == 2000
        def average_strategy(self):
            return {'fixed': 1}
    seen = []
    monkeypatch.setattr(m, 'LeducCFR', Solver)
    monkeypatch.setattr(m, 'all_deals', lambda: [10, 20, 30])
    def seat_payoffs(strategy, schedule, seed, seat, opp, hist):
        seen.append((seed, seat, opp, hist, tuple(schedule)))
        assert opp == hist
        return [float(4 + seat + (opp is None) + (opp == 25) / 2) for _ in schedule]
    monkeypatch.setattr(m, 'seat_payoffs', seat_payoffs)
    result = m.experiment(seeds=(3201, 3202), hands_per_seat=100, bootstrap_samples=50)
    assert len(seen) == 16
    assert len({call[-1] for call in seen[:8]}) == 1
    assert result['primary']['mean'] == 1
    assert result['primary_decision']['criterion_met']
    assert result['secondary']['reset_25']['mean'] == .5
    assert result['secondary']['reset_100']['mean'] == 1
    assert result['design']['arms']['reset_10'] == {'opponent_period': 10, 'history_period': 10}
    assert len(result['design_sha256']) == 64
    assert result == m.experiment(seeds=(3201, 3202), hands_per_seat=100, bootstrap_samples=50)


def test_non_finite_output_rejected(monkeypatch):
    class Solver:
        def train(self, n): pass
        def average_strategy(self): return {}
    monkeypatch.setattr(m, 'LeducCFR', Solver)
    monkeypatch.setattr(m, 'all_deals', lambda: [1])
    monkeypatch.setattr(m, 'seat_payoffs', lambda *args: [math.nan] * len(args[1]))
    with pytest.raises(ValueError):
        m.experiment(seeds=(3201, 3202), hands_per_seat=100)


def test_report_overwrite_refused(tmp_path):
    out = tmp_path / 'report.json'
    out.write_text('{}')
    with pytest.raises(SystemExit):
        m.main(['--output', str(out)])
    assert json.loads(out.read_text()) == {}


def test_interval_positive_negative():
    assert m._summary([1.0, 1.0, 1.0], 10, 1)['paired_replicate_interval_95'] == [1, 1]
    assert m._summary([-1.0, -1.0, -1.0], 10, 1)['paired_replicate_interval_95'] == [-1, -1]
