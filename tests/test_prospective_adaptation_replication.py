import pytest
from pcc_poker.prospective_adaptation_replication import DEFAULT_SEEDS, PRIMARY_BUDGET, SECONDARY_BUDGET, experiment, _summarize


def test_default_seeds_are_prospective_and_have_twenty_replicates():
    assert DEFAULT_SEEDS == tuple(range(2401, 2421))
    assert len(DEFAULT_SEEDS) == 20


@pytest.mark.parametrize('seeds', [[1], [1, 1], [-1, 2], [1.5, 3], [True, 2]])
def test_invalid_seeds_rejected(seeds):
    with pytest.raises(ValueError):
        experiment(seeds=seeds, hands_per_seat=1, bootstrap_samples=10)


def test_invalid_sampling_rejected():
    with pytest.raises(ValueError):
        experiment(seeds=[1, 2], hands_per_seat=0)
    with pytest.raises(ValueError):
        experiment(seeds=[1, 2], bootstrap_samples=0)


def test_paired_interval_deterministic():
    a = _summarize([0.1, -0.2, 0.3], bootstrap_seed=55, bootstrap_samples=100)
    b = _summarize([0.1, -0.2, 0.3], bootstrap_seed=55, bootstrap_samples=100)
    assert a == b
    assert a['mean'] == pytest.approx(1 / 15)


def test_small_end_to_end_replication(monkeypatch):
    from pcc_poker import prospective_adaptation_replication as module
    from pcc_poker.engine import all_deals
    class StubSolver:
        def train(self, n):
            assert n > 0
        def average_strategy(self):
            from pcc_poker.leduc_cfr import LeducCFR
            return {}
    def fake_replicate(weights, strategy, decks, seed, temperature):
        return {name: {"seat_average_payoff": (i + seed % 3) / 10} for i, name in enumerate(module.ABLATIONS)}
    monkeypatch.setattr(module, "LeducCFR", StubSolver)
    monkeypatch.setattr(module, "ablation_replicate", fake_replicate)

    report = experiment(seeds=(2451, 2452), hands_per_seat=2, bootstrap_samples=20)
    assert report['status'] == 'completed'
    assert set(report['results']) == {str(SECONDARY_BUDGET), str(PRIMARY_BUDGET)}
    assert report['results'][str(PRIMARY_BUDGET)]['role'] == 'primary'
    for block in report['results'].values():
        assert len(block['replicates']) == 2
        for rep in block['replicates']:
            c = rep['conditions']
            assert rep['primary_interaction'] == pytest.approx(c['adaptive'] - c['opponent_only'] - c['history_only'] + c['frozen'])
            assert rep['full_vs_frozen'] == pytest.approx(c['adaptive'] - c['frozen'])
            assert all(-20 <= x <= 20 for x in c.values())


def test_design_digest_reproducible(monkeypatch):
    from pcc_poker import prospective_adaptation_replication as module
    class StubSolver:
        def train(self, n): pass
        def average_strategy(self): return {}
    monkeypatch.setattr(module, "LeducCFR", StubSolver)
    monkeypatch.setattr(module, "ablation_replicate", lambda weights, strategy, decks, seed, temperature: {name: {"seat_average_payoff": float(i)} for i, name in enumerate(module.ABLATIONS)})

    a = experiment(seeds=(2453, 2454), hands_per_seat=1, bootstrap_samples=10)
    b = experiment(seeds=(2453, 2454), hands_per_seat=1, bootstrap_samples=10)
    assert a['design_sha256'] == b['design_sha256']
    assert a['results'] == b['results']
