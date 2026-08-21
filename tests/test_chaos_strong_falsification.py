from pcc_poker.chaos_strong_falsification import (
    PredictableValuePolicy,
    UniformRandomPolicy,
    run_chaos_strong_falsification,
)
from pcc_poker.engine import initial_state


def test_predictable_policy_is_deterministic_and_random_is_uniform():
    state = initial_state([0, 1, 0, 1, 2, 2])
    predictable = PredictableValuePolicy().decide(state)
    random_decision = UniformRandomPolicy(seed=3).decide(state)
    assert sum(p == 1.0 for p in predictable.probabilities.values()) == 1
    assert len(set(random_decision.probabilities.values())) == 1


def test_small_strong_falsification_smoke_has_expected_structure():
    report = run_chaos_strong_falsification(
        calibration_hands_per_seat=20,
        evaluation_hands_per_seat=20,
        replicates=2,
        calibration_seed=11,
        neutral_seed=101,
        exploiter_seed=201,
        seed_stride=10,
    )
    assert set(report["families"]) == {"score", "independent"}
    assert report["design"]["human_data_used"] is False
    assert report["exploiter_calibration"]["selection_used_only_predictable_baseline"] is True
