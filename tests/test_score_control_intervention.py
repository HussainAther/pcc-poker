from pcc_poker.engine import State
from pcc_poker.policies import OpponentModel, component_scores
from pcc_poker.score_control_decomposition import _model_for_fold_probability
from pcc_poker.score_control_intervention import (
    CONTEXT_RESPONSE_GAIN,
    NEUTRAL_FOLD_PRIOR,
    ContextualScorePolicy,
)


def _state():
    return State(private=(2, 0), public=None, deck=(0, 1, 1, 2), actor=0)


def test_intervention_is_zero_centered_at_neutral_prior():
    state = _state()
    model = _model_for_fold_probability(state, NEUTRAL_FOLD_PRIOR)
    base = component_scores(state, model, OpponentModel())["control"]
    policy = ContextualScorePolicy((0.1, 0.8, 0.1), seed=1)
    policy.opponent_model = model
    # Reconstruct the policy's only intervention deterministically.
    delta = CONTEXT_RESPONSE_GAIN * (policy.opponent_model.fold_probability(state) - NEUTRAL_FOLD_PRIOR)
    assert abs(delta) < 0.01
    assert set(base) == set(state.legal_actions())


def test_context_gain_has_expected_direction():
    state = _state()
    low = _model_for_fold_probability(state, 0.10).fold_probability(state)
    high = _model_for_fold_probability(state, 0.90).fold_probability(state)
    assert CONTEXT_RESPONSE_GAIN * (high - NEUTRAL_FOLD_PRIOR) > 0
    assert CONTEXT_RESPONSE_GAIN * (low - NEUTRAL_FOLD_PRIOR) < 0
