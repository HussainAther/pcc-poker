import numpy as np
import pytest

from pcc_poker.game_theoretic_baselines import public_decision_proxy, _evaluate, run, MODEL_KEYS


def row(action='call', pot=8, to_call=2):
    return {'action': action, 'pot': pot, 'to_call': to_call,
            'round_index': 0, 'focal_seat': 0}


def test_empty_proxy_rejected():
    with pytest.raises(ValueError, match='empty'):
        public_decision_proxy([])


def test_pot_odds_definition():
    out = public_decision_proxy([row()])
    assert out[0] == 1
    assert out[1] == pytest.approx(.2)
    assert out[2] == pytest.approx(.2)


def test_no_facing_bet_defined():
    out = public_decision_proxy([row(action='check', to_call=0)])
    assert out[2] == 0
    assert np.isfinite(out).all()


def test_proxy_ignores_hidden_fields():
    plain = row()
    secret = dict(plain, private_rank=99, target_pcc_weights={'Chaos': 1},
                  hidden_pcc_weights=(1, 0, 0), showdown_equity=0.99)
    np.testing.assert_equal(public_decision_proxy([plain]), public_decision_proxy([secret]))


def test_raises_on_overlapping_groups():
    sample = {'mixture_id': 'same', 'target': np.array([.3,.4,.3])}
    with pytest.raises(ValueError, match='overlap'):
        _evaluate([sample], [sample])


def test_rejects_duplicate_seeds_before_loading_records():
    with pytest.raises(ValueError, match='unique'):
        run([], seeds=[1601, 1601])


def test_rejects_empty_seed_list():
    with pytest.raises(ValueError, match='nonempty'):
        run([], seeds=[])


def test_rejects_nonpositive_design():
    with pytest.raises(ValueError, match='positive'):
        run([], seeds=[1601], mixtures_per_seed=0)


def test_model_set_includes_unaugmented_baseline():
    assert 'M2' in MODEL_KEYS
    assert 'M2_plus_matched_transitions' in MODEL_KEYS
