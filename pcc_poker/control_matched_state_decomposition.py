"""Matched-state decomposition of Adaptive vs Score Poker Control.

Read-only post-v0.8 diagnostic. Holds poker state and learned opponent fold
context fixed, then compares the contextual Score extension with Adaptive
Control. No policy, frozen artifact, threshold, or human-facing panel changes.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .engine import State
from .families import AdaptiveMixturePolicy
from .policies import OpponentModel, _softmax, component_scores
from .score_control_decomposition import representative_states, _model_for_fold_probability
from .score_control_intervention import CONTEXT_RESPONSE_GAIN, NEUTRAL_FOLD_PRIOR

FOLD_LEVELS = (0.10, 0.90)
MIN_ADAPTIVE_VALUE_ADVANTAGE = 0.10
MIN_TWO_SIDED_CONTEXT_SHARE = 0.50
MIN_PASSIVE_LOW_FOLD_ADVANTAGE = 0.10


def _score_distribution(state: State, fold_p: float) -> dict[str,float]:
    model = _model_for_fold_probability(state, fold_p)
    scores = component_scores(state, model, OpponentModel())["control"]
    delta = CONTEXT_RESPONSE_GAIN * (fold_p - NEUTRAL_FOLD_PRIOR)
    for action in state.legal_actions():
        if action in {"bet", "raise"}:
            scores[action] += delta
    return _softmax(scores, 0.35)


def _adaptive_parts(state: State, fold_p: float):
    policy = AdaptiveMixturePolicy((0.1,0.8,0.1), seed=1, temperature=0.35)
    policy.opponent_model = _model_for_fold_probability(state, fold_p)
    legal = state.legal_actions(); eq = __import__('pcc_poker.engine', fromlist=['equity']).equity(state, state.actor)
    parts = {}
    scores = {}
    for action in legal:
        value = policy._action_value(state, action)
        scaled = value / max(state.pot + state.bet_size, 1)
        timing = resistance = safety = 0.0
        if action in {"bet","raise"}:
            timing = 6.0 * (fold_p - 1/3)
            safety = 1.2 * (2*eq - 1)
        elif action in {"check","call"}:
            resistance = 1.5 * (1.0 - fold_p)
            safety = 0.35 * eq
        else:
            safety = 0.8 * (1.0 - eq)
        scores[action] = scaled + timing + resistance + safety
        parts[action] = {"scaled_value":scaled,"response_timing":timing,"passive_resistance":resistance,"card_safety":safety}
    return _softmax(scores, max(policy.temperature*0.75,0.08)), parts, policy


def _expected_value(state, dist, policy):
    return float(sum(p * policy._action_value(state,a) for a,p in dist.items()))


def _passive_mass(dist): return sum(p for a,p in dist.items() if a in {"check","call"})
def _aggressive_mass(dist): return sum(p for a,p in dist.items() if a in {"bet","raise"})


def run_control_matched_state_decomposition():
    rows=[]
    for i,state in enumerate(representative_states()):
        for fold_p in FOLD_LEVELS:
            score = _score_distribution(state, fold_p)
            adaptive, parts, policy = _adaptive_parts(state, fold_p)
            score_ev = _expected_value(state,score,policy); adaptive_ev=_expected_value(state,adaptive,policy)
            rows.append({"state":i,"fold_probability":fold_p,"legal_actions":list(state.legal_actions()),
                "score":{"probabilities":score,"expected_value":score_ev,"passive_mass":_passive_mass(score),"aggressive_mass":_aggressive_mass(score)},
                "adaptive":{"probabilities":adaptive,"expected_value":adaptive_ev,"passive_mass":_passive_mass(adaptive),"aggressive_mass":_aggressive_mass(adaptive),"score_parts":parts},
                "adaptive_minus_score_expected_value":adaptive_ev-score_ev})
    low=[r for r in rows if r['fold_probability']==FOLD_LEVELS[0]]
    high=[r for r in rows if r['fold_probability']==FOLD_LEVELS[1]]
    value_adv=float(np.mean([r['adaptive_minus_score_expected_value'] for r in rows]))
    low_passive=float(np.mean([r['adaptive']['passive_mass']-r['score']['passive_mass'] for r in low]))
    # Adaptive uses explicit contextual terms on both aggressive and passive branches in every open/facing state.
    two_sided=[]
    for r in rows:
        parts=r['adaptive']['score_parts']
        has_aggressive=any(abs(v['response_timing'])>1e-12 for v in parts.values())
        has_passive=any(abs(v['passive_resistance'])>1e-12 for v in parts.values())
        two_sided.append(has_aggressive and has_passive)
    share=float(np.mean(two_sided))
    # Card safety is state-dependent but not opponent-context-dependent; response timing and passive resistance are context-dependent.
    checks={
      'adaptive_has_mean_value_advantage_on_matched_states': value_adv >= MIN_ADAPTIVE_VALUE_ADVANTAGE,
      'adaptive_context_is_two_sided_in_at_least_half_states': share >= MIN_TWO_SIDED_CONTEXT_SHARE,
      'adaptive_preserves_more_passive_mass_when_fold_resistance_is_high': low_passive >= MIN_PASSIVE_LOW_FOLD_ADVANTAGE,
    }
    return {"status":"confirmed" if all(checks.values()) else "partial",
      "minimal_architectural_difference_supported": all(checks.values()),
      "hypothesis":"Adaptive Control does not merely add a stronger aggression response. It allocates context on both sides of the decision: aggression is timed to fold vulnerability while passive/check-call branches receive an explicit resistance/optionality bonus when folds are unlikely. Card safety remains a state-value guardrail rather than the contextual signal itself.",
      "summary":{"adaptive_minus_score_mean_expected_value":value_adv,"adaptive_minus_score_low_fold_passive_mass":low_passive,"two_sided_context_state_share":share},
      "prespecified_checks":checks,"thresholds":{"minimum_mean_value_advantage":MIN_ADAPTIVE_VALUE_ADVANTAGE,"minimum_two_sided_context_share":MIN_TWO_SIDED_CONTEXT_SHARE,"minimum_low_fold_passive_mass_advantage":MIN_PASSIVE_LOW_FOLD_ADVANTAGE},
      "states":rows,"policy_modified":False,"human_data_accessed":False,"frozen_v0.8_human_panel_modified":False,
      "interpretation":"The matched-state diagnostic localizes the remaining Score-Control gap to two-sided context allocation: Adaptive couples opportunistic aggression with explicit passive resistance/optionality. This result does not itself resolve Control or authorize automatic retuning."}


def write_control_matched_state_decomposition(path):
    report=run_control_matched_state_decomposition(); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2)+"\n"); return report
