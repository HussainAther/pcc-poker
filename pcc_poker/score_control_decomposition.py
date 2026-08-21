"""Post-freeze decomposition of the Score-family Control mechanism.

No policy is modified. We perturb only the learned opponent fold probability
while holding public poker state fixed, then compare how Score and Adaptive
Control action distributions respond. This localizes the cross-family
structural-recovery split to policy architecture rather than retuning the
observable.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .engine import State
from .families import AdaptiveMixturePolicy
from .policies import OpponentModel, component_scores, _softmax

FOLD_LEVELS=(0.10,0.90)
MIN_ADAPTIVE_AGGRESSION_SHIFT=0.40
MAX_SCORE_AGGRESSION_SHIFT=0.20
MIN_SENSITIVITY_RATIO=3.0


def _model_for_fold_probability(state: State, target: float) -> OpponentModel:
    model=OpponentModel(); key=f"r{state.round_index}|facing"
    # Large pseudo-sample makes smoothed fold_probability close to target.
    n=1000; folds=round(target*n); rest=n-folds
    model.context_actions[key]["fold"]=folds
    model.context_actions[key]["call"]=rest//2
    model.context_actions[key]["raise"]=rest-rest//2
    return model


def _score_control_distribution(state: State, fold_p: float):
    opponent=_model_for_fold_probability(state,fold_p)
    scores=component_scores(state,opponent,OpponentModel())["control"]
    return _softmax(scores,0.35)


def _adaptive_control_distribution(state: State, fold_p: float):
    policy=AdaptiveMixturePolicy((0.1,0.8,0.1),seed=1,temperature=0.35)
    policy.opponent_model=_model_for_fold_probability(state,fold_p)
    return policy._adaptive_control_distribution(state)


def _aggression(p): return sum(v for a,v in p.items() if a in {"bet","raise"})
def _tv(a,b): return 0.5*sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))


def representative_states():
    # Fixed public decision states spanning open/facing, rounds, and equity.
    return [
      State(private=(0,2),public=None,deck=(0,1,1,2),actor=0),
      State(private=(2,0),public=None,deck=(0,1,1,2),actor=0),
      State(private=(0,2),public=None,deck=(0,1,1,2),actor=0,contributions=(1,3),round_contributions=(0,2),history=("bet",)),
      State(private=(2,0),public=None,deck=(0,1,1,2),actor=0,contributions=(1,3),round_contributions=(0,2),history=("bet",)),
      State(private=(0,2),public=1,deck=(0,1,2),round_index=1,actor=0,history=("check","check","/")),
      State(private=(2,0),public=2,deck=(0,1,1),round_index=1,actor=0,history=("check","check","/")),
      State(private=(0,2),public=1,deck=(0,1,2),round_index=1,actor=0,contributions=(3,7),round_contributions=(0,4),history=("check","check","/","bet")),
      State(private=(2,0),public=2,deck=(0,1,1),round_index=1,actor=0,contributions=(3,7),round_contributions=(0,4),history=("check","check","/","bet")),
    ]


def run_score_control_decomposition():
    rows=[]
    for i,state in enumerate(representative_states()):
        sl=_score_control_distribution(state,FOLD_LEVELS[0]); sh=_score_control_distribution(state,FOLD_LEVELS[1])
        al=_adaptive_control_distribution(state,FOLD_LEVELS[0]); ah=_adaptive_control_distribution(state,FOLD_LEVELS[1])
        rows.append({"state":i,"legal_actions":list(state.legal_actions()),"score":{"low":sl,"high":sh,"aggression_shift":_aggression(sh)-_aggression(sl),"tv_shift":_tv(sl,sh)},"adaptive":{"low":al,"high":ah,"aggression_shift":_aggression(ah)-_aggression(al),"tv_shift":_tv(al,ah)}})
    score_shift=float(np.mean([abs(r["score"]["aggression_shift"]) for r in rows])); adaptive_shift=float(np.mean([abs(r["adaptive"]["aggression_shift"]) for r in rows]))
    score_tv=float(np.mean([r["score"]["tv_shift"] for r in rows])); adaptive_tv=float(np.mean([r["adaptive"]["tv_shift"] for r in rows]))
    ratio=adaptive_tv/max(score_tv,1e-12)
    checks={"score_response_is_weak":score_shift<=MAX_SCORE_AGGRESSION_SHIFT,"adaptive_response_is_strong":adaptive_shift>=MIN_ADAPTIVE_AGGRESSION_SHIFT,"adaptive_total_sensitivity_at_least_3x_score":ratio>=MIN_SENSITIVITY_RATIO}
    return {"status":"confirmed" if all(checks.values()) else "unresolved","mechanism_split_confirmed":all(checks.values()),"hypothesis":"Score Control is primarily static value/card-state optimization with weak learned-response gain; Adaptive Control explicitly amplifies learned opponent response and therefore expresses the information->context->intervention structure.","fold_probability_perturbation":{"low":FOLD_LEVELS[0],"high":FOLD_LEVELS[1]},"summary":{"score_mean_absolute_aggression_shift":score_shift,"adaptive_mean_absolute_aggression_shift":adaptive_shift,"score_mean_total_variation_shift":score_tv,"adaptive_mean_total_variation_shift":adaptive_tv,"adaptive_to_score_tv_ratio":ratio},"prespecified_checks":checks,"states":rows,"policy_modified":False,"human_data_accessed":False,"interpretation":"This diagnoses the Score-family structural-recovery failure; it does not resolve Poker Control and does not alter the v0.8 human-facing panel."}


def write_score_control_decomposition(path):
    report=run_score_control_decomposition(); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2)+"\n"); return report
