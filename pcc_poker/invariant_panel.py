"""Fresh-seed cross-family invariance test for label-free PCC components.

Candidate measurements are fixed before evaluation.  Each is computed without
PCC weights; assigned synthetic weights are consulted only after aggregation.
A component enters the conservative human-facing panel only if its intended
axis is positive, discriminant, and similar in strength in both independently
coded policy families.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import math
import numpy as np

from .behavioral import PublicActionModel
from .policies import MODES
from .pressure_surprise_decomposition import PressureSurpriseOracle
from .simulate import generate_family_dataset

DEFAULT_CALIBRATION_SEEDS = {"score": 3401, "independent": 3409}
DEFAULT_EVALUATION_SEEDS = {"score": 3601, "independent": 3609}
MIN_TARGET_CORRELATION = 0.20
MIN_DISCRIMINANT_MARGIN = 0.05
MAX_CROSS_FAMILY_GAP = 0.20

CANDIDATES = {
    "pressure_exposure": "pressure",
    "response_compression": "pressure",
    "predicted_fold_probability": "pressure",
    "commitment_ratio": "pressure",
    "normalized_surprisal": "chaos",
    "effective_surprisal": "chaos",
}


def _corr(a, b):
    x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
    if len(x)<2 or x.std()<1e-12 or y.std()<1e-12: return 0.0
    return float(np.corrcoef(x,y)[0,1])


def _aggregate(records):
    groups=defaultdict(list)
    for r in records:
        if r.get("is_focal_policy") and "behavioral_measurements" in r:
            groups[(r["policy_family"],r["mixture_id"],r["focal_seat"])].append(r)
    rows=[]
    for (family,mid,seat), ds in sorted(groups.items()):
        ms=[d["behavioral_measurements"] for d in ds]
        row={"policy_family":family,"mixture_id":mid,"focal_seat":seat,
             "weights":{m:float(ds[0]["target_pcc_weights"][m]) for m in MODES},
             "decisions":len(ds)}
        for metric in CANDIDATES:
            row[metric]=float(np.mean([m[metric] for m in ms]))
        rows.append(row)
    return rows


def _correlations(rows, metric):
    return {m:_corr([r[metric] for r in rows],[r["weights"][m] for r in rows]) for m in MODES}


def summarize_invariant_panel(records):
    rows=_aggregate(records)
    families=sorted({r["policy_family"] for r in rows})
    if families != ["independent","score"]:
        raise ValueError("invariance test requires both independent and score families")
    candidates={}
    selected=[]
    for metric,target in CANDIDATES.items():
        by_family={}
        target_values=[]
        for family in families:
            subset=[r for r in rows if r["policy_family"]==family]
            c=_correlations(subset,metric)
            margin=c[target]-max(c[m] for m in MODES if m!=target)
            by_family[family]={"weight_correlations":c,"target_correlation":c[target],"discriminant_margin":margin}
            target_values.append(c[target])
        gap=abs(target_values[0]-target_values[1])
        checks={
            "target_positive_in_both_families": all(v>=MIN_TARGET_CORRELATION for v in target_values),
            "target_discriminant_in_both_families": all(by_family[f]["discriminant_margin"]>=MIN_DISCRIMINANT_MARGIN for f in families),
            "cross_family_target_gap_at_most_0_20": gap<=MAX_CROSS_FAMILY_GAP,
        }
        invariant=all(checks.values())
        candidates[metric]={"intended_axis":target,"families":by_family,"cross_family_target_gap":gap,"checks":checks,"family_invariant":invariant}
        if invariant: selected.append(metric)
    axis_coverage={m:[metric for metric in selected if CANDIDATES[metric]==m] for m in MODES}
    return {
        "family_invariant_panel_confirmed": bool(selected),
        "selected_invariant_components": selected,
        "axis_coverage": axis_coverage,
        "candidates": candidates,
        "thresholds":{"minimum_target_correlation":MIN_TARGET_CORRELATION,"minimum_discriminant_margin":MIN_DISCRIMINANT_MARGIN,"maximum_cross_family_target_gap":MAX_CROSS_FAMILY_GAP},
        "human_facing_rule":"Only components passing all cross-family checks are eligible for the conservative future human-data panel. Missing axes remain unresolved rather than being filled post hoc.",
    }


def run_invariant_panel(calibration_mixtures=20,calibration_hands_per_seat=25,evaluation_mixtures=60,evaluation_hands_per_seat=100,
                        score_calibration_seed=DEFAULT_CALIBRATION_SEEDS["score"],independent_calibration_seed=DEFAULT_CALIBRATION_SEEDS["independent"],
                        score_evaluation_seed=DEFAULT_EVALUATION_SEEDS["score"],independent_evaluation_seed=DEFAULT_EVALUATION_SEEDS["independent"]):
    calibration=[]
    for family,seed in (("score",score_calibration_seed),("independent",independent_calibration_seed)):
        rs,_=generate_family_dataset(family,calibration_mixtures,calibration_hands_per_seat,seed); calibration.extend(rs)
    oracle=PressureSurpriseOracle(PublicActionModel.from_records(calibration))
    evaluation=[]
    for family,seed in (("score",score_evaluation_seed),("independent",independent_evaluation_seed)):
        rs,_=generate_family_dataset(family,evaluation_mixtures,evaluation_hands_per_seat,seed,measurement_oracle=oracle); evaluation.extend(rs)
    report=summarize_invariant_panel(evaluation)
    report["design"]={"calibration_seeds":{"score":score_calibration_seed,"independent":independent_calibration_seed},"evaluation_seeds":{"score":score_evaluation_seed,"independent":independent_evaluation_seed},"calibration_mixtures":calibration_mixtures,"calibration_hands_per_seat":calibration_hands_per_seat,"evaluation_mixtures":evaluation_mixtures,"evaluation_hands_per_seat":evaluation_hands_per_seat,"weight_boundary":"PCC weights are never measurement inputs and are used only after aggregation for construct-validity correlations."}
    return report


def write_invariant_panel(path,**kwargs):
    report=run_invariant_panel(**kwargs); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); return report
