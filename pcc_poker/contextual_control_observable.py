"""Frozen contextual-Control observable with matched/yoked public history.

The candidate score is the log-likelihood advantage of an aligned public-history
action model over a context-yoked model.  Yoking shuffles actions only among
records sharing the same static public context, preserving static-context action
margins exactly while destroying temporal alignment.  PCC weights are never
measurement inputs; they are consulted only after mixture-level aggregation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import numpy as np

from .chaos_control_decomposition import _static_record_context, _temporal_record_context
from .policies import MODES
from .simulate import generate_family_dataset

DEFAULT_CALIBRATION_SEEDS = {"score": 4101, "adaptive": 4109}
DEFAULT_EVALUATION_SEEDS = {"score": 4301, "adaptive": 4309}
MIN_CONTROL_CORRELATION = 0.20
MIN_DISCRIMINANT_MARGIN = 0.05
MAX_CROSS_FAMILY_GAP = 0.20


def _corr(a, b):
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float)
    if len(x) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _legal_probability(counts: Counter, legal, action: str, smoothing: float) -> float:
    denominator = sum(counts[a] + smoothing for a in legal)
    return (counts[action] + smoothing) / denominator


class FrozenAlignedYokedHistoryModel:
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)
        self.aligned = defaultdict(Counter)
        self.yoked = defaultdict(Counter)
        self.static = defaultdict(Counter)
        self.global_counts = Counter()

    @classmethod
    def from_records(cls, records: list[dict], *, seed: int, smoothing: float = 1.0):
        model = cls(smoothing)
        strata = defaultdict(list)
        for r in records:
            sk = _static_record_context(r)
            tk = _temporal_record_context(r)
            action = r["action"]
            model.static[sk][action] += 1
            model.aligned[tk][action] += 1
            model.global_counts[action] += 1
            strata[sk].append((tk, action))
        rng = random.Random(seed)
        for sk, items in sorted(strata.items(), key=lambda item: repr(item[0])):
            actions = [action for _, action in items]
            rng.shuffle(actions)
            for (tk, _), action in zip(items, actions):
                model.yoked[tk][action] += 1
        return model

    def probability(self, record: dict, *, condition: str) -> float:
        legal = tuple(record["legal_actions"]); action = record["action"]
        if condition == "aligned":
            counts = self.aligned.get(_temporal_record_context(record))
        elif condition == "yoked":
            counts = self.yoked.get(_temporal_record_context(record))
        elif condition == "static":
            counts = self.static.get(_static_record_context(record))
        else:
            raise ValueError(condition)
        if not counts or not sum(counts.values()):
            counts = self.static.get(_static_record_context(record)) or self.global_counts
        return _legal_probability(counts, legal, action, self.smoothing)

    def margin_checks(self) -> dict:
        aligned_static = defaultdict(Counter)
        yoked_static = defaultdict(Counter)
        for (sk, _history), counts in self.aligned.items():
            aligned_static[sk].update(counts)
        for (sk, _history), counts in self.yoked.items():
            yoked_static[sk].update(counts)
        return {
            "static_context_action_margins_preserved": dict(aligned_static) == dict(yoked_static),
            "global_action_margins_preserved": sum(aligned_static.values(), Counter()) == sum(yoked_static.values(), Counter()),
        }


def summarize(records: list[dict], model: FrozenAlignedYokedHistoryModel) -> dict:
    grouped = defaultdict(list)
    for r in records:
        if r.get("is_focal_policy"):
            pa = max(model.probability(r, condition="aligned"), 1e-12)
            py = max(model.probability(r, condition="yoked"), 1e-12)
            ps = max(model.probability(r, condition="static"), 1e-12)
            grouped[(r["policy_family"], r["mixture_id"], r["focal_seat"])].append(
                (math.log(pa) - math.log(py), math.log(pa) - math.log(ps), r)
            )
    rows = []
    for (family, mid, seat), values in sorted(grouped.items()):
        r0 = values[0][2]
        rows.append({
            "policy_family": family,
            "mixture_id": mid,
            "focal_seat": seat,
            "contextual_control_observable": float(np.mean([v[0] for v in values])),
            "aligned_over_static": float(np.mean([v[1] for v in values])),
            "decisions": len(values),
            "weights": {m: float(r0["target_pcc_weights"][m]) for m in MODES},
        })
    families = sorted({r["policy_family"] for r in rows})
    by_family = {}
    target_values = []
    for family in families:
        subset = [r for r in rows if r["policy_family"] == family]
        c = {m: _corr([r["contextual_control_observable"] for r in subset], [r["weights"][m] for r in subset]) for m in MODES}
        margin = c["control"] - max(c["pressure"], c["chaos"])
        by_family[family] = {"weight_correlations": c, "control_correlation": c["control"], "discriminant_margin": margin, "groups": len(subset)}
        target_values.append(c["control"])
    gap = abs(target_values[0] - target_values[1]) if len(target_values) == 2 else 999.0
    checks = {
        "control_positive_in_both_families": len(target_values) == 2 and all(v >= MIN_CONTROL_CORRELATION for v in target_values),
        "control_discriminant_in_both_families": len(by_family) == 2 and all(v["discriminant_margin"] >= MIN_DISCRIMINANT_MARGIN for v in by_family.values()),
        "cross_family_control_gap_at_most_0_20": gap <= MAX_CROSS_FAMILY_GAP,
    }
    return {
        "contextual_control_observable_confirmed": all(checks.values()),
        "families": by_family,
        "cross_family_control_gap": gap,
        "checks": checks,
        "thresholds": {"minimum_control_correlation": MIN_CONTROL_CORRELATION, "minimum_discriminant_margin": MIN_DISCRIMINANT_MARGIN, "maximum_cross_family_gap": MAX_CROSS_FAMILY_GAP},
        "measurement_rule": "mean log p(chosen action | aligned public history) - log p(chosen action | context-yoked public history)",
        "interpretation_rule": "Failure leaves Control observationally unresolved; thresholds and metric are not retuned post hoc.",
    }


def run_contextual_control_observable(calibration_mixtures=30, calibration_hands_per_seat=50, evaluation_mixtures=60, evaluation_hands_per_seat=100):
    calibration = []
    for family, seed in DEFAULT_CALIBRATION_SEEDS.items():
        recs, _ = generate_family_dataset(family, calibration_mixtures, calibration_hands_per_seat, seed)
        calibration.extend(recs)
    model = FrozenAlignedYokedHistoryModel.from_records(calibration, seed=41999)
    evaluation = []
    for family, seed in DEFAULT_EVALUATION_SEEDS.items():
        recs, _ = generate_family_dataset(family, evaluation_mixtures, evaluation_hands_per_seat, seed)
        evaluation.extend(recs)
    report = summarize(evaluation, model)
    report["margin_checks"] = model.margin_checks()
    report["design"] = {
        "calibration_seeds": DEFAULT_CALIBRATION_SEEDS,
        "evaluation_seeds": DEFAULT_EVALUATION_SEEDS,
        "yoke_seed": 41999,
        "calibration_mixtures": calibration_mixtures,
        "calibration_hands_per_seat": calibration_hands_per_seat,
        "evaluation_mixtures": evaluation_mixtures,
        "evaluation_hands_per_seat": evaluation_hands_per_seat,
        "weight_boundary": "PCC weights are never measurement inputs and are used only after mixture aggregation for validation correlations.",
    }
    return report


def write_contextual_control_observable(path, **kwargs):
    report = run_contextual_control_observable(**kwargs)
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
