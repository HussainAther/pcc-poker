"""Observable-history mode recovery and grouped evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURES = ("check", "bet", "fold", "call", "raise")
PCC_LABELS = {"pressure", "control", "chaos"}


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hand_features(records: list[dict]) -> tuple[np.ndarray, str, str]:
    counts = Counter(record["action"] for record in records)
    total = max(len(records), 1)
    values = [counts[action] / total for action in FEATURES]
    values += [
        np.mean([record["pot"] for record in records]),
        np.mean([record["to_call"] for record in records]),
        np.mean([len(record["legal_actions"]) for record in records]),
        len(records),
    ]
    return np.array(values, dtype=float), records[0]["policy_label"], records[0]["hand_id"]


def _is_test(hand_id: str) -> bool:
    return int(hashlib.sha256(hand_id.encode()).hexdigest()[:8], 16) % 5 == 0


def analyze_mode_recovery(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["hand_id"], record["actor"])].append(record)
    examples = [
        hand_features(rows) for rows in grouped.values()
        if rows[0]["policy_label"] in PCC_LABELS
    ]
    train = [item for item in examples if not _is_test(item[2])]
    test = [item for item in examples if _is_test(item[2])]
    labels = sorted({label for _, label, _ in examples})
    if len(labels) < 2 or not train or not test:
        return {"status": "insufficient_labels", "labels": labels, "examples": len(examples)}
    x_train = np.stack([x for x, _, _ in train]); mean=x_train.mean(0); scale=x_train.std(0);scale[scale<1e-9]=1
    centroids = {label: np.mean([(x-mean)/scale for x,y,_ in train if y==label], axis=0) for label in labels}
    confusion = {label: {prediction: 0 for prediction in labels} for label in labels}; correct=0
    for x, truth, _ in test:
        z=(x-mean)/scale; prediction=min(labels,key=lambda label: float(np.sum((z-centroids[label])**2)))
        confusion[truth][prediction]+=1; correct+=prediction==truth
    recalls = {
        label: confusion[label][label] / max(sum(confusion[label].values()), 1)
        for label in labels
    }
    test_counts = Counter(label for _, label, _ in test)
    return {
        "status": "completed", "feature_source": "observable_actions_and_public_betting_state_only",
        "classifier": "nearest_centroid_baseline", "train_examples": len(train), "test_examples": len(test),
        "labels": labels, "accuracy": correct/len(test),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "uniform_chance_accuracy": 1 / len(labels),
        "majority_class_accuracy": max(test_counts.values()) / len(test),
        "per_class_recall": recalls, "confusion": confusion,
        "warning": "Recovery of simulated policies is not evidence of human intention or the PCC cycle.",
    }


def analyze_file(input_path: str | Path, output_path: str | Path) -> dict:
    report = analyze_mode_recovery(load_jsonl(input_path))
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
