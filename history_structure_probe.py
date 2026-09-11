from collections import Counter, defaultdict
from itertools import combinations_with_replacement

import numpy as np

from pcc_poker.analyze import load_jsonl
from pcc_poker.mixed import ACTIONS, _metrics, _ridge_predict
from pcc_poker.mixed_ood_ablation import _nested_features
from pcc_poker.policies import MODES

PAIR_CATEGORIES = tuple(combinations_with_replacement(sorted(ACTIONS), 2))
SAME_PAIR_CATEGORIES = tuple((action, action) for action in sorted(ACTIONS))
MIXED_PAIR_CATEGORIES = tuple(pair for pair in PAIR_CATEGORIES if pair[0] != pair[1])
TOP_CONTROL_PAIRS = (("bet", "call"), ("bet", "check"), ("call", "raise"))

def mixed_pair_block_for_context(
    rows,
    target_round,
    target_facing_bet,
):
    counts = Counter()
    total = 0

    by_hand = defaultdict(list)

    for row in rows:
        by_hand[row["hand_id"]].append(row)

    for hand_rows in by_hand.values():
        hand_rows = sorted(
            hand_rows,
            key=lambda r: r["decision_index"],
        )

        for left, right in zip(
            hand_rows,
            hand_rows[1:],
        ):
            # Assign the transition to the context in which
            # the second action was taken.
            round_index = right["round_index"]
            facing_bet = right["to_call"] > 0

            if (
                round_index != target_round
                or facing_bet != target_facing_bet
            ):
                continue

            pair = tuple(
                sorted(
                    (
                        left["action"],
                        right["action"],
                    )
                )
            )

            if pair in MIXED_PAIR_CATEGORIES:
                counts[pair] += 1
                total += 1

    total = max(total, 1)

    return np.asarray(
        [
            counts[pair] / total
            for pair in MIXED_PAIR_CATEGORIES
        ],
        dtype=float,
    )

def context_key(row):
    return (
        row["focal_seat"],
        row.get("round"),
        bool(row.get("facing_bet")),
    )

def persistence_features(rows):
    by_hand = defaultdict(list)
    for row in rows:
        by_hand[row["hand_id"]].append(row["action"])

    repeated = 0
    transitions = 0
    run_lengths = []
    for actions in by_hand.values():
        if not actions:
            continue
        current_run = 1
        for left, right in zip(actions, actions[1:]):
            transitions += 1
            if left == right:
                repeated += 1
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)

    return np.asarray(
        [
            repeated / max(transitions, 1),
            float(np.mean(run_lengths)) if run_lengths else 0.0,
            float(np.max(run_lengths)) if run_lengths else 0.0,
        ],
        dtype=float,
    )


def make_examples(records):
    grouped = defaultdict(list)
    for row in records:
        if row.get("is_focal_policy"):
            grouped[(row["mixture_id"], row["focal_seat"])].append(row)

    examples = []
    for (mixture_id, focal_seat), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: (r["hand_id"], r["decision_index"]))
        base = _nested_features(rows)

        by_hand = defaultdict(list)
        for row in rows:
            by_hand[row["hand_id"]].append(row["action"])

        pairs = []
        for actions in by_hand.values():
            for left, right in zip(actions, actions[1:]):
                pairs.append(tuple(sorted((left, right))))

        counts = Counter(pairs)
        total = max(len(pairs), 1)
        same_block = np.asarray([counts[pair] / total for pair in SAME_PAIR_CATEGORIES], dtype=float)
        mixed_block = np.asarray([counts[pair] / total for pair in MIXED_PAIR_CATEGORIES], dtype=float)
        unordered_block = np.asarray([counts[pair] / total for pair in PAIR_CATEGORIES], dtype=float)
        persistence_block = persistence_features(rows)
        target = np.asarray([rows[0]["target_pcc_weights"][mode] for mode in MODES], dtype=float)

        examples.append(
            {
                "mixture_id": mixture_id,
                "focal_seat": focal_seat,
                "target": target,
                "M2_features": base["M2"],
                "persistence_features": np.concatenate([base["M2"], persistence_block]),
                "same_pair_features": np.concatenate([base["M2"], same_block]),
                "mixed_pair_features": np.concatenate([base["M2"], mixed_block]),
                "unordered_features": np.concatenate([base["M2"], unordered_block]),
                "M3_features": base["M3"],
                "_rows": rows,  # Added to allow context matching in probes
            }
        )
    return examples


def zero_mixed_pairs(train, test, pairs, feature_name):
    pair_indices = [MIXED_PAIR_CATEGORIES.index(pair) for pair in pairs]
    for dataset in (train, test):
        for example in dataset:
            features = example["mixed_pair_features"].copy()
            offset = len(example["M2_features"])
            for pair_index in pair_indices:
                features[offset + pair_index] = 0.0
            example[feature_name] = features


def main():
    train_records = load_jsonl("outputs/mixed-recovery-data.jsonl")
    test_records = load_jsonl("outputs/mixed-ood-data.jsonl")
    train = make_examples(train_records)
    test = make_examples(test_records)
    truth = np.stack([row["target"] for row in test])

    models = (
        "M2_features",
        "persistence_features",
        "same_pair_features",
        "mixed_pair_features",
        "unordered_features",
        "M3_features",
    )

    print("=== Overall ===\n")
    predictions = {}
    for model in models:
        predictions[model] = _ridge_predict(train, test, model)
        print(f"{model:22s} MAE={_metrics(truth, predictions[model])['mae']:.6f}")

    control_subset = [i for i, row in enumerate(test) if row["mixture_id"].startswith("ood-control_heavy-")]
    control_truth = truth[control_subset]
    full_mixed_mae = _metrics(control_truth, predictions["mixed_pair_features"][control_subset])["mae"]

    print("\n=== Control-heavy fixed-normalization zero-out ===\n")
    pair_deltas = []
    for pair in MIXED_PAIR_CATEGORIES:
        name = "zero_" + "_".join(pair)
        zero_mixed_pairs(train, test, (pair,), name)
        prediction = _ridge_predict(train, test, name)
        mae = _metrics(control_truth, prediction[control_subset])["mae"]
        delta = mae - full_mixed_mae
        pair_deltas.append((pair, delta))
        print(f"zero {str(pair):22s} MAE={mae:.6f} delta={delta:+.6f}")

    print("\n=== Control-heavy cumulative top-k zero-out ===\n")
    cumulative_predictions = {}
    for k in range(1, len(TOP_CONTROL_PAIRS) + 1):
        name = f"top{k}_zeroed"
        zero_mixed_pairs(train, test, TOP_CONTROL_PAIRS[:k], name)
        prediction = _ridge_predict(train, test, name)
        cumulative_predictions[k] = prediction
        mae = _metrics(control_truth, prediction[control_subset])["mae"]
        print(f"zero top {k}: MAE={mae:.6f} delta={mae - full_mixed_mae:+.6f}")

    print("\n=== Seat-wise Control-heavy stability ===\n")
    for seat in sorted({row["focal_seat"] for row in test}):
        indices = [
            i for i, row in enumerate(test)
            if row["focal_seat"] == seat and row["mixture_id"].startswith("ood-control_heavy-")
        ]
        seat_truth = truth[indices]
        baseline = _metrics(seat_truth, predictions["mixed_pair_features"][indices])["mae"]
        top1 = _metrics(seat_truth, cumulative_predictions[1][indices])["mae"]
        top2 = _metrics(seat_truth, cumulative_predictions[2][indices])["mae"]
        top3 = _metrics(seat_truth, cumulative_predictions[3][indices])["mae"]
        print(f"seat {seat}: baseline={baseline:.6f} top1={top1:.6f} ({top1-baseline:+.6f}) top2={top2:.6f} ({top2-baseline:+.6f}) top3={top3:.6f} ({top3-baseline:+.6f})")
    
    print("\n=== True matched-context Control-heavy probe ===")

    TOP_PAIRS = [
        ("bet", "call"),
        ("bet", "check"),
        ("call", "raise"),
    ]

    CONTEXTS = [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]

    for round_index, facing_bet in CONTEXTS:
        feature_name = (
            f"context_r{round_index}_"
            f"{'facing' if facing_bet else 'open'}"
        )

        # Build context-specific mixed-pair features.
        for dataset in (train, test):
            for example in dataset:
                block = mixed_pair_block_for_context(
                    example["_rows"],
                    round_index,
                    facing_bet,
                )

                example[feature_name] = np.concatenate(
                    [
                        example["M2_features"],
                        block,
                    ]
                )

        prediction = _ridge_predict(
            train,
            test,
            feature_name,
        )

        # Evaluate only Control-heavy examples.
        subset = [
            i
            for i, example in enumerate(test)
            if example["mixture_id"].startswith(
                "ood-control_heavy-"
            )
        ]

        subset_truth = truth[subset]
        baseline_prediction = prediction[subset]

        baseline = _metrics(
            subset_truth,
            baseline_prediction,
        )["mae"]

        print(
            f"\nround={round_index} "
            f"facing_bet={facing_bet}"
        )

        print(
            f"  context baseline "
            f"MAE={baseline:.6f}"
        )

        # ---------------------------------------------
        # Zero each top pair within this context only.
        # ---------------------------------------------

        for pair in TOP_PAIRS:
            zero_name = (
                feature_name
                + "_zero_"
                + "_".join(pair)
            )

            pair_index = MIXED_PAIR_CATEGORIES.index(pair)

            for dataset in (train, test):
                for example in dataset:
                    features = example[
                        feature_name
                    ].copy()

                    m2_len = len(
                        example["M2_features"]
                    )

                    features[
                        m2_len + pair_index
                    ] = 0.0

                    example[zero_name] = features

            zero_prediction = _ridge_predict(
                train,
                test,
                zero_name,
            )

            zero_result = _metrics(
                subset_truth,
                zero_prediction[subset],
            )

            delta = zero_result["mae"] - baseline

            print(
                f"  zero {str(pair):18s} "
                f"MAE={zero_result['mae']:.6f} "
                f"delta={delta:+.6f}"
            )

if __name__ == "__main__":
    main()
