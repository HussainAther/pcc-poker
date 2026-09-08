from collections import Counter, defaultdict
from itertools import combinations_with_replacement

import numpy as np

from pcc_poker.analyze import load_jsonl
from pcc_poker.mixed import ACTIONS, _metrics, _ridge_predict
from pcc_poker.mixed_ood_ablation import _nested_features
from pcc_poker.policies import MODES


PAIR_CATEGORIES = tuple(
    combinations_with_replacement(sorted(ACTIONS), 2)
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

    repeat_rate = repeated / max(transitions, 1)

    mean_run = (
        float(np.mean(run_lengths))
        if run_lengths
        else 0.0
    )

    max_run = (
        float(np.max(run_lengths))
        if run_lengths
        else 0.0
    )

    return np.asarray(
        [
            repeat_rate,
            mean_run,
            max_run,
        ],
        dtype=float,
    )


def make_examples(records):
    grouped = defaultdict(list)

    for row in records:
        if row.get("is_focal_policy"):
            grouped[
                (row["mixture_id"], row["focal_seat"])
            ].append(row)

    examples = []

    for (mixture_id, focal_seat), rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda r: (
                r["hand_id"],
                r["decision_index"],
            ),
        )

        base = _nested_features(rows)

        by_hand = defaultdict(list)

        for row in rows:
            by_hand[row["hand_id"]].append(
                row["action"]
            )

        # -------------------------------------------------
        # Directionless adjacent-pair features
        # -------------------------------------------------

        pairs = []

        for actions in by_hand.values():
            for left, right in zip(
                actions,
                actions[1:],
            ):
                pairs.append(
                    tuple(sorted((left, right)))
                )

        counts = Counter(pairs)
        total = max(len(pairs), 1)

        unordered_block = np.asarray(
            [
                counts[pair] / total
                for pair in PAIR_CATEGORIES
            ],
            dtype=float,
        )

        # -------------------------------------------------
        # Persistence / repetition features
        # -------------------------------------------------

        persistence_block = persistence_features(rows)

        # -------------------------------------------------
        # Ground-truth PCC weights
        # -------------------------------------------------

        target = np.asarray(
            [
                rows[0]["target_pcc_weights"][mode]
                for mode in MODES
            ],
            dtype=float,
        )

        examples.append(
            {
                "mixture_id": mixture_id,
                "focal_seat": focal_seat,
                "target": target,

                "M2_features": base["M2"],

                "persistence_features": np.concatenate(
                    [
                        base["M2"],
                        persistence_block,
                    ]
                ),

                "unordered_features": np.concatenate(
                    [
                        base["M2"],
                        unordered_block,
                    ]
                ),

                "M3_features": base["M3"],
            }
        )

    return examples


def evaluate_model(
    train,
    test,
    feature_name,
):
    truth = np.stack(
        [
            row["target"]
            for row in test
        ]
    )

    prediction = _ridge_predict(
        train,
        test,
        feature_name,
    )

    metrics = _metrics(
        truth,
        prediction,
    )

    return prediction, metrics


# ---------------------------------------------------------
# Load frozen datasets
# ---------------------------------------------------------

train_records = load_jsonl(
    "outputs/mixed-recovery-data.jsonl"
)

test_records = load_jsonl(
    "outputs/mixed-ood-data.jsonl"
)

train = make_examples(train_records)
test = make_examples(test_records)

truth = np.stack(
    [
        row["target"]
        for row in test
    ]
)


MODELS = (
    "M2_features",
    "persistence_features",
    "unordered_features",
    "M3_features",
)


# ---------------------------------------------------------
# Overall comparison
# ---------------------------------------------------------

print("=== Overall ===\n")

predictions = {}

overall_mae = {}

for model in MODELS:
    prediction, result = evaluate_model(
        train,
        test,
        model,
    )

    predictions[model] = prediction
    overall_mae[model] = result["mae"]

    print(
        f"{model:22s} "
        f"MAE={result['mae']:.6f}"
    )


# ---------------------------------------------------------
# Overall decomposition
# ---------------------------------------------------------

print("\n=== Overall gains from M2 ===\n")

m2 = overall_mae["M2_features"]

for model in (
    "persistence_features",
    "unordered_features",
    "M3_features",
):
    gain = m2 - overall_mae[model]

    print(
        f"{model:22s} "
        f"gain={gain:.6f}"
    )


# ---------------------------------------------------------
# Regime-specific comparison
# ---------------------------------------------------------

regimes = {
    "balanced": "ood-balanced-",
    "boundary": "ood-boundary-",
    "chaos-heavy": "ood-chaos_heavy-",
    "control-heavy": "ood-control_heavy-",
    "pressure-heavy": "ood-pressure_heavy-",
}

print("\n=== By regime ===")

for label, prefix in regimes.items():
    subset = [
        i
        for i, row in enumerate(test)
        if row["mixture_id"].startswith(prefix)
    ]

    if not subset:
        continue

    subset_truth = truth[subset]

    print(f"\n{label}")

    regime_mae = {}

    for model in MODELS:
        subset_prediction = predictions[model][
            subset
        ]

        result = _metrics(
            subset_truth,
            subset_prediction,
        )

        regime_mae[model] = result["mae"]

        print(
            f"  {model:22s} "
            f"MAE={result['mae']:.6f}"
        )

    # -----------------------------------------------------
    # Gains relative to M2
    # -----------------------------------------------------

    m2 = regime_mae["M2_features"]

    persistence_gain = (
        m2
        - regime_mae["persistence_features"]
    )

    unordered_gain = (
        m2
        - regime_mae["unordered_features"]
    )

    m3_gain = (
        m2
        - regime_mae["M3_features"]
    )

    directional_increment = (
        regime_mae["unordered_features"]
        - regime_mae["M3_features"]
    )

    print()

    print(
        f"  {'M2 -> persistence':22s} "
        f"gain={persistence_gain:.6f}"
    )

    print(
        f"  {'M2 -> unordered':22s} "
        f"gain={unordered_gain:.6f}"
    )

    print(
        f"  {'M2 -> M3':22s} "
        f"gain={m3_gain:.6f}"
    )

    print(
        f"  {'unordered -> M3':22s} "
        f"gain={directional_increment:.6f}"
    )

    # -----------------------------------------------------
    # Mechanism shares
    #
    # Only print percentages when M3 actually improves
    # over M2. Otherwise percentages become misleading.
    # -----------------------------------------------------

    if m3_gain > 0:
        persistence_share = (
            persistence_gain / m3_gain
        )

        unordered_share = (
            unordered_gain / m3_gain
        )

        directional_share = (
            directional_increment / m3_gain
        )

        print(
            f"  {'persistence share':22s} "
            f"={100 * persistence_share:.1f}%"
        )

        print(
            f"  {'unordered share':22s} "
            f"={100 * unordered_share:.1f}%"
        )

        print(
            f"  {'directional share':22s} "
            f"={100 * directional_share:.1f}%"
        )
    else:
        print(
            "  M3 does not improve over M2; "
            "mechanism shares not reported."
        )
