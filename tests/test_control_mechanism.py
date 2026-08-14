import tempfile
from collections import Counter
from pathlib import Path
import unittest

from pcc_poker.control_mechanism import (
    context_yoked_model,
    round_swapped_model,
    run_control_pressure_mechanism,
    write_control_pressure_mechanism,
)
from pcc_poker.policies import OpponentModel


def example_model():
    model = OpponentModel()
    model.context_actions["r0|facing"].update({"fold": 8, "call": 2})
    model.context_actions["r1|facing"].update({"fold": 1, "call": 9})
    model.context_actions["r0|open"].update({"check": 6, "bet": 4})
    return model


def action_totals(model):
    totals = Counter()
    for counts in model.context_actions.values():
        totals.update(counts)
    return totals


class ControlMechanismTests(unittest.TestCase):
    def test_round_swap_preserves_counts_but_reverses_timing(self):
        source = example_model()
        swapped = round_swapped_model(source)
        self.assertEqual(action_totals(source), action_totals(swapped))
        self.assertEqual(
            source.context_actions["r0|facing"],
            swapped.context_actions["r1|facing"],
        )

    def test_context_yoke_preserves_action_and_context_margins(self):
        source = example_model()
        yoked = context_yoked_model(source, seed=17)
        self.assertEqual(action_totals(source), action_totals(yoked))
        self.assertEqual(
            {key: sum(value.values()) for key, value in source.context_actions.items()},
            {key: sum(value.values()) for key, value in yoked.context_actions.items()},
        )
        for stratum in ("open", "facing"):
            source_totals = Counter()
            yoked_totals = Counter()
            for context, counts in source.context_actions.items():
                if context.endswith(f"|{stratum}"):
                    source_totals.update(counts)
            for context, counts in yoked.context_actions.items():
                if context.endswith(f"|{stratum}"):
                    yoked_totals.update(counts)
            self.assertEqual(source_totals, yoked_totals)

    def test_design_uses_fresh_seeds_and_matched_conditions(self):
        report = run_control_pressure_mechanism(
            replicates=2,
            calibration_hands_per_seat=4,
            evaluation_hands_per_seat=6,
            purities=(0.8,),
            temperatures=(0.35,),
        )
        self.assertEqual(
            set(report["design"]["conditions"]),
            {"aligned", "round_swapped", "context_yoked"},
        )
        self.assertEqual(report["design"]["focal_mode"], "control")
        self.assertEqual(report["design"]["targets"], ["pressure", "chaos"])
        self.assertTrue(report["design"]["common_random_numbers_within_comparison"])
        self.assertTrue(report["prespecified_checks"]["all_matched_margins_preserved"])
        self.assertEqual(len(report["condition_rows"]), 4)

    def test_writer_records_null_without_forcing_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanism.json"
            report = write_control_pressure_mechanism(
                path,
                replicates=2,
                calibration_hands_per_seat=4,
                evaluation_hands_per_seat=6,
                purities=(0.8,),
                temperatures=(0.35,),
            )
            self.assertTrue(path.exists())
            self.assertEqual(
                report["control_pressure_mechanism_confirmed"],
                all(report["prespecified_checks"].values()),
            )


if __name__ == "__main__":
    unittest.main()
