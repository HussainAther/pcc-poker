import csv
from pathlib import Path
import tempfile
import unittest

from pcc_poker.robustness import run_robustness_grid, write_robustness_outputs
from pcc_poker.simulate import mode_mixture


class RobustnessTests(unittest.TestCase):
    def test_mode_mixture_places_requested_purity_on_axis(self):
        mixture = mode_mixture("control", 0.8)
        self.assertAlmostEqual(sum(mixture), 1.0)
        self.assertAlmostEqual(mixture[1], 0.8)
        self.assertAlmostEqual(mixture[0], mixture[2])

    def test_grid_crosses_parameters_and_keeps_policy_frozen(self):
        report = run_robustness_grid(
            temperatures=(0.25, 0.5),
            purities=(0.7,),
            hand_counts=(5, 7),
            replicates=2,
            seed=601,
            workers=1,
        )
        self.assertEqual(report["design"]["conditions"], 4)
        self.assertEqual(report["design"]["total_sweeps"], 8)
        self.assertFalse(report["design"]["policies_modified"])
        self.assertEqual(len(report["condition_results"]), 4)
        self.assertTrue(all(len(row["runs"]) == 2 for row in report["condition_results"]))
        self.assertEqual(
            set(report["stratified"]),
            {"temperature", "mode_purity", "hands_per_seat_order"},
        )
        self.assertEqual(
            len(report["failure_conditions"]),
            report["design"]["conditions"]
            - report["aggregate"]["cycle_conditions"],
        )

    def test_writer_emits_one_csv_row_per_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "grid.json"
            csv_path = Path(directory) / "grid.csv"
            report = write_robustness_outputs(
                json_path,
                csv_path,
                temperatures=(0.35,),
                purities=(0.8,),
                hand_counts=(5,),
                replicates=2,
                seed=701,
                workers=1,
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(json_path.exists())
            self.assertEqual(len(rows), report["design"]["conditions"])
            self.assertIn("pressure_over_chaos_ci_low", rows[0])


if __name__ == "__main__":
    unittest.main()
