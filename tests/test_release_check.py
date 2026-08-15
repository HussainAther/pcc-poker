from pathlib import Path
import unittest

from pcc_poker.release_check import run_release_check


class ReleaseCheckTests(unittest.TestCase):
    def test_repository_release_check_passes(self):
        root = Path(__file__).resolve().parents[1]
        report = run_release_check(root)
        self.assertTrue(report["release_check_passed"], report)
        self.assertEqual(report["package_version"], "0.8.0")
        self.assertEqual(report["pyproject_version"], "0.8.0")
        self.assertFalse(report["missing_release_files"])


if __name__ == "__main__":
    unittest.main()
