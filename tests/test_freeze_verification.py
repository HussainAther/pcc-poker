import json
import shutil
import tempfile
from pathlib import Path
import unittest

from pcc_poker.freeze_verification import verify_synthetic_freeze


class FreezeVerificationTests(unittest.TestCase):
    def test_repository_freeze_verifies(self):
        root = Path(__file__).resolve().parents[1]
        report = verify_synthetic_freeze(root)
        self.assertTrue(report["freeze_verified"], report["errors"])

    def test_tampering_is_detected(self):
        source = Path(__file__).resolve().parents[1]
        manifest = json.loads((source / "validation/synthetic-freeze-manifest.json").read_text())
        target_rel = manifest["frozen_artifacts"]["files"][0]["path"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation").mkdir()
            (root / "docs").mkdir()
            shutil.copy2(source / "validation/synthetic-freeze-manifest.json", root / "validation/synthetic-freeze-manifest.json")
            for section in ("frozen_artifacts", "frozen_protocols"):
                for entry in manifest[section]["files"]:
                    dst = root / entry["path"]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / entry["path"], dst)
            with (root / target_rel).open("ab") as handle:
                handle.write(b"\nTAMPERED\n")
            report = verify_synthetic_freeze(root)
            self.assertFalse(report["freeze_verified"])
            self.assertTrue(any("hash mismatch" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
