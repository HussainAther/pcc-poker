import json
from pathlib import Path
import tempfile
import unittest

from pcc_poker.oria_preflight import (
    DEFAULT_FIXTURE,
    run_oria_ingestion_preflight,
    write_oria_ingestion_preflight,
)


class ORIAIngestionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_repository_mock_fixture_passes(self):
        report = run_oria_ingestion_preflight(self.root)
        self.assertTrue(report["oria_preflight_passed"], report)
        self.assertTrue(report["human_data_gate_closed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertGreater(report["counts"]["decision_rows"], 0)
        self.assertEqual(report["schema"]["unknown_fields"], [])

    def test_arbitrary_external_path_is_blocked_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "handhq-real-looking.phhs"
            candidate.write_text("THIS MUST NOT BE READ\n", encoding="utf-8")
            report = run_oria_ingestion_preflight(self.root, input_path=candidate)
        self.assertFalse(report["oria_preflight_passed"])
        self.assertFalse(report["input_allowed"])
        self.assertFalse(report["input"]["content_read"])

    def test_fixture_without_sentinel_is_rejected(self):
        fixture = self.root / "tests" / "fixtures" / "temporary-no-sentinel.phhs"
        try:
            fixture.write_text("[1]\nvariant = 'NT'\n", encoding="utf-8")
            report = run_oria_ingestion_preflight(self.root, input_path=fixture)
        finally:
            fixture.unlink(missing_ok=True)
        self.assertFalse(report["oria_preflight_passed"])
        self.assertTrue(report["input"]["content_read"])
        self.assertFalse(report["checks"]["synthetic_sentinel_present"])

    def test_audit_output_must_stay_under_build_audit(self):
        with self.assertRaises(ValueError):
            write_oria_ingestion_preflight(
                "validation/oria-preflight.json",
                root=self.root,
                input_path=DEFAULT_FIXTURE,
            )

    def test_written_audit_contains_no_raw_mock_identifiers(self):
        with tempfile.TemporaryDirectory(dir=self.root / "build" if (self.root / "build").exists() else self.root) as tmp:
            # Use canonical build/audit target instead; temporary directory may
            # not be nested under audit.
            pass
        target = self.root / "build" / "audit" / "test-oria-preflight.json"
        try:
            report = write_oria_ingestion_preflight(target, root=self.root)
            raw = target.read_text(encoding="utf-8")
        finally:
            target.unlink(missing_ok=True)
        self.assertTrue(report["oria_preflight_passed"])
        self.assertNotIn("mock-alice-source-id", raw)
        self.assertNotIn("MOCK TABLE ONLY", raw)
        self.assertNotIn("999000111", raw)


if __name__ == "__main__":
    unittest.main()
