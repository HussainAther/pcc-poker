"""ORIA-safe preflight for the future HandHQ ingestion pipeline.

This module is deliberately limited to sentinel-marked synthetic fixtures in
``tests/fixtures``.  It provides an auditable way to exercise schema parsing,
identifier minimization, prohibited-field handling, decision-row leakage
checks, and output isolation while the human-data gate is closed.

It must not be used to process HandHQ or other human-source records before the
applicable institutional determination/approval permits that work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .freeze_verification import verify_synthetic_freeze
from .handhq import (
    FORBIDDEN_SOURCE_FIELDS,
    OUTCOME_FIELDS,
    _parse_blocks,
    assert_no_forbidden_values,
    decision_rows,
    ingest_phhs_text,
)

SYNTHETIC_SENTINEL = "# SYNTHETIC_FIXTURE_ONLY"
DEFAULT_FIXTURE = "tests/fixtures/mock_handhq_oria.phhs"
DEFAULT_OUTPUT = "build/audit/oria-ingestion-preflight.json"

# Fields documented in the planned ORIA field inventory for the HandHQ PHH
# representation. Unknown fields are reported rather than silently accepted as
# part of the planned analytic contract.
PLANNED_SOURCE_FIELDS = frozenset(
    {
        "variant",
        "ante_trimming_status",
        "antes",
        "blinds_or_straddles",
        "min_bet",
        "starting_stacks",
        "actions",
        "venue",
        "time",
        "day",
        "month",
        "year",
        "hand",
        "seats",
        "table",
        "players",
        "winnings",
        "currency_symbol",
        "time_zone_abbreviation",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _closed_human_gate(root: Path) -> tuple[bool, dict[str, Any]]:
    freeze = verify_synthetic_freeze(root)
    if not freeze.get("freeze_verified"):
        return False, freeze
    manifest_path = root / "validation" / "synthetic-freeze-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        freeze = dict(freeze)
        freeze.setdefault("errors", []).append("Unable to read synthetic freeze manifest human-data gate.")
        return False, freeze
    allowed = manifest.get("human_data_gate", {}).get("confirmatory_human_analysis_allowed_now")
    if allowed is not False:
        freeze = dict(freeze)
        freeze.setdefault("errors", []).append("Frozen human-data gate is not explicitly closed.")
        return False, freeze
    return True, freeze


def _blocked_report(root: Path, input_path: Path, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "oria_preflight_passed": False,
        "input_allowed": False,
        "human_data_gate_closed": True,
        "input": {
            "path": input_path.as_posix(),
            "content_read": False,
        },
        "checks": {
            "synthetic_fixture_location": False,
            "synthetic_sentinel_present": False,
            "schema_matches_planned_inventory": False,
            "identifiers_scrubbed": False,
            "forbidden_values_absent": False,
            "outcomes_excluded_from_decision_rows": False,
            "private_cards_excluded": False,
            "audit_output_isolated": True,
        },
        "errors": [reason],
        "note": "Blocked before reading input contents. Human-source HandHQ analysis remains closed pending ORIA/IRB guidance.",
    }


def run_oria_ingestion_preflight(
    root: str | Path = ".",
    *,
    input_path: str | Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    """Run the safe preflight against one approved synthetic fixture.

    Any input outside ``tests/fixtures`` is rejected *before the file is read*.
    A file inside the fixture directory must also start with the explicit
    synthetic sentinel.
    """

    root = Path(root).resolve()
    candidate = Path(input_path)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    fixture_root = (root / "tests" / "fixtures").resolve()

    gate_closed, freeze = _closed_human_gate(root)
    if not gate_closed:
        return {
            "schema_version": 1,
            "oria_preflight_passed": False,
            "input_allowed": False,
            "human_data_gate_closed": False,
            "input": {"path": candidate.as_posix(), "content_read": False},
            "checks": {},
            "errors": ["Synthetic freeze verification failed; safe preflight aborted."],
            "freeze_errors": freeze.get("errors", []),
        }

    if not _is_within(candidate, fixture_root):
        return _blocked_report(
            root,
            candidate,
            "Input rejected: ORIA-safe preflight accepts only files inside tests/fixtures while the human-data gate is closed.",
        )
    if not candidate.is_file():
        return _blocked_report(root, candidate, "Synthetic fixture does not exist.")

    data = candidate.read_bytes()
    text = data.decode("utf-8")
    sentinel_present = text.startswith(SYNTHETIC_SENTINEL)
    if not sentinel_present:
        report = _blocked_report(
            root,
            candidate,
            "Input rejected: synthetic fixture sentinel is missing.",
        )
        report["input"]["content_read"] = True
        report["checks"]["synthetic_fixture_location"] = True
        return report

    blocks = _parse_blocks(text)
    source_fields = sorted(
        {key for block in blocks for key in block if not key.startswith("__")}
    )
    unknown_fields = sorted(set(source_fields) - PLANNED_SOURCE_FIELDS)
    missing_core_fields = sorted(
        {
            "variant",
            "antes",
            "blinds_or_straddles",
            "min_bet",
            "starting_stacks",
            "actions",
            "seats",
            "players",
        }
        - set(source_fields)
    )

    # Fixed test key is appropriate only because this pathway is synthetic-only.
    hands = ingest_phhs_text(
        text,
        pseudonymization_key=b"oria-preflight-synthetic-only-key",
        retain_outcome=True,
    )
    rows = tuple(row for hand in hands for row in decision_rows(hand))

    forbidden_values_absent = True
    try:
        for block, hand in zip(blocks, hands, strict=True):
            assert_no_forbidden_values(block, (hand, *decision_rows(hand)))
    except AssertionError:
        forbidden_values_absent = False

    identifiers_scrubbed = all(
        player_id.startswith("P") and len(player_id) == 17
        for hand in hands
        for player_id in hand.player_ids
    )
    private_cards_excluded = all("private_deal" not in repr(row) for row in rows)
    outcomes_excluded = all(not hasattr(row, "outcome") for row in rows)

    checks = {
        "synthetic_fixture_location": True,
        "synthetic_sentinel_present": sentinel_present,
        "schema_matches_planned_inventory": not unknown_fields and not missing_core_fields,
        "identifiers_scrubbed": identifiers_scrubbed,
        "forbidden_values_absent": forbidden_values_absent,
        "outcomes_excluded_from_decision_rows": outcomes_excluded,
        "private_cards_excluded": private_cards_excluded,
        "audit_output_isolated": True,
    }

    return {
        "schema_version": 1,
        "oria_preflight_passed": all(checks.values()),
        "input_allowed": True,
        "human_data_gate_closed": True,
        "input": {
            "path": candidate.relative_to(root).as_posix(),
            "content_read": True,
            "sha256": _sha256_bytes(data),
            "bytes": len(data),
            "synthetic_fixture": True,
        },
        "schema": {
            "observed_source_fields": source_fields,
            "planned_source_fields": sorted(PLANNED_SOURCE_FIELDS),
            "forbidden_source_fields": sorted(FORBIDDEN_SOURCE_FIELDS),
            "outcome_fields": sorted(OUTCOME_FIELDS),
            "unknown_fields": unknown_fields,
            "missing_core_fields": missing_core_fields,
        },
        "counts": {
            "source_records": len(blocks),
            "sanitized_hands": len(hands),
            "decision_rows": len(rows),
            "study_player_ids": sum(len(hand.player_ids) for hand in hands),
        },
        "checks": checks,
        "errors": [],
        "note": "Synthetic/mock-only ingestion preflight. No human HandHQ data were accessed or analyzed.",
    }


def write_oria_ingestion_preflight(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    root: str | Path = ".",
    input_path: str | Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    root = Path(root).resolve()
    report = run_oria_ingestion_preflight(root, input_path=input_path)
    target = Path(output)
    if not target.is_absolute():
        target = root / target
    # Output is intentionally limited to build/audit so it cannot modify the
    # frozen validation bundle.
    audit_root = (root / "build" / "audit").resolve()
    target = target.resolve()
    if not _is_within(target, audit_root):
        raise ValueError("ORIA preflight output must be under build/audit/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
