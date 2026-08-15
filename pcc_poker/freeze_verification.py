"""Verify the immutable v0.8.0 synthetic evidence freeze.

Verification is intentionally read-only: it never regenerates experiments,
rewrites manifests, or accesses human data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reproducibility import sha256_file

DEFAULT_MANIFEST = "validation/synthetic-freeze-manifest.json"
EXPECTED_VERSION = "0.8.0"
EXPECTED_LABEL = "synthetic-evidence-freeze"


def _check_entries(root: Path, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in entries:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append("manifest contains an invalid frozen-file entry")
            continue
        path = root / relative
        if not path.is_file():
            results.append({"path": relative, "status": "missing", "expected_sha256": expected})
            errors.append(f"missing frozen file: {relative}")
            continue
        observed = sha256_file(path)
        status = "ok" if observed == expected else "hash_mismatch"
        results.append({
            "path": relative,
            "status": status,
            "expected_sha256": expected,
            "observed_sha256": observed,
        })
        if status != "ok":
            errors.append(f"hash mismatch: {relative}")
    return results, errors


def verify_synthetic_freeze(root: str | Path = ".", manifest: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        return {"freeze_verified": False, "manifest": str(manifest_path), "errors": ["freeze manifest is missing"]}

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"freeze_verified": False, "manifest": str(manifest_path), "errors": [f"freeze manifest is unreadable: {exc}"]}

    errors: list[str] = []
    release = payload.get("release", {})
    if release.get("version") != EXPECTED_VERSION:
        errors.append(f"unexpected release version: {release.get('version')!r}")
    if release.get("label") != EXPECTED_LABEL:
        errors.append(f"unexpected release label: {release.get('label')!r}")
    if payload.get("synthetic_freeze_ready") is not True:
        errors.append("manifest does not declare synthetic_freeze_ready=true")
    if payload.get("human_data_gate", {}).get("confirmatory_human_analysis_allowed_now") is not False:
        errors.append("human-data gate is not closed")
    if payload.get("scientific_status", {}).get("confirmatory_human_axes") != ["pressure"]:
        errors.append("confirmatory human axes differ from the frozen pressure-only contract")

    artifact_results, artifact_errors = _check_entries(root, payload.get("frozen_artifacts", {}).get("files", []))
    protocol_results, protocol_errors = _check_entries(root, payload.get("frozen_protocols", {}).get("files", []))
    errors.extend(artifact_errors)
    errors.extend(protocol_errors)

    if not artifact_results:
        errors.append("manifest contains no frozen artifacts")
    if not protocol_results:
        errors.append("manifest contains no frozen protocols")

    return {
        "freeze_verified": not errors,
        "release": {"version": release.get("version"), "label": release.get("label")},
        "manifest": manifest_path.relative_to(root).as_posix() if manifest_path.is_relative_to(root) else str(manifest_path),
        "artifacts": artifact_results,
        "protocols": protocol_results,
        "errors": errors,
    }
