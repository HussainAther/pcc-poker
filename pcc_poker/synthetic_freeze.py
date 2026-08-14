"""Build the pre-human synthetic evidence freeze manifest.

This module does not run synthetic experiments or access human data. It hashes
already-frozen artifacts and records the conservative measurement panel that is
eligible for the preregistered human phase.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reproducibility import sha256_file

RELEASE_VERSION = "0.8.0"
RELEASE_LABEL = "synthetic-evidence-freeze"

FROZEN_ARTIFACTS = (
    "validation/balanced-cycle.json",
    "validation/robustness-grid.json",
    "validation/control-pressure-mechanism.json",
    "validation/counterfactual-control.json",
    "validation/temporal-control.json",
    "validation/contextual-control-observable.json",
    "validation/effective-chaos-validation.json",
    "validation/chaos-control-decomposition.json",
    "validation/pressure-surprise-decomposition.json",
    "validation/family-invariant-panel.json",
    "validation/family-transfer-grid-summary.json",
    "validation/mixed-recovery.json",
    "validation/research-status.json",
    "validation/reproducibility-manifest.json",
)

FROZEN_PROTOCOLS = (
    "docs/MEASUREMENT_CONTRACT.md",
    "docs/FAMILY_INVARIANT_PANEL_PROTOCOL.md",
    "docs/HUMAN_DATA_INGESTION_PROTOCOL.md",
    "docs/HUMAN_PCC_OBSERVABLES_PROTOCOL.md",
    "docs/HUMAN_MEASUREMENT_CONTRACT.md",
    "docs/HUMAN_ANALYSIS_PREREGISTRATION.md",
    "docs/SYNTHETIC_EVIDENCE_FREEZE.md",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_inventory(value: Any, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if "seed" in key.lower() and isinstance(child, (int, float, str, list, tuple, dict)):
                found[name] = child
            found.update(_seed_inventory(child, name))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.update(_seed_inventory(child, f"{prefix}[{i}]"))
    return found


def _hash_entries(root: Path, relative_paths: tuple[str, ...]) -> tuple[list[dict], list[str]]:
    entries: list[dict] = []
    missing: list[str] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return entries, missing


def build_synthetic_freeze_manifest(root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    artifacts, missing_artifacts = _hash_entries(root, FROZEN_ARTIFACTS)
    protocols, missing_protocols = _hash_entries(root, FROZEN_PROTOCOLS)

    status_path = root / "validation/research-status.json"
    panel_path = root / "validation/family-invariant-panel.json"
    repro_path = root / "validation/reproducibility-manifest.json"
    status = _load_json(status_path) if status_path.is_file() else {}
    panel = _load_json(panel_path) if panel_path.is_file() else {}
    repro = _load_json(repro_path) if repro_path.is_file() else {}

    seeds: dict[str, Any] = {}
    for entry in artifacts:
        path = root / entry["path"]
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        per_file = _seed_inventory(payload)
        if per_file:
            seeds[entry["path"]] = per_file

    selected = panel.get("selected_invariant_components", [])
    axis_coverage = panel.get("axis_coverage", {})
    pressure_components = list(axis_coverage.get("pressure", []))
    control_components = list(axis_coverage.get("control", []))
    chaos_components = list(axis_coverage.get("chaos", []))

    gates = {
        "all_frozen_artifacts_present": not missing_artifacts,
        "all_freeze_protocols_present": not missing_protocols,
        "reproducibility_ready": bool(repro.get("reproducibility_ready")),
        "human_confirmatory_panel_is_pressure_only": bool(pressure_components) and not control_components and not chaos_components,
    }

    return {
        "schema_version": 1,
        "release": {
            "version": RELEASE_VERSION,
            "label": RELEASE_LABEL,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Freeze synthetic PCC evidence and the pre-human analysis contract before any confirmatory HandHQ analysis.",
        },
        "human_data_gate": {
            "confirmatory_human_analysis_allowed_now": False,
            "reason": "Requires the applicable Georgia Tech ORIA/IRB determination or approval before confirmatory human-data analysis begins.",
            "source_scope": "Future analysis is restricted to the approved HandHQ online-hand-history subset; televised WSOP, Pluribus, and named historical examples are excluded.",
        },
        "scientific_status": {
            "claim_counts": status.get("counts", {}),
            "selected_family_invariant_components": selected,
            "confirmatory_human_axes": ["pressure"] if pressure_components else [],
            "pressure_components": pressure_components,
            "control_status": "exploratory/unresolved" if not control_components else "eligible",
            "chaos_status": "exploratory/unresolved" if not chaos_components else "eligible",
            "rule": "Human results cannot change synthetic thresholds, component definitions, or claim statuses in this freeze. Any later change requires a new version and documented amendment before looking at the affected confirmatory endpoint.",
        },
        "frozen_artifacts": {
            "files": artifacts,
            "missing": missing_artifacts,
        },
        "frozen_protocols": {
            "files": protocols,
            "missing": missing_protocols,
        },
        "seed_inventory": seeds,
        "reproducibility_fingerprints": {
            "source_combined_sha256": repro.get("source", {}).get("combined_sha256"),
            "frozen_validation_combined_sha256": repro.get("frozen_validation", {}).get("combined_sha256"),
        },
        "release_gates": gates,
        "synthetic_freeze_ready": all(gates.values()),
    }


def write_synthetic_freeze_manifest(path: str | Path, *, root: str | Path = ".") -> dict:
    report = build_synthetic_freeze_manifest(root)
    target = Path(path)
    if not target.is_absolute():
        target = Path(root) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
