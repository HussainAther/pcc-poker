"""Read-only release hygiene checks for the v0.8.0 synthetic freeze.

These checks validate packaging/release metadata around the frozen scientific
bundle. They do not regenerate experiments, rewrite frozen artifacts, or access
human data.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .freeze_verification import verify_synthetic_freeze

EXPECTED_VERSION = "0.8.0"
REQUIRED_RELEASE_FILES = (
    "CHANGELOG.md",
    "docs/RELEASE_NOTES_v0.8.0.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/SYNTHETIC_EVIDENCE_FREEZE.md",
    "docs/HUMAN_ANALYSIS_PREREGISTRATION.md",
    "validation/synthetic-freeze-manifest.json",
)


def _pyproject_version(root: Path) -> str | None:
    path = root / "pyproject.toml"
    if not path.is_file():
        return None
    in_project = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _git_diff_check(root: Path) -> dict[str, Any]:
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return {"available": False, "passed": True, "output": "git not available"}
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return {"available": False, "passed": True, "output": "not a git work tree; diff check skipped"}
    result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "available": True,
        "passed": result.returncode == 0,
        "output": result.stdout.strip(),
    }


def run_release_check(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    freeze = verify_synthetic_freeze(root)
    project_version = _pyproject_version(root)
    missing = [relative for relative in REQUIRED_RELEASE_FILES if not (root / relative).is_file()]
    diff_check = _git_diff_check(root)

    checks = {
        "freeze_verified": bool(freeze.get("freeze_verified")),
        "package_version_matches_freeze": __version__ == EXPECTED_VERSION,
        "pyproject_version_matches_freeze": project_version == EXPECTED_VERSION,
        "package_and_pyproject_versions_match": __version__ == project_version,
        "required_release_files_present": not missing,
        "git_diff_check_passed": bool(diff_check["passed"]),
        "human_data_gate_closed": freeze.get("freeze_verified") is True,
    }

    return {
        "release_check_passed": all(checks.values()),
        "expected_version": EXPECTED_VERSION,
        "package_version": __version__,
        "pyproject_version": project_version,
        "checks": checks,
        "missing_release_files": missing,
        "git_diff_check": diff_check,
        "freeze_errors": freeze.get("errors", []),
        "note": "Read-only release hygiene check; no experiments or human data are accessed.",
    }
