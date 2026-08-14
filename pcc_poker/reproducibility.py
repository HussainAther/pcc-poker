from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import __version__


DEFAULT_VALIDATION_FILES = (
    "validation/control-pressure-mechanism.json",
    "validation/effective-chaos-validation.json",
    "validation/chaos-control-decomposition.json",
    "validation/pressure-surprise-decomposition.json",
    "validation/family-invariant-panel.json",
    "validation/contextual-control-observable.json",
)

SOURCE_GLOBS = (
    "pcc_poker/*.py",
    "tests/test_*.py",
    "docs/*PROTOCOL.md",
    "pyproject.toml",
    "requirements.txt",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _collect_paths(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def _hash_entries(root: Path, paths: Iterable[Path]) -> list[dict]:
    entries = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        entries.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return entries


def _combined_hash(entries: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for entry in entries:
        h.update(entry["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(entry["sha256"].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def run_pytest(root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout.strip()
    last_line = output.splitlines()[-1] if output else ""
    return {
        "executed": True,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "summary": last_line,
        "output": output,
    }


def build_reproducibility_manifest(
    root: str | Path = ".",
    *,
    validation_files: Iterable[str] = DEFAULT_VALIDATION_FILES,
    run_tests: bool = False,
) -> dict:
    root = Path(root).resolve()
    requested = [root / path for path in validation_files]
    present = [path for path in requested if path.is_file()]
    missing = [path.relative_to(root).as_posix() for path in requested if not path.is_file()]

    validation_entries = _hash_entries(root, present)
    source_entries = _hash_entries(root, _collect_paths(root, SOURCE_GLOBS))

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproducibility fingerprint for frozen synthetic PCC Poker validation artifacts; not human-data evidence.",
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "pcc_poker_version": _package_version("pcc-poker") or __version__,
            "numpy_version": _package_version("numpy"),
            "pytest_version": _package_version("pytest"),
            "git_commit": _git_commit(root),
        },
        "source": {
            "files": source_entries,
            "combined_sha256": _combined_hash(source_entries),
            "file_count": len(source_entries),
        },
        "frozen_validation": {
            "files": validation_entries,
            "combined_sha256": _combined_hash(validation_entries),
            "requested_count": len(requested),
            "present_count": len(present),
            "missing": missing,
            "complete": not missing,
        },
        "tests": {"executed": False},
        "reproducibility_ready": not missing,
    }

    if run_tests:
        manifest["tests"] = run_pytest(root)
        manifest["reproducibility_ready"] = manifest["reproducibility_ready"] and manifest["tests"]["passed"]

    return manifest


def write_reproducibility_manifest(
    path: str | Path,
    *,
    root: str | Path = ".",
    run_tests: bool = False,
) -> dict:
    manifest = build_reproducibility_manifest(root, run_tests=run_tests)
    target = Path(path)
    if not target.is_absolute():
        target = Path(root) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
