from pathlib import Path

from pcc_poker.reproducibility import (
    build_reproducibility_manifest,
    sha256_file,
    write_reproducibility_manifest,
)


def test_sha256_file_is_stable(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("pcc\n", encoding="utf-8")
    first = sha256_file(p)
    second = sha256_file(p)
    assert first == second
    assert len(first) == 64


def test_manifest_records_missing_validation_without_guessing(tmp_path):
    (tmp_path / "pcc_poker").mkdir()
    (tmp_path / "pcc_poker" / "x.py").write_text("x = 1\n", encoding="utf-8")
    manifest = build_reproducibility_manifest(
        tmp_path,
        validation_files=("validation/does-not-exist.json",),
    )
    assert manifest["frozen_validation"]["complete"] is False
    assert manifest["reproducibility_ready"] is False
    assert manifest["frozen_validation"]["missing"] == ["validation/does-not-exist.json"]


def test_write_manifest_for_current_repo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "manifest.json"
    report = write_reproducibility_manifest(out, root=root, run_tests=False)
    assert out.is_file()
    assert report["frozen_validation"]["complete"] is True
    assert report["source"]["file_count"] > 10
    assert len(report["source"]["combined_sha256"]) == 64
