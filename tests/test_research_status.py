from pathlib import Path

from pcc_poker.research_status import build_research_status, write_research_status


def test_current_repo_status_has_all_four_status_classes():
    root = Path(__file__).resolve().parents[1]
    report = build_research_status(root)
    statuses = {row["status"] for row in report["claims"]}
    assert {"confirmed", "partial", "failed", "unresolved"}.issubset(statuses)
    assert report["missing_sources"] == []


def test_known_frozen_claims_are_not_overstated():
    root = Path(__file__).resolve().parents[1]
    report = build_research_status(root)
    by_id = {row["claim_id"]: row for row in report["claims"]}
    assert by_id["control-pressure-mechanism"]["status"] == "confirmed"
    assert by_id["contextual-control-observable"]["status"] == "partial"
    assert by_id["family-invariant-control-panel"]["status"] == "unresolved"
    assert by_id["family-invariant-chaos-panel"]["status"] == "unresolved"
    assert by_id["implementation-invariant-supervised-coordinates"]["status"] == "failed"


def test_writer_emits_json_csv_and_markdown(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = write_research_status(
        root=root,
        json_output=tmp_path / "status.json",
        csv_output=tmp_path / "status.csv",
        markdown_output=tmp_path / "status.md",
    )
    assert (tmp_path / "status.json").is_file()
    assert (tmp_path / "status.csv").is_file()
    assert (tmp_path / "status.md").is_file()
    text = (tmp_path / "status.md").read_text(encoding="utf-8")
    assert "Frozen synthetic evidence only" in text
    assert len(report["claims"]) >= 10
