from pathlib import Path

from pcc_poker.synthetic_freeze import build_synthetic_freeze_manifest, write_synthetic_freeze_manifest


def test_synthetic_freeze_uses_conservative_pressure_only_panel():
    root = Path(__file__).resolve().parents[1]
    report = build_synthetic_freeze_manifest(root)
    assert report["scientific_status"]["confirmatory_human_axes"] == ["pressure"]
    assert set(report["scientific_status"]["pressure_components"]) == {
        "pressure_exposure",
        "predicted_fold_probability",
    }
    assert report["scientific_status"]["control_status"] == "exploratory/unresolved"
    assert report["scientific_status"]["chaos_status"] == "exploratory/unresolved"
    assert report["human_data_gate"]["confirmatory_human_analysis_allowed_now"] is False


def test_synthetic_freeze_manifest_is_complete(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "freeze.json"
    report = write_synthetic_freeze_manifest(out, root=root)
    assert out.is_file()
    assert report["frozen_artifacts"]["missing"] == []
    assert report["frozen_protocols"]["missing"] == []
    assert report["seed_inventory"]
