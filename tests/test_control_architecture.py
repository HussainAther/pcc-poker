from pcc_poker.control_architecture import run_control_architecture_export


def test_control_architecture_schema_and_seed_boundary():
    report = run_control_architecture_export(
        agents=8,
        calibration_mixtures=8,
        calibration_hands_per_seat=8,
        signature_replicates=1,
        outcome_replicates=1,
        hands_per_seat=8,
        seed=77,
    )
    assert report["game"] == "poker"
    assert report["design"]["signature_outcome_seeds_disjoint"] is True
    assert report["design"]["latent_pcc_weights_used_as_regression_predictors"] is False
    assert set(report["interaction_improvements"]) == {"pressure", "control", "chaos"}
    assert len(report["rows"]) == 8 * 4


def test_control_architecture_has_cross_game_normalized_fields():
    report = run_control_architecture_export(
        agents=8,
        calibration_mixtures=8,
        calibration_hands_per_seat=8,
        signature_replicates=1,
        outcome_replicates=1,
        hands_per_seat=8,
        seed=91,
    )
    for key in ("additive_standardized_mae", "control_context_standardized_mae", "relative_improvement", "primary_pass"):
        assert key in report
