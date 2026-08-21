from pcc_poker.score_control_decomposition import run_score_control_decomposition

def test_score_control_mechanism_split_is_detected():
    r=run_score_control_decomposition()
    assert r["mechanism_split_confirmed"], r
    assert r["summary"]["adaptive_mean_total_variation_shift"] > r["summary"]["score_mean_total_variation_shift"]
    assert r["policy_modified"] is False
    assert r["human_data_accessed"] is False
