from pcc_poker.control_matched_state_decomposition import run_control_matched_state_decomposition

def test_matched_state_decomposition_confirms_two_sided_context_gap():
    r=run_control_matched_state_decomposition()
    assert r['status']=='confirmed'
    assert r['minimal_architectural_difference_supported'] is True
    assert all(r['prespecified_checks'].values())
    assert r['policy_modified'] is False
    assert r['human_data_accessed'] is False
