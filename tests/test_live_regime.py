from pcc_poker.engine import apply_action, initial_state
from pcc_poker.live_regime import LiveRegimeTracker, format_regime_panel


def test_tracker_starts_mixed():
    tracker = LiveRegimeTracker(window=6)
    snap = tracker.rolling()
    assert snap.label == "MIXED"
    assert snap.decisions == 0


def test_bet_creates_control_like_evidence_in_opening_state():
    tracker = LiveRegimeTracker(window=6)
    state = initial_state([2, 0, 1, 0, 1, 2])
    evidence = tracker.observe(state, "bet")
    assert evidence.control > evidence.pressure
    assert 0.0 <= evidence.chaos <= 1.0


def test_facing_wager_registers_pressure_without_hindsight():
    tracker = LiveRegimeTracker(window=6)
    state = initial_state([0, 2, 1, 0, 1, 2])
    state = apply_action(state, "bet")
    evidence = tracker.observe(state, "fold")
    assert evidence.pressure > 0.0
    assert evidence.faced_pressure > 0.0


def test_rolling_window_is_bounded_and_panel_renders():
    tracker = LiveRegimeTracker(window=3)
    state = initial_state([2, 0, 1, 0, 1, 2])
    for _ in range(5):
        tracker.observe(state, "check")
    snap = tracker.rolling()
    assert snap.decisions == 3
    lines = format_regime_panel(snap)
    assert any("Pressure" in line for line in lines)
    assert any("Regime:" in line for line in lines)
