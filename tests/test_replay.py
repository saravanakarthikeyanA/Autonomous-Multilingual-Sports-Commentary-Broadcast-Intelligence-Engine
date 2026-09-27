"""
Unit tests for Stateful Cricket Replay Engine.
"""

from engine.replay_engine import ReplayEngine


def test_preload_state():
    engine = ReplayEngine()
    state = engine.get_state_snapshot()

    # State before Over 18.0
    assert state["inning"] == 1
    assert state["batting_team"] == "England"
    assert state["score"] == 190
    assert state["wickets"] == 5
    assert state["overs_display"] == "18.0"


def test_step_progression():
    engine = ReplayEngine()
    step1 = engine.step()

    assert step1 is not None
    d = step1["delivery"]
    assert d["over"] == 18
    assert d["ball"] == 1
    assert d["batter"] == "LS Livingstone"
    assert d["bowler"] == "HV Patel"
    assert d["runs_total"] == 1

    state = step1["state"]
    assert state["score"] == 191


def test_stream_delivery_count():
    engine = ReplayEngine()
    assert len(engine.stream_deliveries) == 30
