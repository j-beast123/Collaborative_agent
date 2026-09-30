import pytest

from src.effectiveness.states import build_agent_state_pair


def test_pair_rejects_mismatched_scenarios():
    trajectory = {
        "case_id": "case-1",
        "trajectory_id": "trajectory-1",
        "scenario": "source scenario",
        "question": "question",
        "options": {"A": "one"},
        "turns": [
            {"speaker": "doctor", "text": "d1", "round": 1},
            {"speaker": "agent", "text": "a1", "round": 1},
        ],
    }
    before, after = build_agent_state_pair(trajectory, 1)
    after["scenario"] = "different scenario"
    with pytest.raises(ValueError, match="scenario"):
        from src.effectiveness.states import validate_pair

        validate_pair(before, after)
