from src.effectiveness.states import build_agent_state_pair


def test_pair_uses_round_and_filters_value_history():
    trajectory = {
        "case_id": "case-1",
        "trajectory_id": "trajectory-1",
        "scenario": "patient vignette",
        "question": "question",
        "options": {"A": "one"},
        "max_agent_turns": 7,
        "turns": [
            {"speaker": "doctor", "text": "d1", "round": 1},
            {"speaker": "agent", "text": "a1", "round": 1, "reconstruction": {}},
            {"speaker": "doctor", "text": "d2", "round": 2},
            {"speaker": "agent", "text": "a2", "round": 2, "local_judgment": {}},
        ],
    }
    before, after = build_agent_state_pair(trajectory, 3)
    assert before["turn_id"] == 2
    assert before["max_agent_turns"] == 7
    assert after["remaining_agent_turns"] == 5
    assert set(after["history"][1]) == {"speaker", "text", "round"}
    assert "reconstruction" not in after["history"][1]
    assert "local_judgment" not in after["history"][-1]
