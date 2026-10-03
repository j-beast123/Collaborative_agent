"""Causal validation with a frozen intervention: one compliant response per candidate, the same text in every rerun."""
import json

from src.judges.rubric_validation import GENERATION_FAILURE, load_validations, passed_records, validate_candidates


def record(rid, tid):
    return {"rubric_id": rid, "source_trajectory_id": tid, "analysis_status": "valid",
            "analysis": {"intervention_turn_id": 1, "ideal_response_behavior": "State what is missing."}}


def test_the_frozen_text_is_the_failed_turn_of_every_rerun_and_a_generation_failure_is_not_a_fail(tmp_path):
    records = [record("r-ok", "t1"), record("r-lost", "t2"), record("r-none", "t3")]
    trajectories = {t: {"trajectory_id": t} for t in ("t1", "t2", "t3")}
    generated, seen = [], []

    def intervene(candidate, trajectory, instruction):
        generated.append(candidate["rubric_id"])
        assert "State what is missing." in instruction
        if candidate["rubric_id"] == "r-none":
            return {"status": GENERATION_FAILURE, "text": None, "assessment": None, "attempts": [{"attempt": 1, "label": 1}]}
        return {"status": "frozen", "text": f"frozen answer of {candidate['rubric_id']}", "assessment": "X", "attempts": [{"attempt": 1, "label": 2}]}

    def rerun(trajectory, turn, instruction, index, fixed):
        seen.append((trajectory["trajectory_id"], index, fixed["text"], fixed["assessment"]))
        correct = trajectory["trajectory_id"] == "t1" and index != 0  # what follows the frozen turn differs between reruns
        return {"status": "completed", "final_diagnosis": f"d{index}", "correct": correct, "intervention_response": fixed["text"]}

    out = tmp_path / "validations.jsonl"
    counts = validate_candidates(records, trajectories, rerun, num_reruns=3, min_successes=2, output_path=out, intervene=intervene)
    assert counts == {"pass": 1, "fail": 1, "error": 0, "skipped_done": 0, GENERATION_FAILURE: 1}
    assert generated == ["r-ok", "r-lost", "r-none"]  # generated once per candidate, not once per rerun
    assert [s[:2] for s in seen] == [("t1", 0), ("t1", 1), ("t1", 2), ("t2", 0), ("t2", 1), ("t2", 2)]  # no rerun without an intervention
    assert {s[2] for s in seen if s[0] == "t1"} == {"frozen answer of r-ok"} and {s[3] for s in seen} == {"X"}
    rows = load_validations(out)
    assert rows["r-ok"]["validation_status"] == "pass" and rows["r-ok"]["successes"] == 2  # the 2-of-3 rule is unchanged
    assert rows["r-lost"]["validation_status"] == "fail" and rows["r-lost"]["intervention"]["text"] == "frozen answer of r-lost"
    assert rows["r-none"]["validation_status"] == GENERATION_FAILURE and rows["r-none"]["reruns"] == []
    assert [r["final_diagnosis"] for r in rows["r-ok"]["reruns"]] == ["d0", "d1", "d2"]
    assert [r["rubric_id"] for r in passed_records(records, rows, required=True)] == ["r-ok"]
    # a second run decides nothing again: neither the pass, the fail nor the generation failure
    again = validate_candidates(records, trajectories, rerun, num_reruns=3, min_successes=2, output_path=out, intervene=intervene)
    assert again["skipped_done"] == 3 and len(generated) == 3 and len(out.read_text().splitlines()) == 3


def test_an_intervention_that_could_not_be_checked_is_an_error_and_is_retried(tmp_path):
    calls = []

    def intervene(candidate, trajectory, instruction):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("scorer unavailable")
        return {"status": "frozen", "text": "a", "assessment": "", "attempts": []}

    rerun = lambda trajectory, turn, instruction, index, fixed: {"status": "completed", "correct": True, "final_diagnosis": "d"}  # noqa: E731
    args = ([record("r", "t")], {"t": {}}, rerun)
    out = tmp_path / "v.jsonl"
    first = validate_candidates(*args, num_reruns=3, min_successes=2, output_path=out, intervene=intervene)
    assert first["error"] == 1 and json.loads(out.read_text().splitlines()[0])["reruns"] == []
    assert validate_candidates(*args, num_reruns=3, min_successes=2, output_path=out, intervene=intervene)["pass"] == 1


def test_without_an_intervention_the_validation_is_what_it_was(tmp_path):
    rerun = lambda trajectory, turn, instruction, index: {"status": "completed", "correct": index < 2, "final_diagnosis": "d"}  # noqa: E731
    out = tmp_path / "v.jsonl"
    counts = validate_candidates([record("r", "t")], {"t": {}}, rerun, num_reruns=3, min_successes=2, output_path=out)
    assert counts == {"pass": 1, "fail": 0, "error": 0, "skipped_done": 0} and "intervention" not in json.loads(out.read_text())


def test_the_behavioral_gate_counts_the_rubric_behavior_not_the_final_answer(tmp_path):
    from src.judges.rubric_validation import load_validations

    records = [record("r-shown", "t1"), record("r-already", "t2"), record("r-rare", "t3")]
    trajectories = {t: {"trajectory_id": t} for t in ("t1", "t2", "t3")}
    intervene = lambda c, t, i: {"status": "frozen", "text": "x", "assessment": "", "attempts": []}  # noqa: E731
    rerun = lambda trajectory, turn, instruction, index, fixed: {"status": "completed", "correct": False, "final_diagnosis": "wrong"}  # noqa: E731
    labels = {"r-shown": (1, [2, 2, 0]), "r-already": (2, [2, 2, 2]), "r-rare": (0, [2, 1, 1])}

    def behavior(candidate, trajectory, instruction):
        before, drawn = labels[candidate["rubric_id"]]
        return {"before": before, "samples": [{"text": "s", "label": x} for x in drawn]}

    out = tmp_path / "v.jsonl"
    counts = validate_candidates(records, trajectories, rerun, num_reruns=3, min_successes=2, output_path=out,
                                 intervene=intervene, behavior=behavior)
    rows = load_validations(out)
    assert rows["r-shown"]["validation_status"] == "pass" and rows["r-shown"]["task_recovery"]["successes"] == 0  # no correct answer needed
    assert rows["r-already"]["validation_status"] == "fail" and rows["r-already"]["behavior"]["absent_before"] is False
    assert rows["r-rare"]["validation_status"] == "fail" and rows["r-rare"]["behavior"]["shown"] == 1
    assert counts["pass"] == 1 and counts["fail"] == 2
