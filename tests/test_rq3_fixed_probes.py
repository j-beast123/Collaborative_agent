"""RQ3 fixed probes: every checkpoint is read on exactly the frozen probes, and the rows are what probe_transitions counts."""
import pytest

from scripts.rq3_fixed_probes import checkpoint_rows, digest
from src.evaluation.rq3 import probe_transitions

D1 = {"speaker": "doctor", "stage": "D1", "text": "I think X.", "provisional_diagnosis": "X"}


def trajectory(case, correct, seed=0, identity="p", status="valid", opening=D1):
    return {"case_id": case, "rollout_index": seed, "turns": [dict(opening, round=1), {"speaker": "agent", "stage": "A1", "text": "a"}],
            "is_correct": correct, "trajectory_status": status, "final_diagnosis": "X", "num_turns": 1, "stop": "early",
            "collaboration_state": "Accept", "checkpoint_id": "c", "protocol_identity": identity}


def manifest(cases, seeds=(0, 1)):
    return {"probes": [{"probe_id": c, "starting_history_sha256": digest([D1])} for c in cases], "evaluation_seeds": list(seeds),
            "dataset_manifest_sha256": "d" * 64}


def checkpoint(outcomes, **kwargs):
    return [trajectory(case, correct, seed, **kwargs) for (case, seed), correct in outcomes.items()]


def test_checkpoints_on_the_same_probe_units_give_the_transition_counts_and_case_level_shares():
    from scripts.rq3_fixed_probes import transition_shares

    frozen = manifest(["a", "b"])
    base = checkpoint_rows(frozen, checkpoint({("a", 0): False, ("a", 1): False, ("b", 0): True, ("b", 1): True}), 0, "base")
    iter1 = checkpoint_rows(frozen, checkpoint({("a", 0): True, ("a", 1): False, ("b", 0): False, ("b", 1): True}), 1, "iter1")
    assert [r["probe_id"] for r in base] == ["a#s0", "a#s1", "b#s0", "b#s1"]
    rows = [dict(r, category_id="ALL") for r in base + iter1]
    assert probe_transitions(rows) == [{"category_id": "ALL", "from_iteration": 0, "to_iteration": 1, "fail_to_success": 1,
                                        "fail_to_fail": 1, "success_to_fail": 1, "success_to_success": 1}]
    shares = {s["transition"]: s for s in transition_shares(rows)}
    assert all(s["share"] == 0.25 and s["units"] == 4 and s["cases"] == 2 for s in shares.values())  # units 4, resampled as 2 cases
    assert all(r["protocol_hash"] == "p" and r["dataset_hash"] == "d" * 16 for r in rows)


def test_a_checkpoint_evaluated_on_other_units_another_start_or_mixed_protocols_is_refused():
    frozen = manifest(["a", "b"])
    full = {("a", 0): True, ("a", 1): True, ("b", 0): True, ("b", 1): True}
    with pytest.raises(SystemExit, match="not exactly the probe units"):
        checkpoint_rows(frozen, checkpoint({k: v for k, v in full.items() if k != ("b", 1)}), 1, "x")  # a seed is missing
    with pytest.raises(SystemExit, match="not exactly the probe units"):
        checkpoint_rows(frozen, checkpoint({**full, ("z", 0): True}), 1, "x")
    with pytest.raises(SystemExit, match="more than one collaboration"):
        checkpoint_rows(frozen, checkpoint(full) + [trajectory("a", True, 0)], 1, "x")
    with pytest.raises(SystemExit, match="frozen starting history"):
        checkpoint_rows(frozen, checkpoint({k: v for k, v in full.items() if k != ("b", 1)}) + [trajectory("b", True, 1, opening=dict(D1, text="another D1"))], 1, "x")
    with pytest.raises(SystemExit, match="more than one protocol identity"):
        checkpoint_rows(frozen, checkpoint({k: v for k, v in full.items() if k != ("b", 1)}) + [trajectory("b", True, 1, identity="q")], 1, "x")
    invalid = checkpoint_rows(frozen, checkpoint({k: v for k, v in full.items() if k != ("b", 1)}) + [trajectory("b", True, 1, status="invalid")], 1, "x")
    assert [r["success"] for r in invalid] == [True, True, True, None]  # an invalid collaboration is neither a success nor a failure


def test_post_hoc_taxonomy_is_a_partition_of_the_iterations_rubrics_and_feeds_the_bubble_rows():
    import json

    from scripts.rq3_taxonomy import build_taxonomy, registry_of
    from src.evaluation.rq3 import bubble_rows
    from src.judges.llm import CachedChat, ScriptedBackend

    items = [{"key": "iter0:c1", "iteration": 0, "name": "Ground disagreement", "applicability": "a", "scores": {"2": "x"}, "failure_count": 4},
             {"key": "iter0:c2", "iteration": 0, "name": "Ask first", "applicability": "a", "scores": {"2": "x"}, "failure_count": 2},
             {"key": "iter1:c1", "iteration": 1, "name": "Cite findings when disagreeing", "applicability": "a", "scores": {"2": "x"}, "failure_count": 1}]
    answer = {"themes": [{"theme_id": "T01", "name": "Evidence-grounded disagreement", "definition": "d"},
                         {"theme_id": "T02", "name": "Ask before revising", "definition": "d"}],
              "assignments": {"iter0:c1": "T01", "iter1:c1": "T01", "iter0:c2": "T02"}}
    taxonomy = build_taxonomy(items, CachedChat(ScriptedBackend(lambda request: json.dumps(answer)), None), {}, "S")
    rows = bubble_rows(registry_of(items, taxonomy), top_k=8)
    assert {(r["category_name"], r["iteration"]): r["failure_count"] for r in rows} == {
        ("Evidence-grounded disagreement", 0): 4, ("Ask before revising", 0): 2, ("Evidence-grounded disagreement", 1): 1}
    assert abs(sum(r["proportion"] for r in rows if r["iteration"] == 0) - 1) < 1e-9
    for broken in ({**answer, "assignments": {"iter0:c1": "T01", "iter0:c2": "T02"}},            # a rubric is left out
                   {**answer, "assignments": {**answer["assignments"], "iter1:c1": "T09"}}):   # a theme that does not exist
        with pytest.raises(ValueError, match="not a partition"):
            build_taxonomy(items, CachedChat(ScriptedBackend(lambda request, b=broken: json.dumps(b)), None), {}, "S")
