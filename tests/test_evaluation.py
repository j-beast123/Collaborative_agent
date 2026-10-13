"""Offline end-to-end tests of the evaluation pipeline (RQ1, RQ2, RQ3, figures)."""

import csv
import json

import pytest

from scripts import run_eval
from src.evaluation.behavior_judge import BehaviorJudge
from src.evaluation.controlled_history import EvidenceBuilder, claimed_option
from src.evaluation.rq3 import bubble_rows, probe_transitions, rubric_schedule
from src.evaluation.stats import cluster_bootstrap_mean, paired_difference
from src.judges.llm import CachedChat, ScriptedBackend
from test_serl_steps import fake_runtime, rich_trajectory

CONFIG = json.loads((run_eval.ROOT / "configs/eval/eval.json").read_text())
CONFIG["meta"].update(run_id="run-1", source_dataset="ds", dataset_version="v1", split_manifest_hash="h")
for name in ("base", "serl"):
    CONFIG["systems"][name].update(model_family="qwen", model_id=f"{name}-model", checkpoint=name,
                                   iteration=0 if name == "base" else 2)


def eval_trajectory(tid, agent_initial="A", doctor_initial="A", correct=True):
    t = rich_trajectory(tid, correct=correct)
    t.update(agent_initial_option=agent_initial, doctor_initial_option=doctor_initial, original_case_id=f"case-{tid}")
    return t


def judge_saying(label):
    backend = ScriptedBackend(lambda r: json.dumps({"behavior_class": label, "confidence": 0.9, "rationale": "r"}))
    return BehaviorJudge(CachedChat(backend, None), system_prompt="J", prompt_version="behavior-judge-v1", params={})


def test_stats_cluster_bootstrap_and_paired():
    result = cluster_bootstrap_mean([1, 1, 0, 0], ["a", "a", "b", "b"], n_boot=200)
    assert result["mean"] == 0.5 and result["clusters"] == 2 and 0 <= result["ci_low"] <= result["ci_high"] <= 1
    diff = paired_difference([("a", 0.25, 0.75), ("b", 0.5, 0.5)], n_boot=200)
    assert diff["mean_difference"] == pytest.approx(0.25)


def test_rq1_accuracy_table(tmp_path):
    trajectories = [eval_trajectory("t1", "A", "A", True), eval_trajectory("t2", "B", "A", False),
                    dict(eval_trajectory("t3"), trajectory_status="invalid", is_correct=None)]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(t) for t in trajectories))
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps({"case_id": c, "is_correct": c != "case-t2"})
                                                for c in ("case-t1", "case-t2", "case-t3")))
    table = run_eval.run_rq1([{"dataset": "ds", "system": "serl", "checkpoint": "it2", "iteration": 2,
                               "trajectories": str(tmp_path / "t.jsonl"), "standalone": str(tmp_path / "s.jsonl")}],
                             tmp_path)
    row = table[0]
    assert row["collaborative_accuracy"] == pytest.approx(1 / 3) and row["collaborative_accuracy_valid_only"] == 0.5
    assert row["n_invalid"] == 1 and row["standalone_accuracy"] == pytest.approx(2 / 3)
    assert row["final_accuracy_Preserve"] == 1.0 and row["final_accuracy_Accept"] == 0.0


def test_rq2_states_rows_and_export(tmp_path):
    trajectories = {"base": [eval_trajectory("t1", "B", "A"), eval_trajectory("t2", "A", "A")],
                    "serl": [eval_trajectory("t1", "B", "A"), eval_trajectory("t2", "A", "A")]}
    rows = run_eval.run_rq2_states(CONFIG, trajectories, judge_saying("ACCEPT_REVISE"), tmp_path)
    by = {(r["system"], r["case_id"]): r for r in rows}
    assert by[("base", "case-t1")]["collaboration_state"] == "Accept" and by[("base", "case-t1")]["appropriateness_match"] == 1
    assert by[("base", "case-t2")]["collaboration_state"] == "Preserve" and by[("base", "case-t2")]["appropriateness_match"] == 0
    assert (tmp_path / "figure_data/state_results.csv").exists()
    with pytest.raises(Exception):  # results are never silently overwritten
        run_eval.run_rq2_states(CONFIG, trajectories, judge_saying("ACCEPT_REVISE"), tmp_path)


def evidence_builder(reject_strength=None):
    gen = ScriptedBackend(lambda r: json.dumps({"evidence": f"Evidence ({json.loads(r.user)['strength']})."}))

    def validate(r):
        strength = json.loads(r.user)["strength"]
        return json.dumps({"factual_consistency": True, "claim_support": "supports", "evidence_sufficiency": "x",
                           "clinical_plausibility": True, "ambiguity": False,
                           "accept": strength != reject_strength, "reason": "ok"})
    return EvidenceBuilder(CachedChat(gen, None), CachedChat(ScriptedBackend(validate), None), generator_prompt="G",
                           validator_prompt="V", prompt_version="evidence-v1", params={})


def test_rq2_controlled_same_inputs_for_both_systems(tmp_path):
    reference = [eval_trajectory("t1", "B", "A"), eval_trajectory("t2", "A", "A")]  # t1 wrong, t2 correct at round 1
    runtimes = {}

    def factory(config, name):
        runtimes[name] = fake_runtime(agent_text="I would like more evidence before changing.")
        return runtimes[name][0]

    rows = run_eval.run_rq2_controlled(CONFIG, reference, evidence_builder(reject_strength="weak"),
                                       judge_saying("CONSIDER_REQUEST_EVIDENCE"), tmp_path, runtime_factory=factory)
    conditions = {(r["case_id"], r["condition_id"]) for r in rows}
    assert conditions == {("case-t1", "correct_strong"), ("case-t1", "correct_weak"),
                          ("case-t2", "incorrect_strong_looking"), ("case-t2", "incorrect_none")}
    by = {(r["system"], r["case_id"], r["condition_id"]): r for r in rows}
    for key in conditions:  # identical controlled input for Base and SERL
        assert by[("base", *key)]["controlled_input_hash"] == by[("serl", *key)]["controlled_input_hash"]
    assert by[("base", "case-t1", "correct_weak")]["evaluation_status"] == "excluded"  # rejected evidence
    assert by[("serl", "case-t1", "correct_weak")]["appropriateness_match"] is None
    assert by[("base", "case-t2", "incorrect_none")]["appropriateness_match"] == 0
    # The response prompt never contains the Situation Assessment; assessment ran separately.
    prompts = runtimes["base"][1]["agent_prompts"]
    assert prompts and not any("Private current Q1/Q2" in p for p in prompts)
    assert runtimes["base"][1]["judge"] > 0
    items = json.loads((tmp_path / "rq2_controlled_items.json").read_text())
    assert all("I think the answer is option" in i["dialogue"][-1]["text"] for i in items)
    assert claimed_option(reference[1], False) != "A"


def test_rq2_effectiveness_matched_continuation(tmp_path):
    reference = [eval_trajectory(f"t{i}") for i in range(3)]

    def factory(config, name):
        if name == "serl":
            return fake_runtime(agent_text="SERL verifies the evidence first.")[0]
        runtime, _ = fake_runtime(agent_text="BASE simply agrees.")
        # Continuation: the final decision is correct only if the SERL response is in the dialogue.
        runtime.doctor = lambda system, user: ("Final answer: option A - a" if "SERL" in user else "Final answer: option B - b") \
            if "This is the final decision" in user else "Let us keep discussing."
        return runtime

    config = json.loads(json.dumps(CONFIG))
    config["matched_continuation"]["k"] = 2
    summary = run_eval.run_rq2_effectiveness(config, reference, tmp_path, runtime_factory=factory)
    assert summary["mean_success_a"] == 0.0 and summary["mean_success_b"] == 1.0
    assert summary["mean_difference"] == 1.0 and summary["n_pairs"] == 3 and summary["incomplete_pairs"] == 0


def test_rq3_schedule_bubble_transitions(tmp_path):
    sets = ["R0", "R1", "R2"]
    assert rubric_schedule("fixed_initial", 3, sets) == ["R0", "R0", "R0"]
    assert rubric_schedule("evolving", 3, sets) == ["R0", "R1", "R2"]
    assert rubric_schedule("fixed_final_oracle", 3, sets) == ["R2", "R2", "R2"]
    registry = [{"category_id": "cat-001", "name": "Yield", "status": "active", "frequency_history": {"0": 3, "1": 1}},
                {"category_id": "cat-002", "name": "Ignore", "status": "inactive", "frequency_history": {"0": 1, "1": 1}},
                {"category_id": "cat-003", "name": "Dup", "status": "merged", "frequency_history": {}}]
    bubble = bubble_rows(registry)
    assert {(r["category_id"], r["iteration"]): r["proportion"] for r in bubble}[("cat-001", 0)] == 0.75
    transitions = probe_transitions([
        {"probe_id": "p1", "category_id": "cat-001", "iteration": 0, "success": False},
        {"probe_id": "p1", "category_id": "cat-001", "iteration": 1, "success": True},
        {"probe_id": "p2", "category_id": "cat-001", "iteration": 0, "success": True},
        {"probe_id": "p2", "category_id": "cat-001", "iteration": 1, "success": False},
    ])
    assert transitions == [{"category_id": "cat-001", "from_iteration": 0, "to_iteration": 1, "fail_to_success": 1,
                            "fail_to_fail": 0, "success_to_fail": 1, "success_to_success": 0}]
    run_eval.run_rq3(registry, tmp_path)
    assert run_eval.run_plots(tmp_path) == ["fig_rq3_bubble.png"]


def test_plots_from_pipeline_outputs(tmp_path):
    trajectories = {"base": [eval_trajectory("t1", "B", "A")], "serl": [eval_trajectory("t1", "B", "A")]}
    run_eval.run_rq2_states(CONFIG, trajectories, judge_saying("ACCEPT_REVISE"), tmp_path)
    run_eval.run_rq2_controlled(CONFIG, [eval_trajectory("t1", "B", "A")], evidence_builder(), judge_saying("ACCEPT_REVISE"),
                                tmp_path, runtime_factory=lambda c, n: fake_runtime()[0])
    made = run_eval.run_plots(tmp_path)
    assert "fig_rq2_behavior.png" in made and (tmp_path / "fig_rq2_behavior.png").stat().st_size > 0
    with open(tmp_path / "figure_data/controlled_results.csv") as handle:
        assert {r["system"] for r in csv.DictReader(handle)} == {"base", "serl"}


def test_missing_metadata_blocks_evaluation():
    config = json.loads((run_eval.ROOT / "configs/eval/eval.json").read_text())
    with pytest.raises(Exception, match="metadata unresolved"):
        run_eval.run_rq2_states(config, {"base": []}, judge_saying("OTHER"), run_eval.ROOT / "unused")


def test_recover_has_no_fixed_expected_action(tmp_path):
    trajectories = {"base": [eval_trajectory("t9", "B", "C", correct=False)]}  # both initially wrong -> Recover
    assert CONFIG["expected_actions"]["states"]["Recover"] is None
    rows = run_eval.run_rq2_states(CONFIG, trajectories, judge_saying("ACCEPT_REVISE"), tmp_path)
    assert rows[0]["collaboration_state"] == "Recover"
    assert rows[0]["evaluation_status"] == "excluded" and rows[0]["appropriateness_match"] is None


def test_bubble_top_keywords_pool_the_rest_into_other():
    registry = [{"category_id": f"cat-{i:03d}", "name": f"c{i}", "status": "active",
                 "frequency_history": {"0": 10 - i}} for i in range(10)]
    rows = bubble_rows(registry, top_k=8)
    assert len({r["category_id"] for r in rows}) == 9
    other = [r for r in rows if r["category_id"] == "OTHER"][0]
    assert other["failure_count"] == 2 + 1 and other["category_name"] == "OTHER"
    assert sum(r["proportion"] for r in rows) == pytest.approx(1.0)


def test_rq2_effectiveness_retries_format_failures_and_resumes(tmp_path):
    reference = [eval_trajectory(f"t{i}") for i in range(2)]
    calls = {"reconstruct": 0, "final": 0}

    def factory(config, name):
        runtime, _ = fake_runtime(agent_text=f"{name.upper()} reply.")
        reconstruct = runtime.reconstruct

        def flaky_reconstruct(case, dialogue):  # every other reconstruction comes back unparsable
            calls["reconstruct"] += 1
            return {"parse_status": "invalid"} if calls["reconstruct"] % 2 else reconstruct(case, dialogue)

        def doctor(system, user):
            if "This is the final decision" in user:
                calls["final"] += 1
                return "Final answer: option A - a"
            return "Let us keep discussing."

        runtime.reconstruct, runtime.doctor = flaky_reconstruct, doctor
        return runtime

    config = json.loads(json.dumps(CONFIG))
    config["matched_continuation"].update(k=2, max_rollout_attempts=4)
    summary = run_eval.run_rq2_effectiveness(config, reference, tmp_path, runtime_factory=factory)
    assert summary["n_pairs"] == 2 and summary["incomplete_pairs"] == 0  # no outcome lost to a format failure
    finals = calls["final"]
    assert finals == 2 * 2 * 2  # histories x systems x K
    again = run_eval.run_rq2_effectiveness(config, reference, tmp_path, runtime_factory=factory)
    assert calls["final"] == finals and again["mean_difference"] == summary["mean_difference"]  # resumed, nothing rerun


def extractor_reading(read):
    from src.judges.option_extraction import OptionExtractor

    backend = ScriptedBackend(lambda request: read(json.loads(request.user)["message"]))
    return OptionExtractor(CachedChat(backend, None), system_prompt="E", prompt_version="option-extractor-v1", params={})


def test_option_extractor_replaces_regex_initial_options(tmp_path):
    from src.judges.option_extraction import with_initial_options

    trajectory = eval_trajectory("t1", agent_initial="", doctor_initial="D")  # regex: agent unparsed, doctor misread
    extractor = extractor_reading(lambda message: "B" if message == trajectory["turns"][1]["text"] else "a.")
    read = with_initial_options(trajectory, extractor)
    assert (read["agent_initial_option"], read["doctor_initial_option"]) == ("B", "A")
    assert read["initial_option_source"] == "option-extractor-v1" and trajectory["doctor_initial_option"] == "D"
    for reply, expected in (("NONE", ""), ("I think B or C", None), ("Z", None), (None, None)):
        assert extractor_reading(lambda message: reply).stated_option(trajectory, "m") == expected

    # RQ2 states: the unparsed trajectory is no longer dropped and gets the state of the re-read options.
    with pytest.raises(ValueError, match="No result rows"):  # the regex fields leave nothing to evaluate
        run_eval.run_rq2_states(CONFIG, {"base": [trajectory]}, judge_saying("ACCEPT_REVISE"), tmp_path / "regex")
    rows = run_eval.run_rq2_states(CONFIG, {"base": [trajectory]}, judge_saying("ACCEPT_REVISE"), tmp_path / "llm",
                                   extractor)
    assert [r["collaboration_state"] for r in rows] == ["Accept"]  # agent B (wrong), clinician A (correct)
