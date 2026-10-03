"""The experiment matrix: conditions, the unfiltered cumulative rubric set, shared initial rubrics, tables, rubric behavior."""

import json
import os
from pathlib import Path

import pytest

from scripts import build_rubrics, main_experiments, rubric_behavior
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import problems_of
from src.grpo.trainer import NO_RUBRICS, load_rubric_set
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.raw_rubrics import raw_rubric_set
from src.judges.reward import turn_appropriateness
from src.judges.rubric_scoring import ChatRubricScorer, ScoringItem
from test_judges import CONFIG, analysis, manager_responses, schema_rubric, trajectory

ROOT = Path(__file__).resolve().parents[1]
MATRIX = main_experiments.load_matrix()


def registry(schedule, **extra):
    return {"max_active_categories": 8, "schedule": schedule, "fixed_rubric_set": None, "initial_rubrics": None, **extra}


def factory_with(calls):
    def factory(role, name):
        def reply(request):
            calls.append(name)
            tid = json.loads(request.user.split("\n\n", 1)[1])["trajectory_id"] if name == "failure_analyzer" else None
            return analysis(tid, f"Rubric of {tid}") if name == "failure_analyzer" else manager_responses(request)
        return CachedChat(ScriptedBackend(reply), None)
    return factory


def never(*_):
    raise AssertionError("no validation rerun in this condition")


def test_no_filtering_set_is_every_raw_candidate_cumulative_and_causal(tmp_path):
    config = dict(CONFIG, rubric_registry=registry("no_filtering"))
    failures = {0: ["a1", "a2"], 1: ["b1"], 2: ["c1", "c2", "c3"]}
    calls, previous, seen = [], None, []
    for iteration, names in failures.items():
        (tmp_path / f"traj{iteration}.jsonl").write_text("\n".join(
            [json.dumps(trajectory(t)) for t in names] + [json.dumps(trajectory(f"ok{iteration}", correct=True))]))
        out = tmp_path / f"iter{iteration}/rubrics"
        summary = build_rubrics.run(config, tmp_path / f"traj{iteration}.jsonl", out, iteration, chat_factory=factory_with(calls),
                                    rerun_fn=never, previous_rubric_set=previous)
        rubric_set = json.loads((out / "rubric_set.json").read_text())
        seen += names
        # the set of iteration i holds the candidates of iterations 0 .. i and nothing later
        assert sorted(t for r in rubric_set["rubrics"] for t in r["source_trajectories"]) == sorted(seen)
        assert summary["raw_candidates_this_iteration"] == len(names) and summary["active_rubrics"] == len(seen)
        assert rubric_set["discovered_by_iteration"] == {str(i): len(failures[i]) for i in range(iteration + 1)}
        assert all(r["weight"] == 1.0 and r["category_id"] == r["rubric_id"] and r["origin"] == "raw_candidate" for r in rubric_set["rubrics"])
        assert rubric_set["rubric_schedule"] == "no_filtering" and rubric_set["cumulative"] and rubric_set["technical_dedup"] == []
        assert not (out / "validations.jsonl").exists() and not (out / "categories.json").exists()
        previous = out / "rubric_set.json"
    assert set(calls) == {"failure_analyzer"} and len(calls) == 6  # one analysis per failure; the rubric manager is never called
    with pytest.raises(IntegrationUnavailable, match="previous-rubric-set"):
        build_rubrics.run(config, tmp_path / "traj1.jsonl", tmp_path / "alone", 1, chat_factory=factory_with([]), rerun_fn=never)
    # the earlier candidates are kept as they were, in order, and a later set cannot serve an earlier iteration
    first, last = (json.loads((tmp_path / f"iter{i}/rubrics/rubric_set.json").read_text()) for i in (0, 2))
    assert [r["rubric_id"] for r in last["rubrics"][:2]] == [r["rubric_id"] for r in first["rubrics"]]
    with pytest.raises(ValueError, match="iteration before"):
        raw_rubric_set([], 1, last)
    # the same analysis record read twice is one rubric; identical wording from two failures stays two rubrics
    records = [json.loads(line) for line in (tmp_path / "iter0/rubrics/failure_analyses.jsonl").read_text().splitlines()]
    again = raw_rubric_set(records + records[:1], 0)
    assert len(again["rubrics"]) == 2 and [d["rubric_id"] for d in again["technical_dedup"]] == [records[0]["rubric_id"]]

    # the reward reads the whole set: no cap for this schedule, and a set of another schedule is refused
    run = {"output_dir": ".", "rubrics": {"rubric_set": "iter{iteration}/rubrics/rubric_set.json"},
           "rubric_registry": registry("no_filtering", max_active_categories=2), "reward": {"weights": {"appropriateness": 0.3}}}
    assert len(load_rubric_set(run, tmp_path, 2)["rubrics"]) == 6
    with pytest.raises(IntegrationUnavailable, match="does not belong"):
        load_rubric_set(dict(run, rubric_registry=registry("evolving")), tmp_path, 2)


def test_conditions_share_the_iteration_zero_rubric_step(tmp_path, monkeypatch):
    monkeypatch.setattr(build_rubrics, "ROOT", tmp_path)
    (tmp_path / "prompts").symlink_to(ROOT / "prompts")
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(trajectory(t)) for t in ("t1", "t2", "t3")))
    calls = []
    build_rubrics.run(dict(CONFIG, rubric_registry=registry("evolving")), tmp_path / "traj.jsonl", tmp_path / "ours/iter0/rubrics", 0,
                      chat_factory=factory_with(calls))
    made = len(calls)
    shared = registry("evolving", initial_rubrics="ours/iter0/rubrics")
    summary = build_rubrics.run(dict(CONFIG, rubric_registry=shared), tmp_path / "traj.jsonl", tmp_path / "other/iter0/rubrics", 0,
                                chat_factory=factory_with(calls), rerun_fn=never)
    assert len(calls) == made and summary["initial_rubrics_source"].endswith("ours/iter0/rubrics")  # no model call: R_0 is taken over
    for name in ("rubric_set.json", "categories.json", "failure_analyses.jsonl"):
        assert (tmp_path / "other/iter0/rubrics" / name).read_text() == (tmp_path / "ours/iter0/rubrics" / name).read_text()
    # the unfiltered condition takes the same analysis: its raw candidates are the ones Ours validated and consolidated
    raw = build_rubrics.run(dict(CONFIG, rubric_registry=registry("no_filtering", initial_rubrics="ours/iter0/rubrics")),
                            tmp_path / "traj.jsonl", tmp_path / "raw/iter0/rubrics", 0, chat_factory=factory_with(calls), rerun_fn=never)
    assert len(calls) == made and raw["active_rubrics"] == 3
    ours = [json.loads(line)["rubric_id"] for line in (tmp_path / "ours/iter0/rubrics/failure_analyses.jsonl").read_text().splitlines()]
    assert [r["rubric_id"] for r in json.loads((tmp_path / "raw/iter0/rubrics/rubric_set.json").read_text())["rubrics"]] == ours
    # from iteration 1 on a run analyses its own failures
    build_rubrics.run(dict(CONFIG, rubric_registry=registry("no_filtering", initial_rubrics="ours/iter0/rubrics")),
                      tmp_path / "traj.jsonl", tmp_path / "raw/iter1/rubrics", 1, chat_factory=factory_with(calls), rerun_fn=never,
                      previous_rubric_set=tmp_path / "raw/iter0/rubrics/rubric_set.json")
    assert len(calls) == made + 3


def test_scorer_splits_a_long_rubric_list_into_calls_without_changing_labels():
    rubrics = tuple(dict(schema_rubric(f"r{i}"), rubric_id=f"r{i}") for i in range(20))
    item = ScoringItem("i", "c", "t", 1, ({"turn_id": 0, "speaker": "doctor", "text": "d"}, {"turn_id": 1, "speaker": "agent", "text": "a"}), rubrics)
    sizes = []

    def reply(request):
        asked = json.loads(request.user.split("\n\n", 1)[1])["rubrics"]
        sizes.append(len(asked))
        return json.dumps({"turn_id": 1, "scores": [{"rubric_id": r["rubric_id"], "applicable": int(r["rubric_id"][1:]) % 3 != 0,
                                                     "score": int(r["rubric_id"][1:]) % 3, "rationale": "x"} for r in asked]})

    def scorer(limit):
        return ChatRubricScorer("s", CachedChat(ScriptedBackend(reply), None), system_prompt="p", prompt_version="v", params={},
                                max_rubrics_per_call=limit)

    whole, split = scorer(None).score(item), scorer(8)
    parts = split.score(item)
    assert sizes == [20, 8, 8, 4] and split.calls == 3 and parts.call["calls"] == 3
    assert parts.status == whole.status == "scored" and parts.labels == whole.labels and len(parts.labels) == 20
    weights = {r["rubric_id"]: 1.0 for r in rubrics}  # the reward's aggregation sees the same labels either way
    assert turn_appropriateness(parts.labels, weights, "weighted_softmin", 0.5) == turn_appropriateness(whole.labels, weights, "weighted_softmin", 0.5)
    assert scorer(8).identity()["max_rubrics_per_call"] == 8 and "max_rubrics_per_call" not in scorer(None).identity()


def test_matrix_conditions_differ_only_in_reward_components_and_rubric_schedule():
    result = main_experiments.plan(MATRIX)
    assert result["controls"]["problems"] == [] and MATRIX["model"] == "Qwen3-4B"
    rows = {r["condition"]: r for r in result["conditions"]}
    assert list(rows) == ["baseline", "outcome", "effectiveness_outcome", "static_rubric", "no_filtering", "ours", "minus_outcome", "minus_effectiveness"]
    # a removed component has weight 0; the others keep the coefficient of the full method (nothing is renormalized)
    assert {name: row["reward_formula"] for name, row in rows.items()} == {
        "baseline": "no training", "outcome": "R_t = 0.4 R_O", "effectiveness_outcome": "R_t = 0.4 R_O + 0.3 R_E,t",
        "static_rubric": "R_t = 0.4 R_O + 0.3 R_A,t + 0.3 R_E,t", "no_filtering": "R_t = 0.4 R_O + 0.3 R_A,t + 0.3 R_E,t",
        "ours": "R_t = 0.4 R_O + 0.3 R_A,t + 0.3 R_E,t", "minus_outcome": "R_t = 0.3 R_A,t + 0.3 R_E,t",
        "minus_effectiveness": "R_t = 0.4 R_O + 0.3 R_A,t"}
    assert {name: row["rubric_strategy"] for name, row in rows.items() if row["reward_weights"] and row["reward_weights"]["appropriateness"]} == {
        "static_rubric": "static", "no_filtering": "no_filtering_cumulative", "ours": "filtered_evolving",
        "minus_outcome": "filtered_evolving", "minus_effectiveness": "filtered_evolving"}
    assert [(r["filtering"], r["evolving"], r["cumulative"]) for r in (rows["static_rubric"], rows["no_filtering"], rows["ours"])] == [
        (True, False, False), (False, True, True), (True, True, False)]
    # "- Rubric" is the Effectiveness + Outcome run, not a second training
    reuse = [r for r in MATRIX["ablation_table"]["rows"] if r["label"] == "- Rubric"][0]
    assert reuse["condition"] == "effectiveness_outcome" and "reused_from" in reuse and rows["effectiveness_outcome"]["also_used_as"]
    assert sum(r["new_training_required"] for r in rows.values()) == 7 and not rows["baseline"]["new_training_required"]
    # an uncontrolled difference is caught
    base, other = read_config(ROOT / MATRIX["base_config"]), read_config(ROOT / "configs/experiments/main/no_filtering.json")
    assert main_experiments.differences(base, other, MATRIX["allowed_differences"]) == []
    other["grpo"] = dict(other["grpo"], num_train_epochs=2)  # more training for one condition
    assert main_experiments.differences(base, other, MATRIX["allowed_differences"]) == ["grpo.num_train_epochs"]
    for name, settings in MATRIX["conditions"].items():
        config = read_config(ROOT / settings["config"])
        assert config["policy"]["model_id"].endswith(MATRIX["model"]) and config["rubric_aggregation"] == base["rubric_aggregation"]
        assert config["reward"]["missing_component_policy"] == "omit" and config["environment"] == base["environment"]
        generated = (ROOT / config["environment"]["histories"]).is_file()  # the environment's D1 histories exist once it is generated
        found = [p for p in problems_of(config, ROOT) if generated or "environment.histories" not in p]
        assert found == [] or (name == "static_rubric" and all("fixed_rubric_set" in p for p in found)), (name, found)
        if config["reward"]["weights"]["appropriateness"] == 0:
            assert load_rubric_set(dict(config, output_dir="nowhere"), ROOT, 0) is NO_RUBRICS
    assert any("OPEN DECISION" in item for item in result["open_decisions"]) and len(result["open_decisions"]) >= 6


def fake_trajectories(path: Path, correct: dict[str, bool], turns: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"case_id": case, "split": "dev", "trajectory_status": "valid", "is_correct": ok,
                                        "num_turns": turns + (index % 2)}) + "\n" for index, (case, ok) in enumerate(correct.items())))


def test_tables_have_one_schema_pending_cells_and_a_paired_ablation(tmp_path, monkeypatch):
    for name in ("configs", "data"):
        (tmp_path / name).symlink_to(ROOT / name)
    monkeypatch.setattr(main_experiments, "ROOT", tmp_path)
    out = tmp_path / MATRIX["output_dir"]
    cases = [f"c{i}" for i in range(10)]
    for condition, right in (("ours", 8), ("effectiveness_outcome", 6), ("minus_outcome", 5)):
        run = tmp_path / "outputs/serl" / f"run_{condition}"
        for iteration in range(3):
            (run / f"iter{iteration}").mkdir(parents=True)
            (run / f"iter{iteration}/checkpoint.json").write_text(json.dumps(
                {"checkpoint_id": f"run_{condition}-iter{iteration}", "protocol_identity": "p" * 16, "rubric_version": "v",
                 "reward": {"weights": {}}}))
            fake_trajectories(run / f"iter{iteration}/eval_dev/trajectories.jsonl", {c: i < right for i, c in enumerate(cases)})
        main_experiments.register(MATRIX, out, condition, 0, f"run_{condition}")
    with pytest.raises(SystemExit, match="already run"):
        main_experiments.register(MATRIX, out, "ours", 0, "run_minus_outcome")
    with pytest.raises(SystemExit, match="not a trained condition"):
        main_experiments.register(MATRIX, out, "baseline", 0, "run_ours")
    fake_trajectories(out / "baseline/eval_dev/trajectories.jsonl", {c: i < 4 for i, c in enumerate(cases)})
    (out / "baseline/eval_dev/appropriateness.json").write_text(json.dumps({"value": 0.5, "definition": "d"}))

    assert main_experiments.tables(MATRIX, out)["main_table_rows"] == 6 * 3
    main = json.loads((out / "main_table.json").read_text())
    rows = main["rows"]
    assert all(set(main_experiments.FIELDS) - {"split", "cases", "collaborations", "task_success", "task_success_ci_low", "task_success_ci_high",
                                               "behavior_appropriateness", "effectiveness", "mean_turns", "early_stop_distribution"} <= set(r) for r in rows)
    baseline = [r for r in rows if r["condition"] == "baseline"]
    assert [r["iteration"] for r in baseline] == [1, 2, 3] and {r["task_success"] for r in baseline} == {0.4}  # one checkpoint, every column
    assert baseline[0]["behavior_appropriateness"] == 0.5 and baseline[0]["effectiveness"] is None and baseline[0]["checkpoint"] == "base"
    ours = [r for r in rows if r["condition"] == "ours"]
    assert [r["checkpoint"] for r in ours] == [f"run_ours-iter{i}" for i in range(3)] and ours[0]["task_success"] == 0.8
    assert ours[0]["early_stop_distribution"] == {"A1": 5, "A2": 5} and ours[0]["mean_turns"] == 1.5 and ours[0]["dataset_hash"]
    assert ours[0]["reward_components_enabled"] == ["outcome", "appropriateness", "effectiveness"] and ours[0]["rubric_strategy"] == "filtered_evolving"
    assert {r["status"] for r in rows if r["condition"] in ("outcome", "static_rubric", "no_filtering")} == {"pending"}
    assert all(r["split"] == "dev" for r in rows if r["status"] == "evaluated") and "not evaluated" in main["test"]
    assert (out / "main_table.csv").read_text().splitlines()[0].split(",")[:4] == ["method", "condition", "iteration", "seed"]

    ablation = {r["method"]: r for r in json.loads((out / "ablation_table.json").read_text())["rows"]}
    assert list(ablation) == ["Ours", "- Outcome", "- Effectiveness", "- Rubric"]
    assert ablation["Ours"]["final_task_success"] == 0.8 and ablation["Ours"]["delta_vs_ours"] == 0.0
    assert ablation["- Rubric"]["delta_vs_ours"] == pytest.approx(-0.2) and ablation["- Rubric"]["paired_cases"] == 10
    assert ablation["- Rubric"]["paired_ci_low"] <= -0.2 <= ablation["- Rubric"]["paired_ci_high"] and ablation["- Rubric"]["reused_from"]
    assert ablation["- Outcome"]["delta_vs_ours"] == pytest.approx(-0.3) and ablation["- Outcome"]["reward_formula"] == "R_t = 0.3 R_A,t + 0.3 R_E,t"
    assert ablation["- Effectiveness"]["status"] == "pending" and ablation["- Effectiveness"]["delta_vs_ours"] is None

    # a rewritten table keeps the version it replaces; evaluation files of another split are refused
    fake_trajectories(tmp_path / "outputs/serl/run_ours/iter2/eval_dev/trajectories.jsonl", {c: True for c in cases})
    main_experiments.tables(MATRIX, out)
    assert len(list((out / "history").glob("main_table_*.json"))) == 1 and len(list((out / "history").glob("ablation_table_*.csv"))) == 1
    (out / "baseline/eval_dev/trajectories.jsonl").write_text(json.dumps(
        {"case_id": "x", "split": "test", "trajectory_status": "valid", "is_correct": True, "num_turns": 1}) + "\n")
    with pytest.raises(SystemExit, match="outside dev"):
        main_experiments.tables(MATRIX, out)
    with pytest.raises(SystemExit, match="DEV only"):
        main_experiments.evaluate(dict(MATRIX, evaluation=dict(MATRIX["evaluation"], split="test")), out, "baseline", 0, None)


def test_rubric_behavior_counts_failures_and_effectiveness_mismatch_with_denominators(tmp_path):
    rubrics = tmp_path / "rubrics"
    rubrics.mkdir()
    (rubrics / "rubric_set.json").write_text(json.dumps({"rubrics": [
        dict(schema_rubric("Report the finding"), rubric_id="ra", category_id="cat-a", failure_count=6, source_rubric_candidates=["c1", "c2"]),
        dict(schema_rubric("Ask for evidence"), rubric_id="rb", category_id="cat-b", failure_count=2, source_rubric_candidates=["c3"])]}))
    (rubrics / "categories.json").write_text(json.dumps([
        {"category_id": "cat-a", "name": "Report the finding", "status": "active", "members": ["c1", "c2"], "frequency_history": {"0": 6}},
        {"category_id": "cat-b", "name": "Ask for evidence", "status": "active", "members": ["c3"], "frequency_history": {"0": 2}},
        {"category_id": "cat-c", "name": "Inactive one", "status": "inactive", "members": ["c4"], "frequency_history": {"0": 1}},
        {"category_id": "cat-d", "name": "Merged away", "status": "merged", "members": [], "frequency_history": {}}]))
    (rubrics / "failure_selection.json").write_text(json.dumps({"analyzed": 12}))

    def response(stage, ra, rb, delta, available=True):
        return {"stage": stage, "rubric_scores": {stage: {"labels": {"ra": ra, "rb": rb}}},
                "effectiveness": {stage: {"available": available, "delta": delta}}}

    log = [response("A1", 2, "NA", -0.2), response("A1", 2, "NA", 0.1), response("A2", 2, 0, 0.0), response("A3", 2, 1, -0.1, available=False),
           response("A1", 0, 2, 0.3), response("A2", 1, "NA", -0.4), response("A1", "NA", "NA", 0.2)]
    rows = {r["category_id"]: r for r in rubric_behavior.analyse(rubrics, log, 0, [0], 2)}
    assert list(rows) == ["cat-a", "cat-b", "cat-c"]  # by generation count; the merged category is not a row
    a = rows["cat-a"]
    assert (a["generation_count"], a["generation_frequency"], a["analysed_failures"], a["source_rubric_candidates"]) == (6, 0.5, 12, ["c1", "c2"])
    assert (a["scored_n"], a["applicable_n"], a["score_0_n"], a["score_1_n"], a["score_2_n"]) == (7, 6, 1, 1, 4)
    assert (a["behavior_failure_n"], a["behavior_failure_rate"]) == (1, pytest.approx(1 / 6))
    # fully satisfied (score 2) with an Effectiveness value: 3 of the 4 (one turn has none); 1 negative, 1 zero, 1 positive
    assert (a["compliant_effectiveness_n"], a["compliant_negative_effectiveness_n"], a["compliant_zero_effectiveness_n"],
            a["compliant_positive_effectiveness_n"], a["effectiveness_mismatch_rate"]) == (3, 1, 1, 1, pytest.approx(1 / 3))
    b = rows["cat-b"]
    assert (b["applicable_n"], b["score_0_n"], b["score_1_n"], b["score_2_n"], b["behavior_failure_rate"]) == (3, 1, 1, 1, pytest.approx(1 / 3))
    assert (b["compliant_effectiveness_n"], b["compliant_positive_effectiveness_n"], b["effectiveness_mismatch_rate"]) == (1, 1, 0.0)
    c = rows["cat-c"]  # discovered but not in the active set: it was never scored, so its rates are empty, not 0
    assert c["in_training_set"] is False and c["generation_count"] == 1 and c["applicable_n"] is None and c["behavior_failure_rate"] is None
    # the other reading of "failure" (score below 2) is the same counts with another setting
    assert rubric_behavior.analyse(rubrics, log, 0, [0, 1], 2)[0]["behavior_failure_n"] == 2
    # without an Effectiveness predictor the mismatch has no denominator
    blind = rubric_behavior.analyse(rubrics, [dict(r, effectiveness={}) for r in log], 0, [0], 2)[0]
    assert blind["compliant_effectiveness_n"] == 0 and blind["effectiveness_mismatch_rate"] is None
    rubric_behavior.plot(list(rows.values()), tmp_path / "figure.pdf", "title")
    assert (tmp_path / "figure.pdf").stat().st_size > 1000
    assert MATRIX["rubric_behavior"]["failure_scores_status"].startswith("OPEN DECISION")


def test_readiness_check_builds_one_sample_per_reached_turn_for_every_condition(tmp_path):
    """scripts/run_grpo.py check on the frozen base collaborations: the same TRAIN samples for every condition, and a
    condition without Appropriateness and Effectiveness is ready without a rubric set or a predictor."""
    from scripts.run_grpo import check
    from src.common.io import read_jsonl
    from src.common.main_config import load_main_config

    source = ROOT / read_config(ROOT / MATRIX["base_config"])["environment"]["base_trajectories"]
    if not (source.parent / "protocol.frozen.json").is_file():
        pytest.skip("needs the frozen base collaborations of the main environment")
    train = [t for t in read_jsonl(source) if t["split"] == "train"]
    (tmp_path / "base").mkdir()
    (tmp_path / "base/trajectories.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train))
    reached = {f"A{turn}": sum(t["num_turns"] >= turn for t in train if t["trajectory_status"] == "valid") for turn in range(1, 8)}
    results = {}
    for name in ("outcome", "ours", "no_filtering", "minus_effectiveness"):
        config = dict(load_main_config(ROOT / f"configs/experiments/main/{name}.json", ROOT), output_dir=os.path.relpath(tmp_path, ROOT), run_id="check")
        results[name] = check(config, 0)
        assert results[name]["train_prompts_by_stage"] == reached and results[name]["train_prompts"] == sum(reached.values())
    assert len({(r["train_prompts"], r["sampled_responses"], r["responses_per_prompt"]) for r in results.values()}) == 1  # the same budget
    assert results["outcome"]["ready"] and results["outcome"]["problems"] == []
    assert not results["ours"]["ready"] and any("no Effectiveness predictor" in p for p in results["ours"]["problems"])
    assert [p for p in results["minus_effectiveness"]["problems"] if "rubric set" not in p] == []  # it only waits for its rubrics


def test_decided_roles_are_connected_and_a_stand_in_is_refused_for_main_artifacts():
    from src.common.main_config import refuse_stand_ins, stand_in_roles

    main = read_config(ROOT / MATRIX["base_config"])
    roles = main["role_models"]
    assert (roles["failure_analyzer"], roles["rubric_manager"], roles["rubric_scorer"]) == ("Sol", "Sol", "Jev")
    assert (roles["effectiveness_candidate_generator"], roles["effectiveness_compliance_judge"]) == ("Sol", "Jev")
    # the pinned ids, not the latest aliases; Jev directly, not its router
    assert roles["connected"] == {"Sol": True, "Jev": True}
    assert main["roles"]["failure_analyzer"]["model_id"] == main["roles"]["rubric_manager"]["model_id"] == "openai/gpt-6.1-sol"
    assert main["rubrics"]["scorer"]["backend"] == "jev" and main["rubrics"]["scorer"]["model_id"] == "typesafe/jev-1.13"
    labels = read_config(ROOT / main["environment"]["config"])
    assert labels["label_roles"]["candidate_generator"]["model_id"] == "openai/gpt-6.1-sol"
    assert labels["label_roles"]["compliance_judge"]["model_id"] == "typesafe/jev-1.13" and labels["role_models"]["connected"] == {"Sol": True, "Jev": True}
    every = ["failure_analyzer", "rubric_manager", "rubric_scorer", "effectiveness_candidate_generator", "effectiveness_compliance_judge"]
    assert stand_in_roles(main, every + ["doctor", "agent"]) == []
    # a role whose model is not connected stops rubric generation, labels and Appropriateness training
    unconnected = dict(main, role_models=dict(roles, connected={"Sol": False, "Jev": True}))
    assert stand_in_roles(unconnected, every) == ["failure_analyzer = Sol", "rubric_manager = Sol", "effectiveness_candidate_generator = Sol"]
    with pytest.raises(SystemExit, match="not connected"):
        refuse_stand_ins(unconnected, ["failure_analyzer", "rubric_manager"], "rubric generation", allowed=False)
    refuse_stand_ins(unconnected, ["failure_analyzer"], "wiring check", allowed=True)
    refuse_stand_ins(read_config(ROOT / "configs/serl_agentclinic.json"), ["rubric_scorer"], "diagnostic config", allowed=False)  # no decided roles there


def test_scorer_decisions_are_kept_for_an_independent_audit_and_need_no_written_reason():
    from scripts.scorer_audit_sample import sample, units
    from src.judges.rubric_scoring import normalize_categorical, parse_chat_scores

    main = read_config(ROOT / MATRIX["base_config"])
    roles = main["role_models"]
    assert roles["rubric_scorer"] == roles["effectiveness_compliance_judge"] == "Jev" and "probabilities" in roles["decision_outputs"]["rubric_scorer"]
    rubrics = tuple(dict(schema_rubric(f"r{i}"), rubric_id=f"r{i}") for i in range(2))
    item = ScoringItem("i", "c", "t", 1, ({"turn_id": 0, "speaker": "doctor", "text": "d"}, {"turn_id": 1, "speaker": "agent", "text": "a"}), rubrics)
    # a decision without any text is a complete score (chat scorer without rationale; typed scorer with probabilities)
    labels, _, problems = parse_chat_scores({"turn_id": 1, "scores": [{"rubric_id": "r0", "applicable": True, "score": 2},
                                                                      {"rubric_id": "r1", "applicable": False, "score": 0}]}, item)
    assert labels == {"r0": 2, "r1": "NA"} and problems == []
    typed = normalize_categorical(item, "jev", {"r0": {"label": "2", "probabilities": {"NA": 0.02, "0": 0.03, "1": 0.15, "2": 0.8}},
                                                "r1": {"label": "NA", "probabilities": {"NA": 0.9, "0": 0.05, "1": 0.03, "2": 0.02}}}, {})
    assert typed.status == "scored" and typed.labels == {"r0": 2, "r1": "NA"} and typed.probabilities["r0"]["2"] == 0.8
    # the reward log keeps the scorer's input and decision; the audit sample is stratified by rubric and label
    log = [{"sample_id": f"s{i}", "generation": i, "case_id": "c", "stage": "A1",
            "rubric_scores": {"A1": {"status": "scored", "labels": {"r0": i % 3, "r1": "NA"}, "input": item.payload(),
                                     "result": {**typed.to_dict(), "probabilities": typed.probabilities}}}} for i in range(12)]
    found = units(log)
    assert len(found) == 24 and found[0]["rubric"]["rubric_id"] in ("r0", "r1") and "rubrics" not in found[0]["input"]
    chosen = sample(found, 2, seed=0)
    assert len(chosen) == 8 and {(u["rubric_id"], str(u["scorer_label"])) for u in chosen} == {("r0", "0"), ("r0", "1"), ("r0", "2"), ("r1", "NA")}
    assert chosen == sample(found, 2, seed=0) and all(u["auditor_label"] is None and u["scorer_probabilities"] is not None for u in chosen)
