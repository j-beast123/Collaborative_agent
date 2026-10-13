"""The main rubric method: every iteration's set from its own failures, filtered and weighted by the effectiveness gap."""

from conftest import AGENTCLINIC, require_env_histories, require_local  # noqa: F401
import json
from pathlib import Path

import pytest

from scripts import build_rubrics, main_experiments
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import problems_of
from src.grpo.trainer import load_rubric_set
from src.judges.effectiveness_gap import effectiveness_gap, filter_and_weight
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.reward import turn_appropriateness, weighted_soft_min
from test_judges import CONFIG, analysis, construct_reply, schema_rubric, trajectory

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {"method": "effectiveness_gap", "carry_over": False, "high_score": 2, "low_score": 0, "min_group_n": 1,
             "population": "policy_train_turns", "engineering_safety_cap": None}


def row(effectiveness, **labels):
    return {"labels": labels, "effectiveness": effectiveness}


def test_gap_is_mean_effectiveness_at_score_2_minus_mean_at_score_0():
    observations = [row(0.4, k=2), row(0.2, k=2), row(-0.1, k=0), row(-0.3, k=0),
                    row(0.9, k=1),          # partial adherence: in neither group
                    row(0.9, k="NA"),       # not applicable
                    row(None, k=2),         # no Effectiveness value for this turn
                    row(0.9, other=2)]      # another rubric
    measured = effectiveness_gap(observations, "k")
    assert (measured["high_n"], measured["low_n"], measured["partial_n"], measured["applicable_n"]) == (2, 2, 1, 6)
    assert measured["applicable_without_effectiveness_n"] == 1
    assert measured["high_mean_effectiveness"] == pytest.approx(0.3) and measured["low_mean_effectiveness"] == pytest.approx(-0.2)
    assert measured["effectiveness_gap"] == pytest.approx(0.5) and measured["status"] == "retained"


def test_positive_gap_is_kept_with_weight_gap_and_everything_else_is_removed_never_negatively_weighted():
    rubrics = [{"rubric_id": name} for name in ("helps", "hurts", "flat", "never_low", "never_scored")]
    observations = [row(0.5, helps=2, hurts=0, flat=2, never_low=2), row(0.1, helps=0, hurts=2, flat=0, never_low=2),
                    row(0.3, helps=1, hurts=1, flat=1, never_low=1), row(0.3, helps="NA", hurts="NA", flat="NA", never_low="NA"),
                    row(0.5, flat=0), row(0.1, flat=2)]
    kept, table = filter_and_weight(rubrics, observations)
    by = {r["rubric_id"]: r for r in table}
    assert [(r["rubric_id"], r["weight"]) for r in kept] == [("helps", pytest.approx(0.4))]
    assert by["helps"]["retained"] and by["helps"]["final_weight"] == pytest.approx(0.4) and by["helps"]["removal_reason"] is None
    assert by["hurts"]["effectiveness_gap"] == pytest.approx(-0.4) and by["hurts"]["removal_reason"] == "nonpositive_effectiveness_gap"
    assert by["flat"]["effectiveness_gap"] == pytest.approx(0.0) and by["flat"]["removal_reason"] == "nonpositive_effectiveness_gap"
    # an empty group is not a gap of 0: the rubric is removed for lack of support and its gap stays empty
    assert (by["never_low"]["high_n"], by["never_low"]["low_n"], by["never_low"]["effectiveness_gap"]) == (2, 0, None)
    assert by["never_low"]["removal_reason"] == by["never_scored"]["removal_reason"] == "insufficient_support"
    assert by["never_low"]["high_mean_effectiveness"] == pytest.approx(0.3) and by["never_low"]["low_mean_effectiveness"] is None
    assert all(r["final_weight"] is None for r in table if not r["retained"]) and all(r["weight"] > 0 for r in kept)
    # a stricter support rule is a setting; nothing is retained here with three responses per group required
    assert filter_and_weight(rubrics, observations, min_group_n=3)[0] == []


def test_weights_are_normalized_over_the_applicable_retained_rubrics():
    weights = {"a": 0.3, "b": 0.1}  # effectiveness gaps
    both = turn_appropriateness({"a": 2, "b": 0}, weights, "weighted_softmin", 0.5)
    assert both == pytest.approx(weighted_soft_min([1.0, 0.0], [0.75, 0.25], 0.5))  # alpha = gap / sum of gaps
    assert turn_appropriateness({"a": 2, "b": 0}, {"a": 3.0, "b": 1.0}, "weighted_softmin", 0.5) == pytest.approx(both)  # scale-free
    assert turn_appropriateness({"a": "NA", "b": 2}, weights, "weighted_softmin", 0.5) == pytest.approx(1.0)  # one applicable rubric


def gap_config(**extra):
    roles = dict(CONFIG["roles"], rubric_manager=dict(CONFIG["roles"]["rubric_manager"], retention_policy="current_only"))
    return dict(CONFIG, roles=roles, rubric_importance={"strategy": "effectiveness_gap"},
                rubric_registry={"max_active_categories": None, "schedule": "evolving", "fixed_rubric_set": None, "initial_rubrics": None,
                                 "selection": dict(SELECTION, **extra)})


def manager(calls):
    """A rubric manager that opens one category per distinct candidate name and writes its canonical rubric."""
    def factory(role, name):
        def reply(request):
            calls.append(name)
            if name == "failure_analyzer":
                tid = json.loads(request.user.split("\n\n", 1)[1])["trajectory_id"]
                return analysis(tid, f"Behavior {tid[0]}")  # trajectories a1, a2 share a behavior; b1 has its own
            if request.user.startswith('{"categories"'):
                return json.dumps({"merges": []})
            if "existing_categories" in request.user:
                payload = json.loads(request.user)
                wanted = payload["candidate"]["rubric"]["name"]
                same = [c for c in payload["existing_categories"] if c["name"] == wanted]
                if same:
                    return json.dumps({"decision": "existing", "category_id": same[0]["category_id"], "new_category": None, "reason": "same"})
                return json.dumps({"decision": "new", "category_id": None, "reason": "new", "new_category": {"name": wanted, "definition": "d"}})
            return construct_reply(json.loads(request.user.split("\n\n", 1)[1])["category"]["name"] if "\n\n" in request.user else "rubric")
        return CachedChat(ScriptedBackend(reply), None)
    return factory


def write_trajectories(path, names):
    path.write_text("\n".join(json.dumps(trajectory(t)) for t in names))


def observer(effects):
    """Scored responses per rubric name: {name: [(label, effectiveness), ...]}."""
    seen = []

    def observe(rubrics):
        seen.append([r["name"] for r in rubrics])
        rows = []
        for rubric in rubrics:
            rows += [{"item_id": f"{rubric['rubric_id']}:{i}", "labels": {rubric["rubric_id"]: label}, "effectiveness": value}
                     for i, (label, value) in enumerate(effects.get(rubric["category_name"], []))]
        return rows
    observe.seen = seen
    return observe


def test_every_iteration_builds_its_set_from_its_own_failures_and_carries_nothing_over(tmp_path, monkeypatch):
    calls = []
    factory = manager(calls)
    try:
        write_trajectories(tmp_path / "t0.jsonl", ["a1", "a2", "b1"])
        first = observer({"Behavior a": [(2, 0.4), (0, -0.2), (1, 0.9)], "Behavior b": [(2, -0.1), (0, 0.3)]})
        summary = build_rubrics.run(gap_config(), tmp_path / "t0.jsonl", tmp_path / "iter0", 0, chat_factory=factory, observe_fn=first)
    except KeyError as error:  # the scripted manager does not know this prompt layout
        pytest.fail(f"scripted manager: {error}")
    assert summary["consolidated_categories"] == 2 and summary["retained_rubrics"] == 1 and summary["carried_over_from_earlier_iterations"] == 0
    assert summary["removed"] == {"nonpositive_effectiveness_gap": 1}
    r0 = json.loads((tmp_path / "iter0/rubric_set.json").read_text())
    kept, = r0["rubrics"]
    assert kept["category_name"] == "Behavior a" and kept["weight"] == pytest.approx(0.6) == kept["effectiveness_gap"]
    assert (kept["high_n"], kept["low_n"]) == (1, 1) and sorted(kept["source_trajectories"]) == ["a1", "a2"]
    assert r0["selection_method"] == "effectiveness_gap" and r0["max_active_categories"] is None and r0["carried_over_from_earlier_iterations"] == []
    table = {r["category"]: r for r in json.loads((tmp_path / "iter0/rubric_table.json").read_text())["rows"]}
    for field in ("source_failures", "validation_success_rate", "high_n", "low_n", "high_mean_effectiveness", "low_mean_effectiveness",
                  "effectiveness_gap", "retained", "final_weight", "removal_reason"):
        assert field in table["Behavior a"], field
    assert table["Behavior b"]["retained"] is False and table["Behavior b"]["effectiveness_gap"] == pytest.approx(-0.4)
    assert table["Behavior b"]["final_weight"] is None and table["Behavior b"]["removal_reason"] == "nonpositive_effectiveness_gap"
    archive = {c["name"]: c for c in json.loads((tmp_path / "iter0/categories.json").read_text())}
    assert archive["Behavior a"]["status"] == "retained" and archive["Behavior b"]["status"] == "removed"
    assert not any(word in json.dumps(archive) for word in ("inactive", "reactivat"))
    for name in ("failure_analyses.jsonl", "category_assignments.json", "consolidated_rubrics.json", "gap_observations.jsonl",
                 "rubric_table.json", "rubric_set.json", "categories.json", "summary.json"):
        assert (tmp_path / "iter0" / name).is_file(), name

    # iteration 1: other failures. "Behavior a" does not recur, so it is not in the set, although the earlier
    # categories file is passed the way the runner passes it; "Behavior c" is new and "Behavior b" now has a positive gap
    write_trajectories(tmp_path / "t1.jsonl", ["b2", "c1"])
    second = observer({"Behavior b": [(2, 0.5), (0, 0.1)], "Behavior c": [(2, 0.2), (2, 0.4), (1, 0.0)], "Behavior a": [(2, 0.9), (0, -0.9)]})
    summary = build_rubrics.run(gap_config(), tmp_path / "t1.jsonl", tmp_path / "iter1", 1, chat_factory=factory, observe_fn=second,
                                previous_categories=tmp_path / "iter0/categories.json")
    r1 = json.loads((tmp_path / "iter1/rubric_set.json").read_text())
    assert second.seen == [["Behavior b", "Behavior c"]] or sorted(second.seen[0]) == ["Behavior b", "Behavior c"]  # only this iteration's
    assert [(r["category_name"], r["source_trajectories"]) for r in r1["rubrics"]] == [("Behavior b", ["b2"])]
    assert r1["rubrics"][0]["weight"] == pytest.approx(0.4)
    assert summary["removed"] == {"insufficient_support": 1}  # Behavior c has no response at score 0
    assert not set(r["rubric_id"] for r in r1["rubrics"]) & {kept["rubric_id"]}
    assert all(c["iteration"] == 1 and list(c["frequency_history"]) in ([], ["1"]) for c in json.loads((tmp_path / "iter1/categories.json").read_text()))

    # no cap: every category with a positive gap is in the set
    many = [f"{chr(ord('c') + i)}9" for i in range(12)]
    write_trajectories(tmp_path / "t2.jsonl", many)
    build_rubrics.run(gap_config(), tmp_path / "t2.jsonl", tmp_path / "iter2", 2, chat_factory=factory,
                      observe_fn=observer({f"Behavior {name[0]}": [(2, 0.2), (0, 0.1)] for name in many}))
    assert len(json.loads((tmp_path / "iter2/rubric_set.json").read_text())["rubrics"]) == 12
    # an iteration without any positive gap has an empty set and says so
    empty = build_rubrics.run(gap_config(), tmp_path / "t1.jsonl", tmp_path / "iter3", 3, chat_factory=factory, observe_fn=observer({}))
    assert empty["retained_rubrics"] == 0 and "warning" in empty
    with pytest.raises(IntegrationUnavailable, match="observe_fn"):
        build_rubrics.run(gap_config(), tmp_path / "t1.jsonl", tmp_path / "iter4", 4, chat_factory=factory)


def test_validation_comes_first_and_a_rejected_candidate_never_reaches_the_gap(tmp_path):
    write_trajectories(tmp_path / "t.jsonl", ["a1", "b1"])
    config = gap_config()
    config["validation"] = {"enabled": True, "num_reruns": 3, "min_successes": 2}

    def rerun(trajectory_, turn, instruction, index):  # the ideal behavior repairs a1 and never b1
        return {"status": "completed", "correct": trajectory_["trajectory_id"] == "a1", "final_diagnosis": "x", "final_option": "A",
                "is_correct": trajectory_["trajectory_id"] == "a1"}

    watch = observer({"Behavior a": [(2, 0.3), (0, 0.1)], "Behavior b": [(2, 0.9), (0, -0.9)]})
    summary = build_rubrics.run(config, tmp_path / "t.jsonl", tmp_path / "out", 0, chat_factory=manager([]), rerun_fn=rerun, observe_fn=watch)
    assert watch.seen == [["Behavior a"]] and summary["validated_candidates"] == 1 and summary["retained_rubrics"] == 1
    rows = json.loads((tmp_path / "out/rubric_table.json").read_text())["rows"]
    rejected, = [r for r in rows if r["removal_reason"] == "validation_failed"]
    assert rejected["source_failures"] == ["b1"] and rejected["level"] == "candidate" and rejected["validation_success_rate"] == 0.0
    assert "effectiveness_gap" not in rejected and summary["removed"] == {"validation_failed": 1}


def test_main_config_uses_the_gap_method_and_a_set_with_a_non_positive_weight_is_refused(tmp_path):
    require_env_histories("configs/serl_agentclinic_max7.json")
    main = read_config(ROOT / "configs/serl_agentclinic_max7.json")
    registry = main["rubric_registry"]
    assert registry["selection"]["method"] == "effectiveness_gap" and registry["selection"]["carry_over"] is False
    assert registry["max_active_categories"] is None and registry["selection"]["engineering_safety_cap"] is None
    assert (registry["selection"]["high_score"], registry["selection"]["low_score"]) == (2, 0)
    assert main["rubric_importance"]["strategy"] == "effectiveness_gap" and main["roles"]["rubric_manager"]["retention_policy"] == "current_only"
    assert problems_of(main, ROOT) == []
    for change, message in (({"rubric_importance": {"strategy": "failure_frequency"}}, "rubric_importance.strategy"),
                            ({"rubric_registry": dict(registry, max_active_categories=8)}, "no cap"),
                            ({"rubric_registry": dict(registry, selection=dict(registry["selection"], carry_over=True))}, "carry_over"),
                            ({"rubric_registry": dict(registry, selection=dict(registry["selection"], method="failure_frequency_top_n"))},
                             "the main protocol uses")):
        assert any(message in p for p in problems_of({**main, **change}, ROOT)), message
    # the fixed two-turn configuration keeps the selection it was built with
    assert problems_of(read_config(ROOT / "configs/serl_agentclinic.json"), ROOT) == []

    def write(weights):
        (tmp_path / "set.json").write_text(json.dumps({"rubric_set_version": "v", "rubrics": [
            dict(schema_rubric(), rubric_id=f"r{i}", category_id=f"c{i}", weight=w) for i, w in enumerate(weights)]}))

    run = {"output_dir": ".", "rubrics": {"rubric_set": "set.json"}, "rubric_registry": registry, "reward": {"weights": {"appropriateness": 0.3}}}
    write([0.4, 0.05] + [0.1] * 10)
    assert len(load_rubric_set(run, tmp_path, 0)["rubrics"]) == 12  # no cap
    for bad in ([0.4, -0.1], [0.4, 0.0]):
        write(bad)
        with pytest.raises(IntegrationUnavailable, match="positive weight"):
            load_rubric_set(run, tmp_path, 0)


def test_the_three_rubric_conditions_are_stated_apart():
    rows = {r["condition"]: r for r in main_experiments.plan(main_experiments.load_matrix())["conditions"]}
    described = {name: (rows[name]["filtering"], rows[name]["regenerated_each_iteration"], rows[name]["carries_over_earlier_rubrics"],
                        rows[name]["cumulative"]) for name in ("static_rubric", "no_filtering", "ours")}
    assert described == {"static_rubric": (True, False, True, False),   # one filtered set, reused
                         "no_filtering": (False, True, True, True),     # raw candidates, accumulated
                         "ours": (True, True, False, False)}            # a new filtered set every iteration
    assert rows["ours"]["weights"].startswith("effectiveness gap") and rows["no_filtering"]["weights"] == "1 per raw candidate"


def test_rubric_budget_keeps_the_largest_gaps_and_never_fills_up():
    """max_rubrics is a maximum: the largest gaps among the rubrics that pass the filter; fewer pass -> all of them."""
    from src.judges.effectiveness_gap import BEYOND_BUDGET, filter_and_weight

    rubrics = [{"rubric_id": name} for name in ("a", "b", "c", "d")]
    gaps = {"a": 0.1, "b": 0.4, "c": -0.2, "d": 0.3}
    observations = []
    for name, gap in gaps.items():
        observations += [{"labels": {name: 2}, "effectiveness": gap}, {"labels": {name: 0}, "effectiveness": 0.0}]
    kept, rows = filter_and_weight(rubrics, observations, max_rubrics=2)
    assert [r["rubric_id"] for r in kept] == ["b", "d"] and [round(r["weight"], 6) for r in kept] == [0.4, 0.3]
    reasons = {r["rubric_id"]: r["removal_reason"] for r in rows}
    assert reasons == {"a": BEYOND_BUDGET, "b": None, "c": "nonpositive_effectiveness_gap", "d": None}
    assert next(r for r in rows if r["rubric_id"] == "a")["effectiveness_gap"] == pytest.approx(0.1)  # the measured gap is kept in the table
    assert [r["rubric_id"] for r in filter_and_weight(rubrics, observations, max_rubrics=8)[0]] == ["a", "b", "d"]  # 3 pass: 3 are used
    assert [r["rubric_id"] for r in filter_and_weight(rubrics, observations)[0]] == ["a", "b", "d"]  # no budget: as before
