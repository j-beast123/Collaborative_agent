"""The active rubric set: capped, chosen in a fixed order, a view on a registry that keeps everything."""

from conftest import AGENTCLINIC, require_env_histories, require_local  # noqa: F401
import json
from pathlib import Path

import pytest

from src.common.config import IntegrationUnavailable
from src.grpo.trainer import load_rubric_set
from src.judges.llm import CachedChat, ScriptedBackend
from src.judges.reward import group_mean_imputation, turn_appropriateness
from src.judges.rubric_consolidation import build_final_rubrics, update_registry
from src.judges.rubric_retention import build_retention
from test_judges import schema_rubric
from test_serl_steps import valid_record

ROOT = Path(__file__).resolve().parents[1]


def category(cid: str, records: list[dict], created: int = 0) -> dict:
    return {"category_id": cid, "name": cid, "definition": "d", "members": [r["rubric_id"] for r in records],
            "created_iteration": created, "status": "open", "frequency_history": {}}


def build(registry, records, iteration, *, cap=8, validations=None, rubrics_per_category=None):
    """One iteration of selection and registry update; the manager writes one rubric per category unless told otherwise."""
    def reply(request):
        name = json.loads(request.user.split("\n\n", 1)[1])["category"]["name"]
        count = (rubrics_per_category or {}).get(name, 1)
        return json.dumps({"rubrics": [schema_rubric(f"{name} #{i}", user=f"context {i}") for i in range(count)]})

    chat = CachedChat(ScriptedBackend(reply), None)
    rubric_set = build_final_rubrics(registry, records, chat, system_prompt="s",
                                     prompt_version="v", params={}, iteration=iteration, max_active_categories=cap,
                                     retention=build_retention("reactivation"), importance="failure_frequency",
                                     validations=validations, canonical=True)
    assert all("exactly one rubric" in call.user and call.prompt_version == "v+canonical-v1" for call in chat.backend.calls)
    assert len({r["category_id"] for r in rubric_set["rubrics"]}) == len(rubric_set["rubrics"]) <= cap
    return rubric_set, update_registry(registry, rubric_set, records, iteration, validations)


def population(sizes: dict[str, int]) -> tuple[list[dict], list[dict]]:
    records = {cid: [valid_record(f"{cid}-{i}") for i in range(n)] for cid, n in sizes.items()}
    return [category(cid, members) for cid, members in records.items()], [r for members in records.values() for r in members]


def test_active_set_never_exceeds_the_cap_and_takes_the_most_frequent_categories():
    registry, records = population({f"cat-{i:03d}": i + 1 for i in range(12)})  # cat-011 has 12 failures, cat-000 one
    rubric_set, registry = build(registry, records, 0)
    selection = rubric_set["selection"]
    assert len(rubric_set["rubrics"]) == 8 == selection["active_category_count"] == selection["max_active_categories"]
    assert selection["selected_active_categories"] == [f"cat-{i:03d}" for i in range(11, 3, -1)]
    assert selection["fewer_than_max_reason"] is None and selection["all_registry_categories"] == 12
    # the others are inactive in the registry, with rank and reason; nothing is deleted
    assert len(registry) == 12 and sum(c["status"] == "active" for c in registry) == 8
    left_out = next(c for c in registry if c["category_id"] == "cat-003")
    assert left_out["status"] == "inactive" and left_out["selection_rank"] == 9 and "top 8" in left_out["inactive_reason"]
    assert left_out["frequency_history"] == {"0": 4} and left_out["members"]
    row = next(r for r in selection["categories"] if r["category_id"] == "cat-003")
    assert row["eligible"] and not row["selected_active"] and row["current_frequency"] == 4 and row["historical_frequency"] == 4
    # weights are the failure shares among the active categories
    weights = {r["category_id"]: r["weight"] for r in rubric_set["rubrics"]}
    assert weights["cat-011"] == pytest.approx(12 / sum(range(5, 13))) and sum(weights.values()) == pytest.approx(1.0)


def test_fewer_eligible_categories_than_the_cap_are_not_padded():
    registry, records = population({"cat-001": 3, "cat-002": 1, "cat-003": 2})
    rubric_set, _ = build(registry, records, 0)
    selection = rubric_set["selection"]
    assert len(rubric_set["rubrics"]) == 3 and selection["active_category_count"] == 3
    assert "3 eligible categories" in selection["fewer_than_max_reason"] and "not filled up" in selection["fewer_than_max_reason"]
    assert [r["category_id"] for r in rubric_set["rubrics"]] == ["cat-001", "cat-003", "cat-002"]


def test_each_active_category_gives_the_scorer_one_canonical_rubric_and_variants_take_no_slot():
    registry, records = population({"cat-001": 5, "cat-002": 4, "cat-003": 3, "cat-004": 2})
    # the manager writes three rubrics for cat-002: the first is its canonical rubric, the others are provenance
    rubric_set, registry = build(registry, records, 0, cap=3, rubrics_per_category={"cat-002": 3})
    assert [r["category_id"] for r in rubric_set["rubrics"]] == ["cat-001", "cat-002", "cat-003"]  # three categories, three rubrics
    assert rubric_set["selection"]["active_category_count"] == 3 == rubric_set["selection"]["active_rubric_count"]
    second = next(c for c in registry if c["category_id"] == "cat-002")
    assert second["status"] == "active" and second["canonical_rubric"]["name"] == "cat-002 #0" and second["rubrics"] == [second["canonical_rubric"]]
    assert [v["name"] for v in second["variants"]] == ["cat-002 #1", "cat-002 #2"] and len(second["members"]) == 4
    assert rubric_set["variants"] == {"cat-002": second["variants"]}
    left_out = next(c for c in registry if c["category_id"] == "cat-004")
    assert left_out["status"] == "inactive" and "top 3" in left_out["inactive_reason"]
    # the manager was asked for exactly one rubric per category
    first = next(c for c in registry if c["category_id"] == "cat-001")
    assert "variants" not in first and first["canonical_rubric"] == first["rubrics"][0]
    # diagnostic fields of a category: recorded, never used for the order or the weights
    row = next(r for r in rubric_set["selection"]["categories"] if r["category_id"] == "cat-002")
    assert {"validation_success_rate", "effectiveness_gap", "task_success_gap", "current_frequency", "historical_frequency"} <= set(row)
    assert rubric_set["rubric_importance"] == "failure_frequency"
    assert {r["category_id"]: r["weight"] for r in rubric_set["rubrics"]} == pytest.approx({"cat-001": 5 / 12, "cat-002": 4 / 12, "cat-003": 3 / 12})


def test_selection_order_is_frequency_then_validation_then_history_then_id():
    def run(order):
        registry, records = population({"cat-b": 2, "cat-a": 2, "cat-c": 2, "cat-d": 2, "cat-e": 3})
        by_id = {c["category_id"]: c for c in registry}
        by_id["cat-c"]["frequency_history"] = {"0": 5}  # seen often before
        validations = {r["rubric_id"]: {"num_reruns": 3, "successes": 3 if r["source_trajectory_id"].startswith("cat-d") else 2}
                       for r in records}
        rubric_set, _ = build([by_id[c] for c in order], records, 1, cap=3, validations=validations)
        return rubric_set["selection"]

    first = run(["cat-a", "cat-b", "cat-c", "cat-d", "cat-e"])
    # most failures first; among the ties the better validated (cat-d), then the historically frequent (cat-c), then the id
    assert first["selected_active_categories"] == ["cat-e", "cat-d", "cat-c"]
    assert [r["category_id"] for r in sorted(first["categories"], key=lambda r: r["selection_rank"])] == \
        ["cat-e", "cat-d", "cat-c", "cat-a", "cat-b"]
    assert next(r for r in first["categories"] if r["category_id"] == "cat-d")["validation_success_rate"] == 1.0
    assert run(["cat-e", "cat-d", "cat-c", "cat-b", "cat-a"]) == first  # the input order does not matter


def test_inactive_category_stays_in_the_registry_and_is_reactivated_when_its_failure_returns():
    a0, b0 = [valid_record("a0"), valid_record("a0b")], [valid_record("b0")]
    rubric_set, registry = build([category("cat-A", a0), category("cat-B", b0)], a0 + b0, 0, cap=1)
    assert [r["category_id"] for r in rubric_set["rubrics"]] == ["cat-A"]
    by_id = {c["category_id"]: c for c in registry}
    assert by_id["cat-B"]["status"] == "inactive" and "rubrics" not in by_id["cat-B"]  # never active: no rubric written

    # iteration 1: B's failure dominates, A is observed less and falls out of the active set
    a1, b1 = [valid_record("a1")], [valid_record("b1"), valid_record("b1b"), valid_record("b1c")]
    by_id["cat-A"]["members"] += [r["rubric_id"] for r in a1]
    by_id["cat-B"]["members"] += [r["rubric_id"] for r in b1]
    rubric_set, registry = build(registry, a1 + b1, 1, cap=1)
    assert [(r["category_id"], r["retention_reason"]) for r in rubric_set["rubrics"]] == [("cat-B", "observed this iteration")]
    dormant = next(c for c in registry if c["category_id"] == "cat-A")
    assert dormant["status"] == "inactive" and dormant["rubrics"][0]["name"] == "cat-A #0"  # its rubric is kept
    assert dormant["frequency_history"] == {"0": 2, "1": 1} and dormant["reactivation_count"] == 0

    # iteration 2: A's failure returns in force: it is active again with its earlier rubric, counted as a reactivation
    a2 = [valid_record("a2"), valid_record("a2b")]
    dormant["members"] += [r["rubric_id"] for r in a2]
    rubric_set, registry = build(registry, a2, 2, cap=1)
    rubric, = rubric_set["rubrics"]
    assert (rubric["category_id"], rubric["origin"], rubric["retention_reason"], rubric["name"]) == \
        ("cat-A", "reused", "reactivated", "cat-A #0")
    back = next(c for c in registry if c["category_id"] == "cat-A")
    assert back["status"] == "active" and back["reactivation_count"] == 1 and back["active_iterations"] == [0, 2]
    row = next(r for r in rubric_set["selection"]["categories"] if r["category_id"] == "cat-A")
    assert row["reactivation_status"] == "reactivated" and row["historical_frequency"] == 5


def test_weighted_appropriateness_ignores_no_match_in_numerator_and_denominator():
    weights = {"r1": 0.6, "r2": 0.3, "r3": 0.1}
    assert turn_appropriateness({"r1": 2, "r2": 0, "r3": 1}, weights) == pytest.approx(0.6 * 1.0 + 0.3 * 0.0 + 0.1 * 0.5)
    # r2 does not apply: the mean is over r1 and r3 only (0.65 / 0.7), not over all three (0.65)
    assert turn_appropriateness({"r1": 2, "r2": "NA", "r3": 1}, weights) == pytest.approx((0.6 * 1.0 + 0.1 * 0.5) / 0.7)
    assert turn_appropriateness({"r1": "NA", "r2": "NA", "r3": "NA"}, weights) is None  # no value, not 0


def test_group_mean_imputation():
    assert group_mean_imputation([0.8, 0.4, None, 0.6]) == pytest.approx([0.8, 0.4, 0.6, 0.6])
    assert group_mean_imputation([None, None]) == [None, None]
    assert group_mean_imputation([0.2]) == [0.2]


def test_reward_refuses_more_active_categories_than_the_cap_or_two_rubrics_of_one_category(tmp_path):
    def write(categories):
        (tmp_path / "set.json").write_text(json.dumps({"rubric_set_version": "v", "rubrics": [
            dict(schema_rubric(), rubric_id=f"r{i}", category_id=c, weight=1 / len(categories)) for i, c in enumerate(categories)]}))

    config = {"output_dir": "out", "rubrics": {"rubric_set": "set.json"}, "rubric_registry": {"max_active_categories": 8},
              "reward": {"weights": {"outcome": 0.4, "appropriateness": 0.3, "effectiveness": 0.3}}}
    write([f"cat-{i}" for i in range(9)])
    with pytest.raises(IntegrationUnavailable, match="max_active_categories is 8"):
        load_rubric_set(config, tmp_path, 0)
    write([f"cat-{i}" for i in range(8)])
    assert len(load_rubric_set(config, tmp_path, 0)["rubrics"]) == 8
    write(["cat-0", "cat-0", "cat-1"])
    with pytest.raises(IntegrationUnavailable, match="one canonical rubric per active category"):
        load_rubric_set(config, tmp_path, 0)


def test_fixed_schedule_uses_r0_as_it_is_and_calls_no_rubric_model(tmp_path):
    require_env_histories("configs/serl_agentclinic_fixed.json")
    from scripts import build_rubrics
    from src.common.config import read_config
    from test_judges import CONFIG, analysis, manager_responses, trajectory

    for iteration in (0, 1):
        (tmp_path / f"out/iter{iteration}/rubrics").mkdir(parents=True)
        (tmp_path / f"out/iter{iteration}/rubrics/rubric_set.json").write_text(json.dumps(
            {"rubric_set_version": f"R{iteration}", "rubrics": [dict(schema_rubric(), rubric_id="r", weight=1.0)]}))
    config = {"output_dir": "out", "rubrics": {"rubric_set": "iter{iteration}/rubrics/rubric_set.json"},
              "rubric_registry": {"max_active_categories": 8, "schedule": "evolving", "fixed_rubric_set": None}}
    assert [load_rubric_set(config, tmp_path, i)["rubric_set_version"] for i in (0, 1)] == ["R0", "R1"]
    config["rubric_registry"].update(schedule="fixed", fixed_rubric_set="iter0/rubrics/rubric_set.json")
    assert [load_rubric_set(config, tmp_path, i)["rubric_set_version"] for i in (0, 1)] == ["R0", "R0"]  # content and weights

    # the rubric step: Evolving calls the analyzer and the manager; Fixed records the failures and calls nothing
    rows = [trajectory("t1"), trajectory("t2"), trajectory("t3", correct=True)]
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    calls = []

    def factory(role, name):
        def reply(request):
            calls.append(name)
            return analysis(json.loads(request.user.split("\n\n", 1)[1])["trajectory_id"], "Yield") \
                if name == "failure_analyzer" else manager_responses(request)
        return CachedChat(ScriptedBackend(reply), None)

    def no_rerun(*_):
        raise AssertionError("a fixed run must not rerun a collaboration")

    fixed = dict(CONFIG, rubric_registry={"max_active_categories": 8, "schedule": "fixed", "fixed_rubric_set": "r0.json"})
    summary = build_rubrics.run(fixed, tmp_path / "traj.jsonl", tmp_path / "fixed", 1, chat_factory=factory, rerun_fn=no_rerun)
    assert calls == [] and summary["failures_recorded"] == 2 and summary["rubric_schedule"] == "fixed"
    assert sorted(p.name for p in (tmp_path / "fixed").iterdir()) == ["failure_selection.json", "failures.jsonl", "summary.json"]
    assert [json.loads(line)["trajectory_id"] for line in (tmp_path / "fixed/failures.jsonl").read_text().splitlines()] == ["t1", "t2"]

    evolving = dict(CONFIG, rubric_registry={"max_active_categories": 8, "schedule": "evolving", "fixed_rubric_set": None})
    build_rubrics.run(evolving, tmp_path / "traj.jsonl", tmp_path / "evolving", 0, chat_factory=factory)
    assert calls.count("failure_analyzer") == 2 and "rubric_manager" in calls
    for name in ("rubric_set.json", "active_selection.json", "rubric_report.json", "categories.json"):
        assert (tmp_path / "evolving" / name).is_file()
    registry = json.loads((tmp_path / "evolving/categories.json").read_text())
    assert all(len(c["rubrics"]) == 1 and c["canonical_rubric"] == c["rubrics"][0] for c in registry if c["status"] == "active")
    # the optional post-hoc diagnostic on a fixed run is the full pipeline, marked as such
    before = len(calls)
    summary = build_rubrics.run(fixed, tmp_path / "traj.jsonl", tmp_path / "posthoc", 1, chat_factory=factory, post_hoc=True)
    assert len(calls) > before and "post_hoc_diagnostic" in summary

    # the two conditions differ in the rubric schedule only; a fixed run needs an existing R_0 before it starts
    evolving_config, fixed_config = read_config(ROOT / "configs/serl_agentclinic.json"), read_config(ROOT / "configs/serl_agentclinic_fixed.json")
    assert {k for k in evolving_config if evolving_config[k] != fixed_config.get(k)} == {"notes", "run_label", "rubric_registry"}
    assert fixed_config["rubric_registry"]["max_active_categories"] == 8 and fixed_config["reward"] == evolving_config["reward"]
    from src.common.main_config import problems_of
    assert any("fixed_rubric_set" in p for p in problems_of(fixed_config, ROOT))
    fixed_config["rubric_registry"]["fixed_rubric_set"] = "configs/pilot/e2e_rubric_fixture.json"  # any existing file
    assert problems_of(fixed_config, ROOT) == []


def test_controlled_evidence_conditions_are_an_evaluation_path_only():
    """The four controlled D1 conditions are built by the evaluation code; nothing that produces training
    samples, rewards or rubrics imports or names them."""
    training = [ROOT / "src/grpo" / name for name in ("dataset.py", "environment.py", "reward.py", "trainer.py")] + \
        [ROOT / "scripts" / name for name in ("run_grpo.py", "build_rubrics.py")]
    for path in training:
        text = path.read_text()
        assert "controlled" not in text.lower() and "agentclinic.evaluation" not in text and "evidence_generator" not in text, path
    main = json.loads((ROOT / "configs/serl_agentclinic.json").read_text())
    assert "evidence" not in main and "expected_actions" not in main
    assert "evidence" in json.loads((ROOT / "configs/eval/agentclinic.json").read_text())


def test_pilot_compares_uniform_and_failure_frequency_weights_offline():
    """The same rubric scores under both weightings: the R_A values, and where the order of the candidate responses
    to one D1 changes. A diagnostic for the open choice of the weight source; nothing is decided or trained."""
    from scripts.rubric_pilot import rubric_table, weighting_comparison

    config = {"rubric_aggregation": {"method": "weighted_softmin", "temperature": 0.5}}
    rubric_set = {"rubrics": [dict(schema_rubric("heavy"), rubric_id="r1", category_id="cat-1", failure_count=9, weight=0.9),
                              dict(schema_rubric("light"), rubric_id="r2", category_id="cat-2", failure_count=1, weight=0.1)]}

    def candidate(history, name, r1, r2, delta):
        return {"item_id": f"{history}:{name}", "history_id": history, "candidate_id": name, "status": "scored",
                "labels": {"r1": r1, "r2": r2}, "after_rate": 0.5 + delta, "before_rate": 0.5, "delta": delta}

    candidates = [candidate("h1", "a", 2, 0, 0.4),   # good on the frequent rubric, bad on the rare one
                  candidate("h1", "b", 1, 2, -0.2),  # the other way round
                  candidate("h2", "a", 2, 2, 0.3), candidate("h2", "b", 0, 0, -0.3),
                  candidate("h3", "a", "NA", "NA", 0.0), candidate("h3", "b", 2, "NA", 0.1)]  # one response without R_A: no group
    result = weighting_comparison(config, rubric_set, [], candidates)
    assert result["responses_with_R_A"] == 5 and result["candidate_groups"] == 2
    # h1: failure_frequency prefers a (the 0 has weight 0.1), uniform prefers b; h2: the same order under both
    assert result["groups_with_the_same_order"] == 1 and result["groups_with_a_different_best_candidate"] == 1
    assert result["candidate_pairs"] == 2 and result["pairs_ordered_oppositely"] == 1 and result["responses_where_R_A_differs"] == 2
    agreement = result["diagnostic_agreement_with_the_K9_delta_order"]
    assert agreement["failure_frequency"]["share_ordered_like_the_delta"] == 1.0 and agreement["uniform"]["share_ordered_like_the_delta"] == 0.5
    assert result["weights"]["uniform"] == {"r1": 0.5, "r2": 0.5} and result["weights"]["failure_frequency"] == {"r1": 0.9, "r2": 0.1}

    selection = {"categories": [{"category_id": "cat-1", "historical_frequency": 12, "validation_success_rate": 0.67}]}
    heavy, light = rubric_table(rubric_set, selection, [], [], candidates)
    assert heavy["task_success_gap"] == pytest.approx((0.9 + 0.8 + 0.6) / 3 - 0.2) and heavy["effectiveness_gap"] == pytest.approx((0.4 + 0.3 + 0.1) / 3 + 0.3)
    assert heavy["applicability_on_A1_candidates"] == pytest.approx(5 / 6) and heavy["historical_frequency"] == 12
    assert (heavy["applicable_n"], heavy["score_2_n"], heavy["score_1_n"], heavy["score_0_n"]) == (5, 3, 1, 1)  # the n of the gaps
    assert heavy["controlled_candidates_scored"] == 6 and light["validation_success_rate"] is None


def test_pilot_failure_sample_is_seeded_and_spread_over_collaboration_states():
    from scripts.build_rubrics import sample_failures

    failures = [{"trajectory_id": f"t{i:02d}", "collaboration_state": state}
                for i, state in enumerate(["Recover"] * 20 + ["Challenge"] * 5 + ["Accept"] * 2 + [None] * 1)]
    chosen = sample_failures(failures, 12, seed=0)
    counts = {}
    for t in chosen:
        counts[str(t["collaboration_state"])] = counts.get(str(t["collaboration_state"]), 0) + 1
    # taken in turn from each state: the rare states are all in, the frequent one does not fill the sample
    assert len(chosen) == 12 and counts == {"Accept": 2, "None": 1, "Challenge": 5, "Recover": 4}
    assert chosen == sample_failures(list(reversed(failures)), 12, seed=0)  # fixed by the seed, not by the file order
    assert chosen != sample_failures(failures, 12, seed=1) and len(sample_failures(failures, 100, seed=0)) == 28


def test_outcome_only_ablation_differs_in_the_reward_weights_only_and_queries_no_rubric_or_predictor(tmp_path):
    require_env_histories("configs/serl_agentclinic_outcome_only.json", "configs/serl_agentclinic.json")
    from scripts import build_rubrics
    from src.common.config import read_config
    from src.common.main_config import problems_of, training_device_problems
    from src.grpo.trainer import NO_RUBRICS, build_provider
    from test_judges import CONFIG, trajectory

    main, ablation = read_config(ROOT / "configs/serl_agentclinic.json"), read_config(ROOT / "configs/serl_agentclinic_outcome_only.json")
    assert {k for k in main if main[k] != ablation.get(k)} == {"notes", "run_label", "reward"}
    assert {k for k in main["reward"] if main["reward"][k] != ablation["reward"][k]} == {"weights"}
    assert ablation["reward"]["weights"] == {"outcome": 1.0, "appropriateness": 0.0, "effectiveness": 0.0}
    for key in ("grpo", "policy", "training_samples", "environment", "rollout", "collaboration", "data", "seed"):
        assert ablation[key] == main[key], key  # the same samples, policy, GRPO settings and environment
    assert problems_of(ablation, ROOT) == [] and training_device_problems(ablation) == []  # no predictor, so one GPU is enough
    # no rubric set is read and no Effectiveness predictor is built
    assert load_rubric_set(dict(ablation, output_dir="nowhere"), tmp_path, 0) is NO_RUBRICS
    assert build_provider(ablation, 0, tmp_path) is None
    # the rubrics step records the failures and calls no model
    (tmp_path / "traj.jsonl").write_text("\n".join(json.dumps(trajectory(t)) for t in ("t1", "t2")))
    def never(*_):
        raise AssertionError("the outcome-only ablation must not call a rubric model")
    summary = build_rubrics.run(dict(CONFIG, reward=ablation["reward"], rubric_registry=ablation["rubric_registry"]),
                                tmp_path / "traj.jsonl", tmp_path / "out", 0, chat_factory=never, rerun_fn=never)
    assert summary["failures_recorded"] == 2 and "no Appropriateness" in summary["rubric_generation"]
