"""Small real-model rubric pilot on TRAIN failures, before any GRPO training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

from scripts import build_rubrics
from src.agentclinic.pipeline import Appender
from src.common import usage
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import load_main_config, open_run
from src.common.io import read_jsonl
from src.grpo.trainer import base_trajectories, build_scorer, trajectories_path
from src.judges.reward import appropriateness_detail, rubric_weights, soft_min, turn_appropriateness
from src.judges.rubric_schema import validate_rubric
from src.judges.rubric_scoring import build_scoring_items, final_rubric_view

ROOT = Path(__file__).resolve().parents[1]


def missing_settings(config: dict) -> list[str]:
    missing = [f"roles.{role}.model_id" for role in ("failure_analyzer", "rubric_manager")
               if not config["roles"][role].get("model_id")]
    scorer = config["rubrics"]["scorer"]
    if scorer["backend"] != "jev" and not scorer.get("model_id"):
        missing.append("rubrics.scorer.model_id")
    return missing


def train_failures(config: dict) -> tuple[int, int]:
    """(TRAIN trajectories, failed ones) of the base Agent, from the base trajectories when they exist,
    else from the label run's prepared collaborations."""
    path = trajectories_path(config, ROOT, 0)
    if path.is_file():
        rows = [t for t in read_jsonl(path) if t["trajectory_status"] == "valid"]
        return len(rows), sum(not t["is_correct"] for t in rows)
    source = read_config(ROOT / config["environment"]["config"])
    diagnostics = {r["case_id"]: r for r in read_jsonl(ROOT / source["output_dir"] / "prepare_diagnostics.jsonl", missing_ok=True)}
    from src.agentclinic.selection import split_eligible_cases

    train = [cid for cid, case in split_eligible_cases(source, ROOT).items() if case["split"] == "train" and cid in diagnostics]
    return len(train), sum(not diagnostics[cid]["natural_final_correct"] for cid in train)


def estimate(config: dict, max_failures: int, score_trajectories: int) -> dict:
    trajectories, failures = train_failures(config)
    analysed = min(max_failures, failures)
    reruns = analysed * config["validation"]["num_reruns"]
    scored = min(score_trajectories, trajectories)
    base_missing = not trajectories_path(config, ROOT, 0).is_file()
    return {
        "missing_settings": missing_settings(config), "ready": not missing_settings(config),
        "train_trajectories": trajectories, "failed_trajectories": failures, "failures_analysed": analysed,
        "agent_model_calls": {"state_elicitation_for_base_trajectories": 2 * trajectories if base_missing else 0,
                              "rerun_responses_upper_bound": 2 * reruns},
        "api_calls_upper_bound": {
            "diagnosis_judge_for_base_trajectories": 3 * trajectories if base_missing else 0,
            "failure_analyzer": analysed,
            "rerun_doctor": 3 * reruns, "rerun_diagnosis_judge": reruns,
            "rubric_manager_assign": analysed, "rubric_manager_reconcile": 1, "rubric_manager_construct": analysed,
            "rubric_scorer": 2 * scored},
        "notes": "Upper bounds: only valid analyses are rerun, only validated candidates are assigned, one construct call "
                 "per active category, exact-match diagnoses need no judge call, cached calls are free. Reruns run one at "
                 "a time (about 10-20 s each with the 8B Agent on one GPU); loading the Agent model takes a few minutes, twice.",
    }


def score_trajectories(config: dict, trajectories: list[dict], rubric_set: dict, out: Path, workers: int) -> list[dict]:
    """Rubric scores of A1 and A2 of every given trajectory (resumable)."""
    rubrics = [final_rubric_view(r) for r in rubric_set["rubrics"]]
    log = Appender(out / "scores.jsonl", key=lambda r: r["item_id"])
    if not rubrics:
        return []
    scorer = build_scorer(config, ROOT)
    items = [item for t in trajectories for item in build_scoring_items(t, rubrics, include_initial_turn=True)
             if item.item_id not in log.keys]

    def run(item):
        with usage.case(item.case_id):
            result = scorer.score(item)
        log.add({"item_id": item.item_id, "trajectory_id": item.trajectory_id, "case_id": item.case_id,
                 "turn_id": item.turn_id, "status": result.status, "labels": result.labels})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, items))
    return log.rows


def score_candidates(config: dict, trajectories: list[dict], rubric_set: dict, out: Path, workers: int, histories: int) -> list[dict]:
    """Rubric scores of the A1 label candidates of the first `histories` TRAIN histories: different responses to the
    same D1, scored like an A1 (with the state the base Agent had before A1).
    """
    rubrics = [final_rubric_view(r) for r in rubric_set["rubrics"]]
    log = Appender(out / "candidate_scores.jsonl", key=lambda r: r["item_id"])
    if not rubrics:
        return []
    source = read_config(ROOT / config["environment"]["config"])
    by_history = {t["history_id"]: t for t in trajectories}
    chosen = sorted(by_history)[:histories]
    labels = [r for r in read_jsonl(ROOT / source["output_dir"] / "labels_train.jsonl") if r["history_id"] in chosen]
    scorer = build_scorer(config, ROOT)

    def run(label):
        base = by_history[label["history_id"]]
        candidate = dict(base, trajectory_id=f"{label['history_id']}:candidate:{label['candidate_id']}",
                         turns=[base["turns"][0], dict(base["turns"][1], text=label["response"])])
        item, = build_scoring_items(candidate, rubrics, include_initial_turn=True)
        if item.item_id in log.keys:
            return
        with usage.case(item.case_id):
            result = scorer.score(item)
        log.add({"item_id": item.item_id, "history_id": label["history_id"], "case_id": item.case_id,
                 "candidate_id": label["candidate_id"], "status": result.status, "labels": result.labels,
                 "after_rate": label["after_rate"], "before_rate": label["before_rate"], "delta": label["delta_effectiveness"]})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, labels))
    return [r for r in log.rows if r["history_id"] in chosen]


def failure_points(analysis: dict, trajectories: dict[str, dict]) -> list[str]:
    """The stages (A1, A2) of the turns an analysis names as the failure."""
    trajectory = trajectories.get(str(analysis.get("source_trajectory_id", analysis.get("trajectory_id"))))
    turns = (analysis.get("analysis") or {}).get("failure_turn_ids") or []
    return [trajectory["turns"][i].get("stage", f"turn{i}") if trajectory and i < len(trajectory["turns"]) else f"turn{i}" for i in turns]


def weight_strategies(rubric_set: dict) -> dict[str, dict]:
    ids = [r["rubric_id"] for r in rubric_set["rubrics"]]
    return {"failure_frequency": rubric_weights(rubric_set) or {i: 1.0 for i in ids}, "uniform": {i: 1.0 / len(ids) for i in ids} if ids else {}}


def order(values: list[float]) -> tuple:
    return tuple(sorted(range(len(values)), key=lambda i: (-round(values[i], 9), i)))


def weighting_comparison(config: dict, rubric_set: dict, scores: list[dict], candidate_scores: list[dict]) -> dict:
    """R_A under uniform and under failure_frequency weights on the same rubric scores, and the order of the
    candidate responses to one D1 under each. Offline; the weight source is not decided here."""
    method, tau = config["rubric_aggregation"]["method"], config["rubric_aggregation"]["temperature"]
    strategies = weight_strategies(rubric_set)

    def value(row: dict, strategy: str):
        return turn_appropriateness(row["labels"], strategies[strategy], method, tau)

    rows = [r for r in scores + candidate_scores if r["status"] == "scored"]
    both = [(value(r, "failure_frequency"), value(r, "uniform")) for r in rows]
    both = [(a, b) for a, b in both if a is not None]
    groups: dict[str, list[dict]] = {}
    for row in candidate_scores:
        if row["status"] == "scored" and value(row, "uniform") is not None:
            groups.setdefault(row["history_id"], []).append(row)
    groups = {h: g for h, g in groups.items() if len(g) > 1}
    same_order = top_differs = pairs = flipped = 0
    agreement = {name: [0, 0] for name in strategies}  # [pairs ordered as the K = 9 delta orders them, pairs with different delta]
    for members in groups.values():
        by = {name: [value(r, name) for r in members] for name in strategies}
        same_order += order(by["failure_frequency"]) == order(by["uniform"])
        top_differs += order(by["failure_frequency"])[0] != order(by["uniform"])[0]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = by["failure_frequency"][i] - by["failure_frequency"][j], by["uniform"][i] - by["uniform"][j]
                pairs += 1
                flipped += (a > 1e-9 and b < -1e-9) or (a < -1e-9 and b > 1e-9)
                gap = members[i]["delta"] - members[j]["delta"]
                if gap:
                    for name in strategies:
                        difference = by[name][i] - by[name][j]
                        agreement[name][1] += 1
                        agreement[name][0] += 1.0 if difference * gap > 0 else 0.5 if abs(difference) < 1e-9 else 0.0
    return {
        "aggregation": method, "temperature": tau, "responses_with_R_A": len(both),
        "R_A_failure_frequency": quantiles([a for a, _ in both]), "R_A_uniform": quantiles([b for _, b in both]),
        "mean_abs_difference": mean(abs(a - b) for a, b in both) if both else None,
        "responses_where_R_A_differs": sum(abs(a - b) > 1e-9 for a, b in both),
        "weights": {name: weights for name, weights in strategies.items()},
        "candidate_groups": len(groups), "groups_with_the_same_order": same_order, "groups_with_a_different_best_candidate": top_differs,
        "candidate_pairs": pairs, "pairs_ordered_oppositely": flipped,
        "diagnostic_agreement_with_the_K9_delta_order": {name: {"pairs_with_different_delta": int(n),
                                                              "share_ordered_like_the_delta": c / n if n else None}
                                                         for name, (c, n) in agreement.items()},
        "note": "Controlled candidate diagnostic: the candidate responses to one D1 are five deliberately different behaviors, so "
                "they differ more than the samples of a policy in GRPO do; a change of order here is an upper indication, not a "
                "prediction of what GRPO would see. What matters for GRPO is the order within a group, not the mean R_A. The weight "
                "source (uniform, failure_frequency or a gap-based weight) is decided by a person after this pilot.",
    }


def rubric_table(rubric_set: dict, selection: dict, trajectories: list[dict], scores: list[dict], candidate_scores: list[dict]) -> list[dict]:
    """Per active rubric: what a weight could be based on, side by side."""
    stage_of = {(t["trajectory_id"], i): turn["stage"] for t in trajectories for i, turn in enumerate(t["turns"])}
    correct = {t["trajectory_id"]: float(t["is_correct"]) for t in trajectories}
    categories = {c["category_id"]: c for c in selection.get("categories", [])}
    table = []
    for rubric in rubric_set["rubrics"]:
        rid, category = rubric["rubric_id"], categories.get(rubric["category_id"], {})
        natural = [(stage_of[(r["trajectory_id"], r["turn_id"])], r["labels"].get(rid), correct[r["trajectory_id"]])
                   for r in scores if r["status"] == "scored"]
        candidates = [(r["labels"].get(rid), r["after_rate"], r["delta"]) for r in candidate_scores if r["status"] == "scored"]

        def rate(stage: str):
            rows = [label for s, label, _ in natural if s == stage]
            return sum(label != "NA" for label in rows) / len(rows) if rows else None

        def gap(values_by_score: dict):
            return mean(values_by_score[2]) - mean(values_by_score[0]) if values_by_score[2] and values_by_score[0] else None

        success = {score: [after for label, after, _ in candidates if label == score] for score in (0, 1, 2)}
        delta = {score: [d for label, _, d in candidates if label == score] for score in (0, 1, 2)}
        natural_success = {score: [ok for _, label, ok in natural if label == score] for score in (0, 1, 2)}
        applicable_candidates = sum(label != "NA" for label, _, _ in candidates)
        table.append({
            "rubric_id": rid, "category_id": rubric["category_id"], "name": rubric["name"],
            "current_failure_frequency": rubric["failure_count"], "historical_frequency": category.get("historical_frequency"),
            "validation_success_rate": category.get("validation_success_rate"),
            "weight_failure_frequency": rubric.get("weight"), "weight_uniform": 1 / len(rubric_set["rubrics"]),
            "applicability_A1": rate("A1"), "applicability_A2": rate("A2"),
            "applicability_on_A1_candidates": applicable_candidates / len(candidates) if candidates else None,
            "no_match_rate": 1 - sum(label != "NA" for _, label, _ in natural) / len(natural) if natural else None,
            # the gaps are score 2 minus score 0 among the candidates the rubric APPLIES to; the sample sizes belong to them
            "controlled_candidates_scored": len(candidates), "applicable_n": applicable_candidates,
            "score_2_n": len(success[2]), "score_1_n": len(success[1]), "score_0_n": len(success[0]),
            "task_success_gap": gap(success), "effectiveness_gap": gap(delta),
            "task_success_gap_on_natural_trajectories": gap(natural_success),
            "natural_score_2_n": len(natural_success[2]), "natural_score_0_n": len(natural_success[0])})
    return table


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(ordered), "min": ordered[0], "median": median(ordered), "mean": mean(ordered), "max": ordered[-1],
            "share_zero": sum(v == 0 for v in ordered) / len(ordered), "share_one": sum(v == 1 for v in ordered) / len(ordered)}


def build_report(config: dict, out: Path, trajectories: list[dict], scores: list[dict],
                 candidate_scores: list[dict] | None = None) -> dict:
    candidate_scores = candidate_scores or []
    rubrics_dir = out / "rubrics"
    summary = json.loads((rubrics_dir / "summary.json").read_text(encoding="utf-8"))
    rubric_set = json.loads((rubrics_dir / "rubric_set.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((rubrics_dir / "category_diagnostics.json").read_text(encoding="utf-8"))
    analyses = read_jsonl(rubrics_dir / "failure_analyses.jsonl", missing_ok=True)
    selection_of_failures = json.loads((rubrics_dir / "failure_selection.json").read_text(encoding="utf-8")) \
        if (rubrics_dir / "failure_selection.json").is_file() else {}
    trajectories_by_id = {str(t["trajectory_id"]): t for t in read_jsonl(out / "base_trajectories.jsonl", missing_ok=True)} or \
        {str(t["trajectory_id"]): t for t in trajectories}
    selection_path = rubrics_dir / "active_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
    aggregation = (config.get("rubric_aggregation") or {}).get("method", "weighted_mean"), (config.get("rubric_aggregation") or {}).get("temperature")
    weights = weight_strategies(rubric_set)[(config.get("rubric_importance") or {}).get("strategy", "failure_frequency")] or None
    stage_of = {(t["trajectory_id"], i): turn["stage"] for t in trajectories for i, turn in enumerate(t["turns"])}
    by_stage: dict[str, list[dict]] = {"A1": [], "A2": []}
    per_trajectory: dict[str, list[float]] = {}
    for row in scores:
        if row["status"] != "scored":
            continue
        detail = appropriateness_detail(row["labels"], weights, *aggregation)
        by_stage[stage_of[(row["trajectory_id"], row["turn_id"])]].append(detail)
        if detail["appropriateness"] is not None:
            per_trajectory.setdefault(row["trajectory_id"], []).append(detail["appropriateness"])

    def applicability(details: list[dict]) -> dict:
        units = sum(len(d["applicable_rubrics"]) + len(d["no_match_rubrics"]) for d in details)
        applicable = sum(len(d["applicable_rubrics"]) for d in details)
        return {"scored_turns": len(details),
                "turns_with_an_applicable_rubric": sum(not d["no_applicable_rubric"] for d in details) / len(details) if details else None,
                "applicable_share_of_turn_rubric_units": applicable / units if units else None,
                "no_match_rate": 1 - applicable / units if units else None,
                "turn_appropriateness": quantiles([d["appropriateness"] for d in details if d["appropriateness"] is not None])}

    ledger = out / "api_usage.jsonl"
    prices = json.loads((ROOT / "configs/pricing.json").read_text(encoding="utf-8"))["prices"]
    tau = config["reward"]["softmin_temperature"]
    report = {
        "1_failed_trajectories": summary["selection"],
        "1b_failure_sample": {k: selection_of_failures.get(k) for k in ("rule", "analyzed", "analyzed_by_collaboration_state",
                                                                         "failed_by_collaboration_state", "analyzed_trajectory_ids")},
        "2_rubric_candidates": summary["analysis"],
        # where the analyzer placed the failure (the Agent turn its candidate rubric is about)
        "2b_failure_points": dict(Counter(point for a in analyses if a["analysis_status"] == "valid"
                                          for point in failure_points(a, trajectories_by_id))),
        "2_invalid_candidate_reasons": dict(Counter(p for a in analyses if a["analysis_status"] == "invalid"
                                                    for p in a["validation_problems"]).most_common(10)),
        "3_validation": summary["validation"],
        "4_consolidated_categories": summary["categories"],
        "5_category_failure_frequencies": summary["frequencies"],
        "6_rubric_weights": [{"rubric_id": r["rubric_id"], "category_id": r["category_id"], "name": r["name"],
                              "failure_count": r["failure_count"], "weight": r["weight"]} for r in rubric_set["rubrics"]],
        "7_example_rubrics": [{k: r[k] for k in ("rubric_id", "name", "applicability", "scores")} for r in rubric_set["rubrics"][:3]],
        "8_context_state_consistency": {"active_rubrics": len(rubric_set["rubrics"]),
                                        "rubrics_with_schema_problems": {r["rubric_id"]: validate_rubric(r) for r in rubric_set["rubrics"]
                                                                         if validate_rubric(r)},
                                        "categories_excluded": [{"category_id": e["category_id"], "reason": e["reason"]}
                                                                for e in rubric_set["excluded_categories"]]},
        "9_a1_applicability": applicability(by_stage["A1"]),
        "10_a2_applicability": applicability(by_stage["A2"]),
        "11_no_match_rate_overall": applicability(by_stage["A1"] + by_stage["A2"])["no_match_rate"],
        "12_appropriateness_per_response": {"aggregation": aggregation[0], "rubric_temperature": aggregation[1],
                                            "weight_strategy_in_config": (config.get("rubric_importance") or {}).get("strategy"),
                                            "per_response": quantiles([d["appropriateness"] for d in by_stage["A1"] + by_stage["A2"]
                                                                       if d["appropriateness"] is not None]),
                                            "A1": quantiles([d["appropriateness"] for d in by_stage["A1"] if d["appropriateness"] is not None]),
                                            "A2": quantiles([d["appropriateness"] for d in by_stage["A2"] if d["appropriateness"] is not None]),
                                            "diagnostic_cross_turn_softmin_per_trajectory (not a training reward)":
                                                quantiles([soft_min(v, tau) for v in per_trajectory.values()]),
                                            "scorer_failures": sum(r["status"] != "scored" for r in scores + candidate_scores)},
        "13_duplicate_or_fragmented_categories": {k: diagnostics[k] for k in ("near_duplicate_pairs", "fragmentation")},
        "14_reactivation": "not testable in one iteration (reactivation needs a category that disappears and returns); "
                           "covered by tests/test_method_config.py::test_reactivation_across_three_iterations",
        "15_api_cost": usage.summarize([ledger], prices) if ledger.is_file() else None,
        "16_categories_and_the_cap": {
            "max_active_categories": selection.get("max_active_categories"),
            "registry_categories": selection.get("all_registry_categories"),
            "eligible_categories": len(selection.get("eligible_categories", [])),
            "active_categories": selection.get("active_category_count"),
            "fewer_than_max_reason": selection.get("fewer_than_max_reason"),
            "failure_frequency_by_rank": [{"rank": c["selection_rank"], "category_id": c["category_id"], "name": c["name"],
                                           "current_frequency": c["current_frequency"], "selected_active": c["selected_active"],
                                           "reason_for_inactive": c["reason_for_inactive"]}
                                          for c in sorted((c for c in selection.get("categories", []) if c["selection_rank"]),
                                                          key=lambda c: c["selection_rank"])],
            "note": "The cap is a starting value; whether it fits is judged from this distribution, not decided here."},
        "17_rubric_weight_diagnostics (gaps on the controlled A1 candidates; read them with their n)": rubric_table(rubric_set, selection, trajectories, scores, candidate_scores),
        "18_uniform_vs_failure_frequency (controlled candidate diagnostic)": weighting_comparison(config, rubric_set, scores, candidate_scores) if rubric_set["rubrics"] else None,
        "19_models": {"agent": config["policy"]["model_id"], "doctor": config["resolved_environment"]["models"]["doctor"]["model_id"],
                      "outcome_judge": config["resolved_environment"]["models"]["judge"]["model_id"],
                      "failure_analyzer": config["roles"]["failure_analyzer"]["model_id"],
                      "rubric_manager": config["roles"]["rubric_manager"]["model_id"],
                      "rubric_scorer": config["rubrics"]["scorer"]["model_id"],
                      "note": "One model family in five roles is accepted for the pilot only; evaluator independence is decided before the main experiment."}
        if "resolved_environment" in config else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Rubric pilot report", ""]
    for key, value in report.items():
        lines += [f"## {key.replace('_', ' ')}", "", "```json", json.dumps(value, indent=2, ensure_ascii=False), "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run(config: dict, out: Path, max_failures: int, score_count: int, candidate_histories: int = 30) -> dict:
    missing = missing_settings(config)
    if missing:
        raise IntegrationUnavailable("the rubric pilot needs: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(usage.ENV, str(out / "api_usage.jsonl"))
    os.environ.setdefault(usage.STAGE_ENV, "rubric_pilot")
    base = trajectories_path(config, ROOT, 0)
    if not base.is_file():
        base_trajectories(config, ROOT)
    shutil.copyfile(base, out / "base_trajectories.jsonl")
    rerun_fn, intervene_fn = build_rubrics.clinic_rerun(config, 0, out / "rubrics")[:2] if config["validation"].get("enabled") else (None, None)
    build_rubrics.run(config, base, out / "rubrics", 0, max_failures=max_failures, failure_sample=config["rubric_pilot"]["failure_sample"],
                      rerun_fn=rerun_fn, intervene_fn=intervene_fn,
                      observe_fn=build_rubrics.gap_observer(config, 0, base, out / "rubrics") if build_rubrics.gap_selection(config) else None)
    trajectories = [t for t in read_jsonl(base) if t["trajectory_status"] == "valid"][:score_count]
    rubric_set = json.loads((out / "rubrics/rubric_set.json").read_text(encoding="utf-8"))
    scores = score_trajectories(config, trajectories, rubric_set, out, config["rollout"]["workers"])
    candidate_scores = score_candidates(config, [t for t in read_jsonl(base) if t["trajectory_status"] == "valid"], rubric_set, out,
                                        config["rollout"]["workers"], candidate_histories)
    report = build_report(config, out, trajectories, scores, candidate_scores)
    return {"report": str((out / "report.md").relative_to(ROOT)), "active_rubrics": len(rubric_set["rubrics"]),
            "api_cost_usd": (report["15_api_cost"] or {}).get("total_usd")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("estimate", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-failures", type=int, help="failed TRAIN trajectories sent to rubric generation (default: rubric_pilot.max_failures)")
    parser.add_argument("--score-trajectories", type=int, help="TRAIN trajectories whose A1 and A2 are scored (default: rubric_pilot.score_trajectories)")
    parser.add_argument("--score-candidates", type=int,
                        help="TRAIN histories whose A1 label candidates are scored (for the gap and weighting diagnostics)")
    parser.add_argument("--run-id", help="run: continue this pilot run instead of creating a new one")
    args = parser.parse_args()
    config = load_main_config(args.config, ROOT)
    pilot = config["rubric_pilot"]  # the sample sizes and the failure sampling rule, fixed before the pilot
    args.max_failures = args.max_failures or pilot["max_failures"]
    args.score_trajectories = args.score_trajectories or pilot["score_trajectories"]
    args.score_candidates = args.score_candidates or pilot["score_candidate_histories"]
    if args.command == "estimate":
        result = estimate(dict(config, output_dir=f"{config['output_root']}/_none"), args.max_failures, args.score_trajectories)
    else:
        config = open_run(config, ROOT, run_id=args.run_id, resume=bool(args.run_id), label="rubric_pilot",
                          purpose="small real-model rubric pilot on TRAIN failures")
        result = dict(run(config, ROOT / config["output_dir"], args.max_failures, args.score_trajectories, args.score_candidates),
                      run_id=config["run_id"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
