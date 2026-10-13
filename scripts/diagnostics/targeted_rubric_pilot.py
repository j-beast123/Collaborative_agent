"""Diagnostic pilot (not the method, not read by any run): for the failures an earlier audit read as recoverable by
interaction, generate the rubric candidate again under an interaction-specific instruction and put it through the
frozen-intervention causal validation as it is.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from scripts import build_rubrics
from scripts.diagnostics.failure_audit import names
from src.common.io import read_jsonl
from src.judges.agent_state import resolve_include_initial_turn
from src.judges.failure_analysis import analyze_failures, load_latest
from src.judges.jev import JevClient
from src.judges.llm import CachedChat, build_chat_backend
from src.judges.rubric_scoring import build_scoring_items, rubric_view
from src.judges.rubric_validation import GENERATION_FAILURE, load_validations, validate_candidates

ROOT = Path(__file__).resolve().parents[2]
ADDENDUM = "prompts/agentclinic/rubric_interaction_targeted_addendum.txt"
VERSION = "+interaction-targeted-v1"
INVALID = "rubric_generation_invalid"
# Checker v2.
CHECKS = "candidate_checks_v2.jsonl"
QUESTIONS = {
    "A": ("What does the rubric evaluate?",
          {"interaction": "How the consulting Agent handles the collaborator's claim and the available evidence in its reply: a "
                          "collaboration or communication behavior.",
           "diagnostic": "Whether the Agent reaches or argues for a particular diagnosis, or applies particular medical reasoning."}),
    "B": ("Does the rubric text name a disease or state a disease-specific medical finding, fact or diagnostic tip?",
          {"none": "No. The text names no disease and states no disease-specific finding, fact or diagnostic tip.",
           "contains": "Yes. The text names a disease or states a disease-specific finding, fact or diagnostic tip."}),
    "D": ("Could the rubric be applied as written in a different clinical case about a different disease?",
          {"reusable": "Yes. Its situation and behavior are described without the particulars of one case.",
           "case_bound": "No. It depends on the particulars of one case or one disease."}),
}
PASSING = {"A": "interaction", "B": "none", "D": "reusable"}
_CLEAN = {"name": "Separating missing results from negative results",
          "applicability": {"user_side_context": "The collaborator asks for a result as the way to settle a hypothesis.",
                            "agent_side_state": "The agent does not have that result."},
          "score_0": "The agent restates its view and recommends obtaining the result.",
          "score_1": "The agent says the result is missing but does not separate that from evidence against the hypothesis.",
          "score_2": "The agent states which requested results are not available, separates that from evidence against the "
                     "hypothesis, and says what can be concluded now.",
          "ideal_response_behavior": "State what is missing, separate it from negative evidence, and say what can be concluded now."}
CONTROLS = [
    ("clean_missing_vs_negative", "Boerhaave syndrome", _CLEAN, True),
    ("clean_surface_conflict", "Chronic lymphocytic leukemia", dict(
        _CLEAN, name="Surfacing conflicting evidence before agreeing",
        score_2="The agent states the evidence it holds that conflicts with the collaborator's provisional conclusion and asks how "
                "the collaborator accounts for it before agreeing.",
        ideal_response_behavior="Name the conflicting evidence you hold and ask the collaborator to account for it."), True),
    ("names_disease", "Boerhaave syndrome", dict(
        _CLEAN, name="Recognizing Boerhaave syndrome",
        score_2="The agent points out that vomiting followed by chest pain and subcutaneous emphysema indicates Boerhaave syndrome "
                "and recommends a contrast esophagram.",
        ideal_response_behavior="Tell the collaborator to consider Boerhaave syndrome."), False),
    ("disease_specific_tip", "Chronic lymphocytic leukemia", dict(
        _CLEAN, name="Checking lymphocytosis",
        score_2="The agent checks the lymphocyte count and smudge cells, because an older patient with lymphadenopathy and "
                "lymphocytosis has a lymphoproliferative disorder.",
        ideal_response_behavior="In suspected CLL, confirm lymphocytosis on the blood count."), False),
]


def decide(client: JevClient, text: dict, gold: str) -> dict:
    # the checker sees the rubric only: with correct_diagnosis in its input it answered "contains" for rubrics without any
    # medical content. That the correct diagnosis is not named in the rubric is checked on the text itself (names()).
    answers, _ = client.decide(json.dumps({"rubric": text}, ensure_ascii=False),
                               {k: {"type": "choice", "instructions": q, "criteria": c} for k, (q, c) in QUESTIONS.items()},
                               purpose="targeted_rubric_check")
    return {k: {"choice": answers[k]["value"], "probabilities": answers[k]["probabilities"]} for k in QUESTIONS}


def controls(client: JevClient, path: Path) -> None:
    """The checker is used only if it separates the control rubrics; otherwise the pilot stops before any candidate is checked."""
    rows = []
    for name, gold, text, clean in CONTROLS:
        answer = decide(client, text, gold)
        passed = all(answer[k]["choice"] == PASSING[k] for k in QUESTIONS)
        rows.append({"control": name, "expected_to_pass": clean, "passed": passed, "answers": answer})
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    wrong = [r["control"] for r in rows if r["passed"] != r["expected_to_pass"]]
    if wrong:
        raise SystemExit(f"the checker misjudges the control rubrics {wrong}: see {path}")


def rubric_text(record: dict) -> dict:
    rubric = record["candidate_rubric"]
    return {"name": rubric["name"], "applicability": rubric["applicability"],
            "score_0": rubric["scores"]["0"]["behavior"], "score_1": rubric["scores"]["1"]["behavior"],
            "score_2": rubric["scores"]["2"]["behavior"], "ideal_response_behavior": record["analysis"]["ideal_response_behavior"]}


def check(record: dict, trajectory: dict, client: JevClient, scorer) -> dict:
    """A-D for one candidate."""
    gold = trajectory["ground_truth"]["correct_diagnosis"]
    text = rubric_text(record)
    p = decide(client, text, gold)
    named = names(gold, json.dumps(text, ensure_ascii=False))
    turn = record["analysis"]["intervention_turn_id"]
    item, = [i for i in build_scoring_items(trajectory, [rubric_view(record)], include_initial_turn=True) if i.turn_id == turn]
    scored = scorer.score(item)
    label = scored.labels.get(record["rubric_id"])
    result = {"rubric_id": record["rubric_id"], "case_id": record["source_case_id"], "failed_turn": trajectory["turns"][turn]["stage"],
              "A_interaction_specific": p["A"]["choice"] == PASSING["A"],
              "B_no_diagnostic_knowledge": p["B"]["choice"] == PASSING["B"] and not named,
              "C_applicable_in_failed_state": scored.status == "scored" and label != "NA",
              "D_reusable": p["D"]["choice"] == PASSING["D"],
              "probabilities": p, "correct_diagnosis_named_in_rubric": named, "original_turn_score": label,
              "original_turn_score_probabilities": scored.probabilities.get(record["rubric_id"]),
              "checker": {**client.identity(), "version": "v2-choice", "applicability": scorer.identity()}}
    result["passed"] = all(result[k] for k in ("A_interaction_specific", "B_no_diagnostic_knowledge", "C_applicable_in_failed_state", "D_reusable"))
    return result


def checks(path: Path, records: list[dict], trajectories: dict, client, scorer) -> dict[str, dict]:
    done = {r["rubric_id"]: r for r in read_jsonl(path, missing_ok=True)}
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record["rubric_id"] not in done:
                done[record["rubric_id"]] = check(record, trajectories[record["source_trajectory_id"]], client, scorer)
                handle.write(json.dumps(done[record["rubric_id"]], ensure_ascii=False) + "\n")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--failure-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("run", "report"), default="run")
    parser.add_argument("--old-local-effect", help="report: local_effect.json of the earlier candidates")
    args = parser.parse_args()
    source, out = ROOT / args.source_run, ROOT / args.output_dir
    rubrics = out / "iter0/rubrics"
    audit = {f["case_id"]: f for f in json.loads((ROOT / args.failure_audit).read_text(encoding="utf-8"))["failures"]}
    targets = sorted(c for c, f in audit.items() if f["audit_interaction_recoverable"] == "yes")
    if args.stage == "report":
        return report(source, out, audit, targets, args)
    rubrics.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((source / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    if not (out / "resolved_config.json").exists():
        (out / "resolved_config.json").write_text(json.dumps(dict(
            config, diagnostic_pilot={"what": "interaction-targeted rubric generation on audit-recoverable failures; not a run of the method",
                                      "source_run": args.source_run, "failure_audit": args.failure_audit, "targets": targets,
                                      "generator_addendum": ADDENDUM}), ensure_ascii=False, indent=1), encoding="utf-8")
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(source / "base/trajectories.jsonl") if t["case_id"] in targets}
    assert sorted(t["case_id"] for t in trajectories.values()) == targets and not any(t["is_correct"] for t in trajectories.values())
    (out / "base").mkdir(exist_ok=True)
    if not (out / "base/trajectories.jsonl").exists():
        (out / "base/trajectories.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trajectories.values()), encoding="utf-8")
    old = [a for a in read_jsonl(source / "iter0/rubrics/failure_analyses.jsonl") if a["source_case_id"] in targets]
    (out / "old_candidates.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in old), encoding="utf-8")  # comparison only

    # 1. generation: the run's analyzer and its prompt, plus the interaction-targeted addendum; its own cache (nothing reused)
    analyzer = config["roles"]["failure_analyzer"]
    prompt = (ROOT / analyzer["prompt"]).read_text(encoding="utf-8")
    if analyzer.get("prompt_addendum"):
        prompt += (ROOT / analyzer["prompt_addendum"]).read_text(encoding="utf-8")
    prompt += (ROOT / ADDENDUM).read_text(encoding="utf-8")
    counts = analyze_failures(list(trajectories.values()), CachedChat(build_chat_backend(analyzer), out / "cache_failure_analyzer.jsonl"),
                              system_prompt=prompt, prompt_version=analyzer["prompt_version"] + VERSION, params=analyzer["params"],
                              output_path=rubrics / "failure_analyses.jsonl", include_initial_turn=resolve_include_initial_turn(config))
    print("generation", counts)
    generated = list(load_latest(rubrics / "failure_analyses.jsonl").values())
    valid = [r for r in generated if r["analysis_status"] == "valid"]

    # 2. sanity check of the new candidates and, for the comparison, of the earlier ones
    from src.grpo.trainer import build_scorer

    scorer = build_scorer(config, ROOT)
    settings = config["rubrics"]["scorer"]
    client = JevClient(endpoint=settings.get("endpoint"), model=settings.get("model_id"), api_key_env=settings.get("api_key_env"))
    controls(client, out / "checker_controls_v2.json")
    new_checks = checks(out / CHECKS, valid, trajectories, client, scorer)
    checks(out / f"old_{CHECKS}", [a for a in old if a["analysis_status"] == "valid"], trajectories, client, scorer)
    passing = [r for r in valid if new_checks[r["rubric_id"]]["passed"]]
    print("checks", {r["source_case_id"][-4:]: new_checks[r["rubric_id"]]["passed"] for r in valid})

    # 3. frozen intervention and causal validation, exactly as the source run's rubric step does it
    if passing:
        rerun, intervene = build_rubrics.clinic_rerun(config, 0, rubrics)[:2]
        assert intervene is not None, "the source run has no validation.intervention"
        validation = config["validation"]
        print("validation", validate_candidates(passing, trajectories, rerun, num_reruns=validation["num_reruns"],
                                                min_successes=validation["min_successes"], output_path=rubrics / "validations.jsonl",
                                                intervene=intervene))


def report(source: Path, out: Path, audit: dict, targets: list[str], args) -> None:
    rubrics = out / "iter0/rubrics"
    new = {r["source_case_id"]: r for r in load_latest(rubrics / "failure_analyses.jsonl").values()}
    old = {r["source_case_id"]: r for r in read_jsonl(out / "old_candidates.jsonl")}
    new_checks = {r["case_id"]: r for r in read_jsonl(out / CHECKS, missing_ok=True)}
    old_checks = {r["case_id"]: r for r in read_jsonl(out / f"old_{CHECKS}", missing_ok=True)}
    new_validations = load_validations(rubrics / "validations.jsonl")
    old_validations = load_validations(source / "iter0/rubrics/validations.jsonl")
    local = {}
    for name, path in (("new", out / "local_effect/local_effect.json"), ("old", ROOT / args.old_local_effect if args.old_local_effect else None)):
        if path and path.exists():
            local[name] = {c["case_id"]: c for c in json.loads(path.read_text(encoding="utf-8"))["candidates"]}

    def side(record, check, validation, effect) -> dict:
        valid = bool(record) and record["analysis_status"] == "valid"
        intervention = (validation or {}).get("intervention") or {}
        status = (validation or {}).get("validation_status")
        return {"rubric": record["candidate_rubric"]["name"] if valid else None,
                "rubric_text": rubric_text(record) if valid else None,
                "failed_turn": (check or {}).get("failed_turn"),
                "checks": {k: check[k] for k in ("A_interaction_specific", "B_no_diagnostic_knowledge", "C_applicable_in_failed_state", "D_reusable")} if check else None,
                "check_probabilities": (check or {}).get("probabilities"),
                "valid": bool(check) and check["passed"],
                "attempt_scores": [a["label"] for a in intervention.get("attempts", [])],
                "intervention_feasible": intervention.get("status") == "frozen",
                "intervention_text": intervention.get("text"),
                "rerun_final_diagnoses": [r.get("final_diagnosis") for r in (validation or {}).get("reruns", [])],
                "rerun_outcomes": [bool(r.get("correct")) for r in (validation or {}).get("reruns", [])],
                "causal_result": status if status else None,
                "local_effect": {k: effect.get(k) for k in ("reader_doctor_next_message_closer", "reader_discussion_after_closer",
                                                             "reader_final_diagnosis_closer", "majority_reader_doctor_next_message_closer",
                                                             "majority_reader_discussion_after_closer", "original_delta", "frozen_delta")} if effect else None}

    rows = []
    for case in targets:
        o, n = old.get(case), new.get(case)
        row = {"case_id": case, "audit_failure_reason": audit[case]["audit_failure_reason"], "correct_diagnosis": audit[case]["correct_diagnosis"],
               "old": side(o, old_checks.get(case), old_validations.get(o["rubric_id"]) if o else None, local.get("old", {}).get(case)),
               "new": side(n, new_checks.get(case), new_validations.get(n.get("rubric_id")) if n else None, local.get("new", {}).get(case))}
        if not row["new"]["valid"]:
            row["new"]["causal_result"] = INVALID
        rows.append(row)

    def tally(key: str) -> dict:
        sides = [r[key] for r in rows]
        valid, feasible = sum(s["valid"] for s in sides), sum(s["intervention_feasible"] for s in sides)
        passed = sum(s["causal_result"] == "pass" for s in sides)
        rate = lambda n, d: round(n / d, 3) if d else None  # noqa: E731
        return {"target_failures": len(sides), "valid_interaction_specific_rubrics": valid, "invalid_rubric_generations": len(sides) - valid,
                "intervention_feasible": feasible,
                "intervention_generation_failures": sum(s["causal_result"] == GENERATION_FAILURE for s in sides),
                "causal_pass": passed, "causal_fail": sum(s["causal_result"] == "fail" for s in sides),
                "rubric_validity_rate": rate(valid, len(sides)), "intervention_feasibility_rate": rate(feasible, len(sides)),
                "intervention_feasibility_rate_among_valid": rate(feasible, valid),
                "conditional_causal_success_rate": rate(passed, feasible), "causal_pass_rate_over_targets": rate(passed, len(sides))}

    summary = {"targets": targets, "new_targeted": tally("new"),
               "old_candidates_same_cases": dict(tally("old"), note="the earlier candidates were validated without the A-D check; 'valid' "
                                                                    "here is the same checker applied to them afterwards, for comparison only"),
               "note": "diagnostic pilot; nothing here enters the rubric set of any run. Thresholds, judge, predictor and weights unchanged."}
    (out / "paired_comparison.json").write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{"case_id": r["case_id"], "audit_failure_reason": r["audit_failure_reason"],
                 **{f"{s}_{k}": (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for s in ("old", "new") for k, v in r[s].items()}}
                for r in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
