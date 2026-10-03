"""Per-candidate table of a rubric step's causal validation (validation.intervention: frozen compliant response)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.judges.rubric_validation import GENERATION_FAILURE, load_validations


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rubrics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--before")
    p.add_argument("--failure-audit")
    a = p.parse_args()
    rubrics, out = Path(a.rubrics), Path(a.out)
    analyses = {r["rubric_id"]: r for r in jsonl(rubrics / "failure_analyses.jsonl") if r["analysis_status"] == "valid"}
    validations = load_validations(rubrics / "validations.jsonl")
    before = load_validations(Path(a.before) / "validations.jsonl") if a.before else {}
    audit = {f["case_id"]: f for f in json.loads(Path(a.failure_audit).read_text())["failures"]} if a.failure_audit else {}
    trajectories = {t["trajectory_id"]: t for t in jsonl(rubrics.parent.parent / "base/trajectories.jsonl")}
    table = {r.get("rubric_id") or tuple(r["source_rubric_candidates"])[0]: r
             for r in json.loads((rubrics / "rubric_table.json").read_text())["rows"]} if (rubrics / "rubric_table.json").exists() else {}
    rubric_set = json.loads((rubrics / "rubric_set.json").read_text())["rubrics"] if (rubrics / "rubric_set.json").exists() else []
    in_set = {c: r["rubric_id"] for r in rubric_set for c in r.get("source_rubric_candidates", [])}
    consolidated = json.loads((rubrics / "consolidated_rubrics.json").read_text())["rubrics"] if (rubrics / "consolidated_rubrics.json").exists() else []
    category = {c: r for r in consolidated for c in r.get("source_rubric_candidates", [])}
    rows = []
    for rid, record in analyses.items():
        v, t = validations.get(rid), trajectories.get(record["source_trajectory_id"], {})
        if v is None:
            continue
        intervention = v.get("intervention") or {}
        reruns = v["reruns"]
        category_row = table.get(category[rid]["rubric_id"]) if rid in category else None
        row = {
            "case_id": record["source_case_id"], "rubric_id": rid, "candidate": record["candidate_rubric"]["name"],
            "collaboration_state": t.get("collaboration_state"), "gold": (t.get("ground_truth") or {}).get("correct_diagnosis"),
            "original_final_diagnosis": t.get("final_diagnosis"), "failed_turn": intervention.get("stage"),
            "intervention_status": intervention.get("status"),
            "compliance_passed": (intervention.get("compliance") or {}).get("passed"),
            "accepted_attempt": (intervention.get("compliance") or {}).get("accepted_attempt"),
            "attempt_scores": [x.get("label") for x in intervention.get("attempts", [])],
            "attempt_texts": [x.get("text") for x in intervention.get("attempts", [])],
            "intervention_generation_failure": intervention.get("status") == GENERATION_FAILURE,
            "accepted_score2_probability": next(((x.get("probabilities") or {}).get("2") for x in intervention.get("attempts", [])
                                                 if x.get("label") == 2), None),
            "intervention_text": intervention.get("text"),
            "rerun_final_diagnoses": [x.get("final_diagnosis") for x in reruns],
            "rerun_outcomes": [bool(x.get("correct")) for x in reruns],
            "same_text_in_every_rerun": bool(reruns) and {x.get("intervention_response") for x in reruns} == {intervention.get("text")},
            "successes": v["successes"] if v["validation_status"] != GENERATION_FAILURE else None,
            "validation_status": v["validation_status"],
            "category": category[rid]["category_name"] if rid in category else None,
            "category_removal_reason": (category_row or {}).get("removal_reason"),
            "effectiveness_gap": (category_row or {}).get("effectiveness_gap"),
            "retained_in_rubric_set": rid in in_set, "retained_rubric_id": in_set.get(rid),
        }
        if a.before:
            row["before_successes"] = before.get(rid, {}).get("successes")
            row["before_validation_status"] = before.get(rid, {}).get("validation_status")
        if audit:
            f = audit.get(record["source_case_id"], {})
            row.update({"audit_failure_reason": f.get("audit_failure_reason"), "audit_interaction_recoverable": f.get("audit_interaction_recoverable")})
        rows.append(row)
    status = Counter(r["validation_status"] for r in rows)
    feasible = [r for r in rows if r["intervention_status"] == "frozen"]
    passed = [r for r in rows if r["validation_status"] == "pass"]
    first = [r for r in feasible if r["accepted_attempt"] == 1]  # diagnostic only: had one generation attempt been allowed
    first_pass = [r for r in first if r["validation_status"] == "pass"]
    rate = lambda n, d: round(n / d, 3) if d else None  # noqa: E731
    summary = {"rubrics_dir": str(rubrics), "candidates": len(rows), "validation_status": dict(status),
               "intervention_frozen": sum(r["intervention_status"] == "frozen" for r in rows),
               "frozen_at_first_attempt": sum(r["accepted_attempt"] == 1 for r in rows),
               "pass_among_first_attempt_frozen": sum(r["accepted_attempt"] == 1 and r["validation_status"] == "pass" for r in rows),
               "reruns_with_the_frozen_text": sum(r["same_text_in_every_rerun"] for r in rows),
               "retained_in_rubric_set": sum(r["retained_in_rubric_set"] for r in rows), "rubrics_in_set": len(rubric_set),
               "intervention_feasible": len(feasible),
               "intervention_generation_failure": sum(r["intervention_generation_failure"] for r in rows),
               "causal_pass": len(passed), "causal_fail": sum(r["validation_status"] == "fail" for r in rows),
               "intervention_feasibility_rate": rate(len(feasible), len(rows)),
               "conditional_causal_success_rate": rate(len(passed), len(feasible)),
               "validation_pass_rate_over_all_candidates": rate(len(passed), len(rows)),
               "first_attempt_sensitivity": {
                   "note": "diagnostic only; the selection rule (first score-2 response within 3 attempts) is not changed by it",
                   "feasible": len(first), "causal_pass": len(first_pass),
                   "causal_pass_candidates": [r["case_id"] for r in first_pass],
                   "same_passed_candidates_as_main_rule": {r["rubric_id"] for r in first_pass} == {r["rubric_id"] for r in passed},
                   "retained_rubrics": (len(rubric_set) if {r["rubric_id"] for r in first_pass} == {r["rubric_id"] for r in passed}
                                        else 0 if not first_pass else "differs from the main rule: not recomputed here")},
               "rule": "3 reruns, pass at >= 2 correct final diagnoses (unchanged); intervention_generation_failure is neither pass nor fail"}
    if audit:
        keys = sorted({str(r["audit_interaction_recoverable"]) for r in rows})
        group = lambda key: [r for r in rows if str(r["audit_interaction_recoverable"]) == key]  # noqa: E731
        summary["recoverable_x_intervention_feasible"] = {
            key: {"feasible": sum(r["intervention_status"] == "frozen" for r in group(key)),
                  "generation_failure": sum(r["intervention_generation_failure"] for r in group(key))} for key in keys}
        summary["recoverable_x_causal_pass"] = {
            key: {"pass": sum(r["validation_status"] == "pass" for r in group(key)),
                  "fail": sum(r["validation_status"] == "fail" for r in group(key)),
                  "not_tested": sum(r["intervention_generation_failure"] for r in group(key))} for key in keys}
        summary["failure_reason_x_status"] = {
            key: dict(Counter(r["validation_status"] for r in rows if str(r["audit_failure_reason"]) == key))
            for key in sorted({str(r["audit_failure_reason"]) for r in rows})}
    out.mkdir(parents=True, exist_ok=True)
    (out / "causal_validation_table.json").write_text(json.dumps({"summary": summary, "candidates": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "causal_validation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
