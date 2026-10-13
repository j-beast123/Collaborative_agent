"""Diagnostic comparison (not the method, not read by any run): original vs decision-focused rubric generation on the same
Qwen3-14B-native TRAIN30 failures, both validated with the frozen-intervention causal validation (3 reruns, keep at 2).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.diagnostics.decision_rubric_pilot import EXPECTED, decide
from scripts.diagnostics.near_miss_diagnostic import PROMPT as NEAR_PROMPT
from scripts.diagnostics.targeted_rubric_pilot import check, rubric_text
from src.agentclinic.adapter import agent_view, render
from src.agentclinic.pipeline import QUESTION
from src.agentclinic.runtime import build_judge
from src.agentclinic.selection import split_eligible_cases
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.failure_analysis import load_latest
from src.judges.jev import JevClient
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object
from src.judges.rubric_validation import GENERATION_FAILURE, load_validations

ROOT = Path(__file__).resolve().parents[2]
INVALID = "rubric_generation_invalid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((ROOT / args.original / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    source = read_config(ROOT / config["environment"]["config"])
    cases = split_eligible_cases(source, ROOT, allowed_splits=("train",))
    judge = build_judge(config, ROOT, source)
    equivalent = lambda gold, text: bool(text) and bool(judge.equivalent(gold, text)["equivalent"])  # noqa: E731
    reader = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader.jsonl")
    from src.effectiveness.service import build_effectiveness
    from src.grpo.trainer import build_scorer, effectiveness_settings

    provider = build_effectiveness(effectiveness_settings(config, 0), 1.0, ROOT)
    scorer = build_scorer(config, ROOT)
    settings = config["rubrics"]["scorer"]
    client = JevClient(endpoint=settings.get("endpoint"), model=settings.get("model_id"), api_key_env=settings.get("api_key_env"))
    controls = [{"control": name, "as_expected": all(decide(client, text)[k]["choice"] == v for k, v in expected.items())}
                for name, (gold, text, expected) in EXPECTED.items()]
    if not all(c["as_expected"] for c in controls):
        raise SystemExit(f"the checker misjudges its controls: {controls}")

    def near(gold: str, predicted: str) -> dict:
        if not predicted:
            return {"label": "clearly_wrong"}
        answer = reader.complete(ChatRequest(NEAR_PROMPT, json.dumps({"gold_diagnosis": gold, "predicted_final_diagnosis": predicted}, ensure_ascii=False),
                                             "near-miss-v1", {"temperature": 0, "max_tokens": 300}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        return parsed or {"label": None}

    def side(run: Path) -> dict:
        trajectories = {t["trajectory_id"]: t for t in read_jsonl(run / "base/trajectories.jsonl")}
        analyses = {r["source_case_id"]: r for r in load_latest(run / "iter0/rubrics/failure_analyses.jsonl").values()}
        validations = load_validations(run / "iter0/rubrics/validations.jsonl")
        rubric_set = json.loads((run / "iter0/rubrics/rubric_set.json").read_text(encoding="utf-8"))["rubrics"]
        result = {}
        for case_id, record in sorted(analyses.items()):
            t = trajectories[record["source_trajectory_id"]]
            gold = t["ground_truth"]["correct_diagnosis"]
            row = {"case_id": case_id, "gold": gold, "generation": record["analysis_status"], "problems": record["validation_problems"]}
            if record["analysis_status"] != "valid":
                result[case_id] = dict(row, valid=False, status=INVALID)
                continue
            checked = check(record, t, client, scorer)
            extra = decide(client, rubric_text(record))
            checks = {"A": checked["A_interaction_specific"], "B": checked["B_no_diagnostic_knowledge"], "C": checked["C_applicable_in_failed_state"],
                      "D": checked["D_reusable"], "E": extra["E"]["choice"] == "compares", "F": extra["F"]["choice"] == "decides"}
            v = validations.get(record["rubric_id"], {})
            intervention = v.get("intervention") or {}
            turn = record["analysis"]["intervention_turn_id"]
            reruns = []
            for index, x in enumerate(v.get("reruns", [])):
                dialogue = x["dialogue"]
                deltas = []
                for i, m in enumerate(dialogue):
                    if m["speaker"] != "agent" or i < turn or not m.get("text"):
                        continue
                    history = {"history_id": f"{record['rubric_id']}:{index}:{i}", "intervention_turn": m["stage"], "question": QUESTION,
                               "options": {}, "scenario": render(agent_view(cases[case_id])),
                               "turns": [{"speaker": y["speaker"], "text": y["text"]} for y in dialogue[:i]]}
                    deltas.append((i == turn, provider.predict(history, m["text"], m["stage"], trajectory_id=history["history_id"]).delta))
                reading = near(gold, x.get("final_diagnosis") or "") if not x.get("correct") else {"label": "strict_correct"}
                reruns.append({"final_diagnosis": x.get("final_diagnosis"), "strict_correct": bool(x.get("correct")),
                               "secondary": "STRICT_CORRECT" if x.get("correct") else
                               "NEAR_MISS" if reading.get("label") in ("near_miss", "equivalent") else
                               "CLEARLY_WRONG" if reading.get("label") == "clearly_wrong" else "UNREAD",
                               "reader_relation": reading.get("relation"), "reader_reason": reading.get("reason"),
                               "agent_reached_gold": any(equivalent(gold, m.get("assessment") or "") for m in dialogue[turn:] if m["speaker"] == "agent"),
                               "doctor_reached_gold": any(equivalent(gold, m.get("provisional_diagnosis") or m.get("final_diagnosis") or "")
                                                          for m in dialogue[turn + 1:] if m["speaker"] == "doctor"),
                               "R_E_failed_turn": next((d for first, d in deltas if first), None),
                               "R_E_later_turns": (sum(d for first, d in deltas if not first) / max(1, sum(not first for first, _ in deltas)))
                               if any(not first for first, _ in deltas) else None})
            valid = all(checks.values())
            result[case_id] = dict(row, rubric_id=record["rubric_id"], rubric=record["candidate_rubric"]["name"], rubric_text=rubric_text(record),
                                   failed_turn=t["turns"][turn]["stage"], checks=checks, valid=valid,
                                   check_probabilities={**checked["probabilities"], **{k: extra[k] for k in ("E", "F")}},
                                   attempt_scores=[a.get("label") for a in intervention.get("attempts", [])],
                                   intervention_feasible=intervention.get("status") == "frozen", intervention_text=intervention.get("text"),
                                   pipeline_status=v.get("validation_status"), successes=v.get("successes"), reruns=reruns,
                                   in_rubric_set=any(record["rubric_id"] in r.get("source_rubric_candidates", []) for r in rubric_set))
        return result

    original, decision = side(ROOT / args.original), side(ROOT / args.decision)

    def primary(row: dict, gated: bool) -> str:
        if row.get("status") == INVALID or (gated and not row["valid"]):
            return INVALID
        return row["pipeline_status"]

    def tally(rows: dict, gated: bool) -> dict:
        status = {k: primary(r, gated) for k, r in rows.items()}
        generated = len(rows)
        valid = sum(s != INVALID for s in status.values())
        feasible = sum(s in ("pass", "fail") for s in status.values())
        passed = sum(s == "pass" for s in status.values())
        rate = lambda n, d: round(n / d, 3) if d else None  # noqa: E731
        reruns = [x for k, r in rows.items() if status[k] in ("pass", "fail") for x in r["reruns"]]
        mean = lambda v: round(sum(v) / len(v), 4) if v else None  # noqa: E731
        return {"generated": generated, "valid": valid, "intervention_feasible": feasible,
                "intervention_generation_failure": sum(s == GENERATION_FAILURE for s in status.values()),
                "causal_pass": passed, "causal_fail": sum(s == "fail" for s in status.values()),
                "validity_rate": rate(valid, generated), "feasibility_rate": rate(feasible, generated),
                "conditional_causal_success_rate": rate(passed, feasible), "overall_causal_pass_rate": rate(passed, generated),
                "secondary_over_feasible_reruns": {"reruns": len(reruns), **Counter(x["secondary"] for x in reruns),
                                                   "agent_reached_gold": sum(x["agent_reached_gold"] for x in reruns),
                                                   "doctor_reached_gold": sum(x["doctor_reached_gold"] for x in reruns),
                                                   "R_E_failed_turn": mean([x["R_E_failed_turn"] for x in reruns if x["R_E_failed_turn"] is not None]),
                                                   "R_E_later_turns": mean([x["R_E_later_turns"] for x in reruns if x["R_E_later_turns"] is not None])}}

    summary = {"failures": sorted(set(original) | set(decision)),
               "original": dict(tally(original, gated=False), note="the pipeline's own validation; A-F are reported, not applied"),
               "decision_focused": dict(tally(decision, gated=True), note="A-F applied: a rubric failing any is rubric_generation_invalid"),
               "decision_focused_without_gate": tally(decision, gated=False),
               "checker_controls": controls,
               "note": "11 failures; primary = strict majority-v1, pass at >= 2 of 3; near-miss and R_E are secondary only"}
    cases = [{"case_id": k, "original": original.get(k), "decision_focused": decision.get(k),
              "original_primary": primary(original[k], False) if k in original else None,
              "decision_primary": primary(decision[k], True) if k in decision else None} for k in summary["failures"]]
    (out / "native_rubric_comparison.json").write_text(json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "native_rubric_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case_id", "failed_turn_original", "failed_turn_decision", "rubric_original", "rubric_decision", "checks_original", "checks_decision",
                  "feasible_original", "feasible_decision", "outcomes_original", "outcomes_decision", "primary_original", "primary_decision",
                  "final_diagnoses_original", "final_diagnoses_decision", "score2_original", "score2_decision"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            o, d = c["original"] or {}, c["decision_focused"] or {}
            writer.writerow({"case_id": c["case_id"], "failed_turn_original": o.get("failed_turn"), "failed_turn_decision": d.get("failed_turn"),
                             "rubric_original": json.dumps(o.get("rubric_text"), ensure_ascii=False), "rubric_decision": json.dumps(d.get("rubric_text"), ensure_ascii=False),
                             "checks_original": json.dumps(o.get("checks")), "checks_decision": json.dumps(d.get("checks")),
                             "feasible_original": o.get("intervention_feasible"), "feasible_decision": d.get("intervention_feasible"),
                             "outcomes_original": json.dumps([x["strict_correct"] for x in o.get("reruns", [])]),
                             "outcomes_decision": json.dumps([x["strict_correct"] for x in d.get("reruns", [])]),
                             "primary_original": c["original_primary"], "primary_decision": c["decision_primary"],
                             "final_diagnoses_original": json.dumps([x["final_diagnosis"] for x in o.get("reruns", [])], ensure_ascii=False),
                             "final_diagnoses_decision": json.dumps([x["final_diagnosis"] for x in d.get("reruns", [])], ensure_ascii=False),
                             "score2_original": o.get("intervention_text"), "score2_decision": d.get("intervention_text")})
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for c in cases:
        o, d = c["original"] or {}, c["decision_focused"] or {}
        print(c["case_id"][-4:], "| O", o.get("failed_turn"), o.get("checks") and "".join("Y" if v else "N" for v in o["checks"].values()),
              o.get("attempt_scores"), c["original_primary"], [x["secondary"][:6] for x in o.get("reruns", [])],
              "| D", d.get("failed_turn"), d.get("checks") and "".join("Y" if v else "N" for v in d["checks"].values()),
              d.get("attempt_scores"), c["decision_primary"], [x["secondary"][:6] for x in d.get("reruns", [])])


if __name__ == "__main__":
    main()
