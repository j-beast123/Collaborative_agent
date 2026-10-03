"""Offline eligibility audit (diagnostic; not the method, not read by any run): which failures of a run are failures that a
behavioral rubric could address at all? Nothing is generated or sampled; the stored trajectories and Agent-side states only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agentclinic.adapter import agent_view, render
from src.agentclinic.outcome import normalize
from src.agentclinic.runtime import build_judge
from src.agentclinic.selection import split_eligible_cases
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.failure_analysis import load_latest
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

ROOT = Path(__file__).resolve().parents[2]
VERSION = "eligibility-audit-v1"
CATEGORIES = ("ELIGIBLE_INTERACTION_FAILURE", "KNOWLEDGE_REASONING_LIMITED", "MIXED_BUT_NOT_ELIGIBLE", "UNCERTAIN")
PROMPT = """You audit one failed clinical collaboration for a research analysis. A consulting Agent (who alone has the
examination and test findings) talks with a primary Doctor (who has only the history); the Doctor's final diagnosis was
wrong. You receive the correct diagnosis, everything the Agent knew, the dialogue with turn ids, the Agent-side states
(the Agent's own assessment before each of its turns), the failed Agent turn, and mechanical evidence extracted beforehand
(which is correct; do not contradict it).

Use only what is written in the dialogue, the states and the Agent's information. Do not credit the Agent with a hypothesis,
an inference or a knowledge it did not state. "Before the failed turn" means the turns with a smaller turn id and the state
elicited right before the failed turn.

Answer:
- a2_support_maintained: only if the standalone diagnosis is correct: does the Agent, before the failed turn, still state
  evidence that supports that diagnosis? "yes", "no", or "not_applicable" (standalone not correct)
- a3_gold_supporting_evidence: does the Agent's own information or its turns before the failed turn contain evidence that
  clearly supports the correct diagnosis, whether or not the Agent named it? "yes" / "no"
- agent_hypothesis_before_failed_turn: the Agent's working diagnosis right before the failed turn (as stated)
- doctor_hypothesis_at_failed_turn: the Doctor's diagnosis in the message the failed turn answers
- agent_behavior_at_failed_turn: one sentence on what the Agent did in the failed turn (agreed, challenged, surfaced which
  evidence, requested what), with the turn id
- agent_surfaced_contradictory_evidence: in the failed turn, did the Agent state evidence it holds that contradicts the
  Doctor's diagnosis? "yes" / "no" / "no_contradiction_held"
- gate_b: is there an observable interaction bottleneck at or before the failed turn: the Agent held the correct hypothesis
  or decisive evidence it had already connected to it, and failed to act on it in the interaction (premature agreement,
  not surfacing it, not asserting it as primary, not challenging, not requesting a discriminator it named, not turning
  conveyed evidence into a decision update, demoting the correct diagnosis to secondary)? "yes", "no" or "unclear".
  "no" when the Agent itself kept a wrong diagnosis, had no correct hypothesis before the failed turn, or the failure cannot
  be tied to a concrete interaction action
- gate_b_bottleneck: one concrete sentence with turn ids, e.g. "At turn 3 the Agent held X but accepted the Doctor's Y
  without stating finding Z"; or why there is none
- gate_c: could a rubric fix that bottleneck by changing behavior only, using the hypothesis and evidence the Agent already
  had, without new disease knowledge, the correct diagnosis or case-specific diagnostic knowledge? "yes", "no" or "unclear"
- gate_c_reason: one sentence
- category: "ELIGIBLE_INTERACTION_FAILURE", "KNOWLEDGE_REASONING_LIMITED", "MIXED_BUT_NOT_ELIGIBLE" or "UNCERTAIN"
- confidence: "high", "medium" or "low"

Return exactly one JSON object and no Markdown with these keys."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-audit")
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((run / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    source = read_config(ROOT / config["environment"]["config"])
    cases = split_eligible_cases(source, ROOT, allowed_splits=("train",))
    judge = build_judge(config, ROOT, source)
    correct = lambda gold, text: bool(text) and bool(judge.equivalent(gold, text)["equivalent"])  # noqa: E731
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(run / "base/trajectories.jsonl")}
    analyses = {r["source_trajectory_id"]: r for r in load_latest(run / "iter0/rubrics/failure_analyses.jsonl").values()}
    reader = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader.jsonl")
    previous = {f["case_id"]: f for f in json.loads((ROOT / args.previous_audit).read_text())["failures"]} if args.previous_audit else {}

    def audit(tid: str) -> dict:
        t, a = trajectories[tid], analyses[tid]
        gold, turn = t["ground_truth"]["correct_diagnosis"], a["analysis"]["intervention_turn_id"]
        turns = t["turns"]
        state = turns[turn].get("pre_response_state") or {}
        before = [(i, m.get("assessment")) for i, m in enumerate(turns[:turn]) if m["speaker"] == "agent" and m.get("assessment")]
        before += [(i, (m.get("pre_response_state") or {}).get("agent_current_assessment")) for i, m in enumerate(turns[:turn + 1])
                   if m["speaker"] == "agent" and (m.get("pre_response_state") or {}).get("agent_current_assessment")]
        after = [(i, m.get("assessment")) for i, m in enumerate(turns[turn:], start=turn) if m["speaker"] == "agent" and m.get("assessment")]
        correct_before = sorted({i for i, h in before if correct(gold, h)})
        correct_after = sorted({i for i, h in after if correct(gold, h)})
        doctor = next((m for m in reversed(turns[:turn]) if m["speaker"] == "doctor"), {})
        doctor_dx = doctor.get("provisional_diagnosis") or ""
        mechanical = {
            "standalone_diagnosis": t.get("agent_initial_position"), "standalone_correct": bool(t.get("agent_initial_correct")),
            "correct_hypothesis_before_failed_turn": bool(correct_before), "correct_hypothesis_turns_before": correct_before,
            "correct_hypothesis_after_failed_turn_only": not correct_before and bool(correct_after),
            "agent_hypothesis_before_failed_turn": state.get("agent_current_assessment"),
            "agent_state_better_supported": state.get("better_supported_position"), "agent_state_evidence_sufficient": state.get("evidence_sufficient"),
            "doctor_hypothesis_at_failed_turn": doctor_dx, "doctor_wrong_at_failed_turn": not correct(gold, doctor_dx),
            "agent_assessment_at_failed_turn": turns[turn].get("assessment"),
            "agent_assessment_same_string_as_doctor": bool(doctor_dx) and normalize(turns[turn].get("assessment") or "") == normalize(doctor_dx),
        }
        user = json.dumps({"correct_diagnosis": gold, "everything_the_agent_knew": render(agent_view(cases[t["case_id"]])),
                           "dialogue": [{"turn_id": i, "speaker": m["speaker"], "stage": m["stage"], "text": m["text"],
                                         **({"agent_state_before_this_turn": {k: (m.get("pre_response_state") or {}).get(k) for k in
                                             ("agent_current_assessment", "doctor_current_assessment", "better_supported_position",
                                              "evidence_sufficient", "supporting_evidence_summary")}} if m["speaker"] == "agent" else {})}
                                        for i, m in enumerate(turns)],
                           "failed_turn_id": turn, "failed_turn_stage": turns[turn]["stage"], "mechanical_evidence": mechanical},
                          ensure_ascii=False)
        answer = reader.complete(ChatRequest(PROMPT, user, VERSION, {"temperature": 0, "max_tokens": 900}))
        r, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        r = r or {}
        a1 = bool(correct_before)
        a2 = mechanical["standalone_correct"] and r.get("a2_support_maintained") == "yes"
        gate_a = a1 or a2
        b, c = r.get("gate_b"), r.get("gate_c")
        if gate_a and b == "yes" and c == "yes":
            category = "ELIGIBLE_INTERACTION_FAILURE"
        elif "unclear" in (b, c) or not r:
            category = "UNCERTAIN"
        elif not gate_a and b == "yes":
            category = "MIXED_BUT_NOT_ELIGIBLE"
        else:
            category = "KNOWLEDGE_REASONING_LIMITED"
        p = previous.get(t["case_id"], {})
        return {"case_id": t["case_id"], "gold": gold, "failed_turn": turns[turn]["stage"], "failed_turn_id": turn, **mechanical,
                "gate_a": gate_a, "gate_a1": a1, "gate_a2": a2, "a3_gold_supporting_evidence_flag": r.get("a3_gold_supporting_evidence"),
                "a2_support_maintained": r.get("a2_support_maintained"),
                "agent_behavior_at_failed_turn": r.get("agent_behavior_at_failed_turn"),
                "agent_surfaced_contradictory_evidence": r.get("agent_surfaced_contradictory_evidence"),
                "gate_b": b, "gate_b_bottleneck": r.get("gate_b_bottleneck"), "gate_c": c, "gate_c_reason": r.get("gate_c_reason"),
                "category": category, "reader_category": r.get("category"), "reader_agrees": r.get("category") == category,
                "confidence": r.get("confidence"), "final_diagnosis": t.get("final_diagnosis"),
                "previous_audit": {"recoverable": p.get("audit_interaction_recoverable"), "failure_reason": p.get("audit_failure_reason"),
                                   "note": "earlier audit of the 4B base collaboration of the same case (another trajectory)"} if p else None}

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(audit, sorted(t for t in trajectories if t in analyses and not trajectories[t]["is_correct"])))
    n = len(rows)
    count = Counter(r["category"] for r in rows)
    summary = {"run": args.run, "failures": n, "categories": {k: count.get(k, 0) for k in CATEGORIES},
               "eligible_case_ids": [r["case_id"][-4:] for r in rows if r["category"] == CATEGORIES[0]],
               "standalone_correct": sum(r["standalone_correct"] for r in rows),
               "correct_hypothesis_before_failed_turn": sum(r["correct_hypothesis_before_failed_turn"] for r in rows),
               "correct_hypothesis_after_failed_turn_only": sum(r["correct_hypothesis_after_failed_turn_only"] for r in rows),
               "evidence_only_A3_without_correct_hypothesis": sum(r["a3_gold_supporting_evidence_flag"] == "yes" and not r["gate_a"] for r in rows),
               "gate_a": sum(r["gate_a"] for r in rows), "identifiable_interaction_bottleneck_B": sum(r["gate_b"] == "yes" for r in rows),
               "behavior_only_actionable_C": sum(r["gate_c"] == "yes" for r in rows),
               "all_three_gates": count.get(CATEGORIES[0], 0),
               "primary_eligibility_rate": round(count.get(CATEGORIES[0], 0) / n, 3) if n else None,
               "reader_agrees_with_rule": sum(r["reader_agrees"] for r in rows),
               "reader": config["roles"]["rubric_manager"]["model_id"] + " (analysis only; gates B, C and the A2/A3 readings)",
               "note": "diagnostic only; nothing here selects failures for any run"}
    if previous:
        summary["previous_recoverable_yes_now"] = {r["case_id"][-4:]: r["category"] for r in rows
                                                   if (r["previous_audit"] or {}).get("recoverable") == "yes"}
    (out / "eligibility_audit.json").write_text(json.dumps({"summary": summary, "failures": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "eligibility_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in rows:
        print(r["case_id"][-4:], r["failed_turn"], "| alone", r["standalone_diagnosis"], r["standalone_correct"], "| before", r["agent_hypothesis_before_failed_turn"],
              "| doctor", r["doctor_hypothesis_at_failed_turn"], "| A", r["gate_a"], "(A1", r["gate_a1"], "A2", r["gate_a2"], "A3", r["a3_gold_supporting_evidence_flag"],
              ") B", r["gate_b"], "C", r["gate_c"], "|", r["category"], "/ reader", r["reader_category"], r["confidence"])
        print("     B:", r["gate_b_bottleneck"])
        print("     C:", r["gate_c_reason"])


if __name__ == "__main__":
    main()
