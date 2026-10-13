"""Audit of the failed TRAIN collaborations behind an iteration's rubric step: why did no (or few) rubric candidates pass
causal validation? Nothing of the method is changed; no threshold, rule or selection is touched.
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
from src.agentclinic.selection import split_eligible_cases
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

ROOT = Path(__file__).resolve().parents[2]
GENERIC = {"syndrome", "disease", "disorder", "of", "the", "with", "and", "acute", "chronic", "type"}
PROMPT = """You audit one failed clinical collaboration for a research analysis. A consulting Agent (who alone has the
examination and test findings) talked with a primary Doctor (who has only the history) and the Doctor's final diagnosis
was wrong. Afterwards the collaboration was rerun three times from one Agent turn, with the Agent instructed to show an
"ideal behavior" at that turn; the final diagnosis was wrong again every time.

You receive the correct diagnosis, everything the Agent knew, the original dialogue, the ideal behavior, and the three
reruns (the Agent's reply at the rerun turn, the rest of the dialogue, the final diagnosis).

Answer about what happened, not about what should be done:
- private_evidence_supports_correct: do the Agent's own findings support the correct diagnosis over the diagnoses that
  were discussed? "yes" (clearly), "partly" (compatible but not discriminative), "no"
- agent_reached_correct_diagnosis: did the Agent itself state or argue for the correct diagnosis at any point of the
  original dialogue or of the reruns? true / false
- rerun_performed_ideal_behavior: did the rerun replies actually perform the ideal behavior? "yes", "partly", "no"
- failure_reason, one of:
    "knowledge_diagnostic_error"  the Agent (and the Doctor) did not arrive at the correct diagnosis: the Agent's own
                                  reasoning from its findings went to another diagnosis, so no way of interacting could
                                  have carried the correct one
    "interaction_error"           the Agent had or could state the correct diagnosis or the decisive finding but did not
                                  convey, defend or ask in a way that changed the outcome
    "doctor_resistance"           the Agent conveyed the correct diagnosis or the decisive evidence and the Doctor still
                                  finalized another diagnosis
    "mixed"                       more than one of these, or unclear
- interaction_recoverable: could a different interaction behavior of the Agent at that turn, with the knowledge the Agent
  showed, plausibly have led to the correct final diagnosis? "yes", "no", "unclear"
- reason: two sentences.

Return exactly one JSON object and no Markdown:
{"private_evidence_supports_correct": "yes", "agent_reached_correct_diagnosis": false, "rerun_performed_ideal_behavior": "yes",
 "failure_reason": "knowledge_diagnostic_error", "interaction_recoverable": "no", "reason": "..."}"""


def names(gold: str, text: str) -> bool:
    words = [w for w in normalize(gold).split() if w not in GENERIC]
    return bool(words) and all(w in normalize(text).split() for w in words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    cases = split_eligible_cases(read_config(ROOT / config["environment"]["config"]), ROOT, allowed_splits=("train",))
    rubrics = run / f"iter{args.iteration}" / "rubrics"
    source = run / ("base" if args.iteration == 0 else f"iter{args.iteration - 1}") / "trajectories.jsonl"
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(source)}
    analyses = {a["source_trajectory_id"]: a for a in read_jsonl(rubrics / "failure_analyses.jsonl")}
    validations = {v["source_trajectory_id"]: v for v in read_jsonl(rubrics / "validations.jsonl")}
    chat = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_auditor.jsonl")

    def audit(tid: str) -> dict:
        t, a, v = trajectories[tid], analyses[tid], validations[tid]
        case, gold = cases[t["case_id"]], t["ground_truth"]["correct_diagnosis"]
        turn = t["turns"][v["intervention_turn_id"]]
        state = turn.get("pre_response_state") or {}
        ideal = (a["candidate_rubric"].get("scores") or {}).get("2")
        agent_texts = [m["text"] for m in t["turns"] if m["speaker"] == "agent"] + [t.get("agent_initial_position") or ""]
        rerun_texts = [m["text"] for r in v["reruns"] for m in r["dialogue"] if m["speaker"] == "agent"]
        row = {"case_id": t["case_id"], "correct_diagnosis": gold, "turns": t["num_turns"], "stop": t.get("stop"),
               "agent_standalone_diagnosis": t.get("agent_initial_position"), "agent_standalone_correct": t.get("agent_initial_correct"),
               "doctor_initial_diagnosis": t.get("doctor_initial_claim"), "doctor_initial_correct": t.get("doctor_initial_correct"),
               "collaboration_state": t.get("collaboration_state"), "final_diagnosis": t.get("final_diagnosis"),
               "failed_turn": turn["stage"], "state_better_supported_position": state.get("better_supported_position"),
               "state_evidence_sufficient": state.get("evidence_sufficient"), "state_agent_assessment": state.get("agent_current_assessment"),
               "rubric": a["candidate_rubric"].get("name"), "score_2_behavior": ideal if isinstance(ideal, str) else json.dumps(ideal, ensure_ascii=False),
               "rerun_final_diagnoses": [r.get("final_diagnosis") for r in v["reruns"]], "rerun_successes": v["successes"],
               "agent_named_correct_in_base_dialogue": any(names(gold, x) for x in agent_texts),
               "agent_named_correct_in_reruns": any(names(gold, x) for x in rerun_texts),
               "doctor_named_correct_anywhere": any(names(gold, m["text"]) for m in t["turns"] if m["speaker"] == "doctor")
               or any(names(gold, m["text"]) for r in v["reruns"] for m in r["dialogue"] if m["speaker"] == "doctor")}
        user = json.dumps({"correct_diagnosis": gold, "everything_the_agent_knew": render(agent_view(case)),
                           "original_dialogue": [{"speaker": m["speaker"], "stage": m["stage"], "text": m["text"]} for m in t["turns"]],
                           "rerun_from_turn": turn["stage"], "ideal_behavior": ideal,
                           "reruns": [{"agent_reply": r.get("intervention_response"), "final_diagnosis": r.get("final_diagnosis"),
                                       "dialogue_after": [{"speaker": m["speaker"], "stage": m["stage"], "text": m["text"]}
                                                          for m in r["dialogue"][v["intervention_turn_id"] + 1:]]} for r in v["reruns"]]}, ensure_ascii=False)
        answer = chat.complete(ChatRequest(PROMPT, user, "failure-audit-v1", {"temperature": 0, "max_tokens": 700}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        row.update({f"audit_{k}": (parsed or {}).get(k) for k in ("private_evidence_supports_correct", "agent_reached_correct_diagnosis",
                                                                    "rerun_performed_ideal_behavior", "failure_reason", "interaction_recoverable", "reason")})
        return row

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(audit, sorted(validations)))
    summary = {"run": args.run, "iteration": args.iteration, "failures": len(rows),
               "auditor": config["roles"]["rubric_manager"]["model_id"] + " (analysis only; not part of the method, not a human judgment)",
               "agent_standalone_correct": dict(Counter(str(r["agent_standalone_correct"]) for r in rows)),
               "doctor_initial_correct": dict(Counter(str(r["doctor_initial_correct"]) for r in rows)),
               "collaboration_state": dict(Counter(str(r["collaboration_state"]) for r in rows)),
               "stop": dict(Counter(str(r["stop"]) for r in rows)), "failed_turn": dict(Counter(r["failed_turn"] for r in rows)),
               "agent_named_correct_in_base_dialogue": sum(r["agent_named_correct_in_base_dialogue"] for r in rows),
               "agent_named_correct_in_reruns": sum(r["agent_named_correct_in_reruns"] for r in rows),
               "doctor_named_correct_anywhere": sum(r["doctor_named_correct_anywhere"] for r in rows),
               **{f"audit_{k}": dict(Counter(str(r[f"audit_{k}"]) for r in rows)) for k in
                  ("private_evidence_supports_correct", "agent_reached_correct_diagnosis", "rerun_performed_ideal_behavior", "failure_reason", "interaction_recoverable")}}
    (out / "failure_audit.json").write_text(json.dumps({"summary": summary, "failures": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "failure_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for r in rows:
        print(r["case_id"][-4:], "|", r["collaboration_state"], "| agent0", r["agent_standalone_correct"], "| state", r["state_better_supported_position"], r["state_evidence_sufficient"],
              "| agent named gold base/rerun", r["agent_named_correct_in_base_dialogue"], r["agent_named_correct_in_reruns"], "|", r["audit_failure_reason"], "| recoverable", r["audit_interaction_recoverable"],
              "| evidence", r["audit_private_evidence_supports_correct"])


if __name__ == "__main__":
    main()
