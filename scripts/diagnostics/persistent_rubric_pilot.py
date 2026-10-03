"""Diagnostic pilot (not the method, not read by any run): single-turn intervention vs persistent rubric guidance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.diagnostics.failure_audit import names
from scripts.diagnostics.intervention_local_effect import PROMPT, after
from scripts.diagnostics.targeted_rubric_pilot import rubric_text
from src.agentclinic.adapter import agent_view, render
from src.agentclinic.pipeline import QUESTION, Appender
from src.collaboration.rerun import intervention_instruction
from src.collaboration.runtime import run_with_retries
from src.common import usage
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.failure_analysis import load_latest
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object
from src.judges.rubric_scoring import build_scoring_items, rubric_view
from src.judges.rubric_validation import load_validations

ROOT = Path(__file__).resolve().parents[2]
RERUNS, MIN_SUCCESSES = 3, 2
# adherence class of a case, fixed before any rollout was run: the share of score-2 turns among the guided turns the scorer
# found applicable (not NA), over the case's three rollouts
ADHERENCE = {"high": 0.75, "low": 0.25}  # >= high: "A high adherence"; < low: "B low adherence"; between: "C mixed"


def predictor_history(case: dict, dialogue: list[dict], index: int, key: str) -> dict:
    return {"history_id": key, "intervention_turn": dialogue[index]["stage"], "question": QUESTION, "options": {},
            "scenario": render(agent_view(case)), "turns": [{"speaker": m["speaker"], "text": m["text"]} for m in dialogue[:index]]}


def run(args) -> None:
    from src.agentclinic.protocol import continue_dialogue
    from src.agentclinic.runtime import build_clinic, load_policy
    from src.agentclinic.trajectory import agent_state, dialogue_of
    from src.effectiveness.service import build_effectiveness
    from src.grpo.trainer import build_scorer, effectiveness_settings, previous_adapter, spec_of

    pilot, out = ROOT / args.targeted_pilot, ROOT / args.output_dir
    rubrics = out / "iter0/rubrics"
    rubrics.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((pilot / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    config["diagnostic_pilot"] = {"what": "persistent rubric guidance vs single-turn intervention; not a run of the method",
                                  "targeted_pilot": args.targeted_pilot, "reruns": RERUNS, "min_successes": MIN_SUCCESSES,
                                  "adherence_classes": ADHERENCE}
    if not (out / "resolved_config.json").exists():
        (out / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "base").mkdir(exist_ok=True)
        (out / "base/trajectories.jsonl").write_text((pilot / "base/trajectories.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(pilot / "base/trajectories.jsonl")}
    records = sorted(load_latest(pilot / "iter0/rubrics/failure_analyses.jsonl").values(), key=lambda r: r["source_case_id"])
    assert len(records) == 6 and all(r["analysis_status"] == "valid" for r in records)

    model, tokenizer = load_policy(config["policy"], previous_adapter(config, ROOT, 0), trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    spec, scorer = spec_of(clinic.source), build_scorer(config, ROOT)
    provider = build_effectiveness(effectiveness_settings(config, 0), 1.0, ROOT)
    rollouts = Appender(out / "persistent_rollouts.jsonl", key=lambda r: (r["rubric_id"], r["rerun"]))

    def rollout(job: tuple) -> None:
        record, index = job
        if (record["rubric_id"], index) in rollouts.keys:
            return
        t, turn = trajectories[record["source_trajectory_id"]], record["analysis"]["intervention_turn_id"]
        case, gold = clinic.cases[t["case_id"]], t["ground_truth"]["correct_diagnosis"]
        instruction = intervention_instruction(record["analysis"]["ideal_response_behavior"])
        attempts = []

        def once() -> dict:
            states = {}
            with usage.case(case["case_id"]):
                result = continue_dialogue(case, dialogue_of(t, turn), clinic.runtime, guideline=instruction, states=states,
                                           spec=spec, persistent_guideline=True)
            attempts.append(result["status"])
            return dict(result, states=states)

        result = run_with_retries(once, format_attempts=config["validation"].get("max_rerun_attempts", 1))  # format failures only
        verdict = clinic.judge.equivalent(gold, result["final_diagnosis"] or "") if result["status"] == "completed" else {"equivalent": None}
        dialogue, guided = result["dialogue"], []
        # the scorer's input for a guided turn is what it is in training: the dialogue through the turn and the state before it
        # (trajectory.agent_state). The failed turn keeps the state stored with the original turn: nothing before it differs.
        state = lambda i, m: t["turns"][turn]["pre_response_state"] if i == turn else agent_state(case, result["states"][m["stage"]])  # noqa: E731
        scored = dict(t, turns=[dict(m, pre_response_state=state(i, m)) if m["speaker"] == "agent" and i >= turn
                                else (t["turns"][i] if i < turn else m) for i, m in enumerate(dialogue)])
        for item in build_scoring_items(scored, [rubric_view(record)], include_initial_turn=True):
            if item.turn_id < turn:
                continue
            label = scorer.score(item)
            reply = dialogue[item.turn_id + 1] if item.turn_id + 1 < len(dialogue) else {}
            guided.append({"turn_id": item.turn_id, "stage": dialogue[item.turn_id]["stage"], "agent_response": dialogue[item.turn_id]["text"],
                           "agent_assessment": dialogue[item.turn_id].get("assessment"), "score": label.labels.get(record["rubric_id"]),
                           "score_probabilities": label.probabilities.get(record["rubric_id"]), "scorer_status": label.status,
                           "doctor_response": reply.get("text"),
                           "doctor_diagnosis_after": reply.get("provisional_diagnosis") or reply.get("final_diagnosis")})
        rollouts.add({"case_id": t["case_id"], "rubric_id": record["rubric_id"], "source_trajectory_id": record["source_trajectory_id"],
                      "intervention_turn_id": turn, "failed_turn": t["turns"][turn]["stage"], "rerun": index, "instruction": instruction,
                      "status": result["status"], "attempt_statuses": attempts, "final_diagnosis": result["final_diagnosis"],
                      "correct": bool(verdict["equivalent"]), "guided_turns": guided, "dialogue": dialogue})

    with ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(rollout, [(r, i) for r in records for i in range(RERUNS)]))

    # R_E of every Agent response from the failed turn on: condition B, condition A's stored reruns, and the original
    effects = Appender(out / "effectiveness.jsonl", key=lambda r: r["key"])
    single = load_validations(pilot / "iter0/rubrics/validations.jsonl")
    by_id = {r["rubric_id"]: r for r in records}
    jobs = [("B", r["rubric_id"], r["rerun"], r["dialogue"]) for r in rollouts.rows if r["status"] == "completed"]
    jobs += [("A", rid, i, x["dialogue"]) for rid, v in single.items() for i, x in enumerate(v["reruns"])]
    jobs += [("original", r["rubric_id"], 0, trajectories[r["source_trajectory_id"]]["turns"]) for r in records]
    for condition, rid, index, dialogue in jobs:  # the predictor is called from one thread
        t, turn = trajectories[by_id[rid]["source_trajectory_id"]], by_id[rid]["analysis"]["intervention_turn_id"]
        for i, m in enumerate(dialogue):
            key = f"{condition}:{rid}:{index}:{i}"
            if m["speaker"] != "agent" or i < turn or key in effects.keys or not m.get("text"):
                continue
            prediction = provider.predict(predictor_history(clinic.cases[t["case_id"]], dialogue, i, key), m["text"], m["stage"], trajectory_id=key)
            effects.add({"key": key, "condition": condition, "rubric_id": rid, "case_id": t["case_id"], "rerun": index, "turn_id": i,
                         "stage": m["stage"], "failed_turn": i == turn, "delta": prediction.delta, "p_before": prediction.p_before,
                         "p_after": prediction.p_after})

    # the rollouts in the form of a validation file, for scripts/diagnostics/intervention_local_effect.py --any-reruns
    with (rubrics / "validations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            rows = sorted((r for r in rollouts.rows if r["rubric_id"] == record["rubric_id"]), key=lambda r: r["rerun"])
            successes = sum(r["correct"] for r in rows)
            handle.write(json.dumps({"rubric_id": record["rubric_id"], "source_trajectory_id": record["source_trajectory_id"],
                                     "intervention_turn_id": record["analysis"]["intervention_turn_id"],
                                     "validation_status": "pass" if successes >= MIN_SUCCESSES else "fail", "successes": successes,
                                     "num_reruns": RERUNS, "min_successes": MIN_SUCCESSES, "error": None, "condition": "persistent_guideline",
                                     "reruns": [{"status": r["status"], "final_diagnosis": r["final_diagnosis"], "correct": r["correct"],
                                                 "dialogue": r["dialogue"]} for r in rows]}, ensure_ascii=False) + "\n")
    (rubrics / "failure_analyses.jsonl").write_text((pilot / "iter0/rubrics/failure_analyses.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    print("rollouts", Counter(r["status"] for r in rollouts.rows), "correct", sum(r["correct"] for r in rollouts.rows))


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def adherence(turns: list[dict]) -> dict:
    scores = [x["score"] for x in turns]
    applicable = [s for s in scores if s in (0, 1, 2)]
    return {"guided_agent_turns": len(scores), **{f"score_{k}": sum(s == k for s in scores) for k in (2, 1, 0)},
            "not_applicable": sum(s == "NA" for s in scores), "unscored": sum(s not in (0, 1, 2, "NA") for s in scores),
            "mean_rubric_score": mean(applicable),
            "score2_adherence_rate": sum(s == 2 for s in scores) / len(scores) if scores else None,
            "score_ge1_adherence_rate": sum(s in (1, 2) for s in scores) / len(scores) if scores else None,
            "score2_rate_among_applicable": sum(s == 2 for s in applicable) / len(applicable) if applicable else None}


def report(args) -> None:
    pilot, out = ROOT / args.targeted_pilot, ROOT / args.output_dir
    config = json.loads((out / "resolved_config.json").read_text(encoding="utf-8"))
    records = sorted(load_latest(pilot / "iter0/rubrics/failure_analyses.jsonl").values(), key=lambda r: r["source_case_id"])
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(pilot / "base/trajectories.jsonl")}
    single = load_validations(pilot / "iter0/rubrics/validations.jsonl")
    rollouts = read_jsonl(out / "persistent_rollouts.jsonl")
    effects = read_jsonl(out / "effectiveness.jsonl")
    local = {name: {c["case_id"]: c for c in json.loads(path.read_text(encoding="utf-8"))["candidates"]} if path.exists() else {}
             for name, path in (("A", pilot / "local_effect/local_effect.json"), ("B", out / "local_effect/local_effect.json"))}
    from src.agentclinic.runtime import build_judge

    source = read_config(ROOT / config["environment"]["config"])
    judge = build_judge(dict(config, output_dir=args.output_dir), ROOT, source)
    correct = lambda gold, text: bool(text) and bool(judge.equivalent(gold, text)["equivalent"])  # noqa: E731
    chat = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader_a_vs_b.jsonl")

    def delta(condition: str, rid: str, failed: bool | None = None, rerun: int | None = None):
        return mean([e["delta"] for e in effects if e["condition"] == condition and e["rubric_id"] == rid
                     and (failed is None or e["failed_turn"] == failed) and (rerun is None or e["rerun"] == rerun)])

    def versus(record: dict, index: int) -> dict | None:
        """Blind reading, condition A's rerun against condition B's rollout of the same index (analysis only)."""
        a = (single.get(record["rubric_id"]) or {}).get("reruns") or []
        b = [r for r in rollouts if r["rubric_id"] == record["rubric_id"] and r["rerun"] == index and r["status"] == "completed"]
        if index >= len(a) or not b:
            return None
        t, turn = trajectories[record["source_trajectory_id"]], record["analysis"]["intervention_turn_id"]
        b_is_x = int(hashlib.sha256(f"avb:{record['rubric_id']}:{index}".encode()).hexdigest(), 16) % 2 == 0
        branch = {"A": {"continuation": after(a[index]["dialogue"], turn), "final_diagnosis": a[index].get("final_diagnosis")},
                  "B": {"continuation": after(b[0]["dialogue"], turn), "final_diagnosis": b[0]["final_diagnosis"]}}
        user = json.dumps({"correct_diagnosis": t["ground_truth"]["correct_diagnosis"],
                           "shared_dialogue_before": [{"speaker": m["speaker"], "stage": m["stage"], "text": m["text"]} for m in t["turns"][:turn]],
                           "X": branch["B" if b_is_x else "A"], "Y": branch["A" if b_is_x else "B"]}, ensure_ascii=False)
        answer = chat.complete(ChatRequest(PROMPT, user, "intervention-local-effect-v1:a-vs-b", {"temperature": 0, "max_tokens": 500}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        side = {"X": "persistent" if b_is_x else "single_turn", "Y": "single_turn" if b_is_x else "persistent", "same": "same"}
        return {k: side.get((parsed or {}).get(k)) for k in ("doctor_next_message", "final_diagnosis", "discussion_after")}

    cases, turn_rows = [], []
    for record in records:
        t, rid = trajectories[record["source_trajectory_id"]], record["rubric_id"]
        gold, turn = t["ground_truth"]["correct_diagnosis"], record["analysis"]["intervention_turn_id"]
        mine = sorted((r for r in rollouts if r["rubric_id"] == rid), key=lambda r: r["rerun"])
        usable = [r for r in mine if r["status"] == "completed"]
        per_rollout = []
        for r in mine:
            later = r["dialogue"][turn:]
            per_rollout.append({"rerun": r["rerun"] + 1, "status": r["status"], "final_diagnosis": r["final_diagnosis"], "correct": r["correct"],
                                **adherence(r["guided_turns"]), "scores_by_turn": {x["stage"]: x["score"] for x in r["guided_turns"]},
                                "correct_diagnosis_appeared": any(names(gold, m["text"]) for m in later),
                                "doctor_reached_correct": any(correct(gold, m.get("provisional_diagnosis") or m.get("final_diagnosis") or "")
                                                              for m in later if m["speaker"] == "doctor"),
                                "agent_assessment_reached_correct": any(correct(gold, m.get("assessment") or "") for m in later if m["speaker"] == "agent"),
                                "mean_R_E_guided_turns": delta("B", rid, rerun=r["rerun"])})
            turn_rows += [{"case_id": r["case_id"], "rerun": r["rerun"] + 1, "turn_id": x["turn_id"], "stage": x["stage"],
                           "rubric": record["candidate_rubric"]["name"], "rubric_score_2": record["candidate_rubric"]["scores"]["2"]["behavior"],
                           "instruction": r["instruction"], "agent_response": x["agent_response"], "jev_score": x["score"],
                           "agent_assessment": x["agent_assessment"], "doctor_response": x["doctor_response"],
                           "doctor_diagnosis_after": x["doctor_diagnosis_after"]} for x in r["guided_turns"]]
        overall = adherence([x for r in usable for x in r["guided_turns"]])
        rate = overall["score2_rate_among_applicable"]
        kind = "B low adherence" if rate is None or rate < ADHERENCE["low"] else "A high adherence" if rate >= ADHERENCE["high"] else "C mixed"
        successes = sum(r["correct"] for r in usable)
        a = single.get(rid) or {}
        readings = [v for v in (versus(record, i) for i in range(RERUNS)) if v]
        cases.append({
            "case_id": t["case_id"], "correct_diagnosis": gold, "failed_turn": t["turns"][turn]["stage"],
            "rubric": record["candidate_rubric"]["name"], "rubric_text": rubric_text(record),
            "single_turn": {"intervention_feasible": (a.get("intervention") or {}).get("status") == "frozen",
                            "causal_result": a.get("validation_status"), "final_success": a.get("successes") if a.get("reruns") else None,
                            "final_diagnoses": [x.get("final_diagnosis") for x in a.get("reruns", [])]},
            "persistent": {"usable_rollouts": len(usable), "causal_result": "pass" if successes >= MIN_SUCCESSES else "fail",
                           "final_success": successes, "final_diagnoses": [r["final_diagnosis"] for r in mine], **overall,
                           "failure_mode": None if successes >= MIN_SUCCESSES else kind,
                           "failure_mode_basis": f"score 2 in {overall['score_2']} of {overall['guided_agent_turns'] - overall['not_applicable'] - overall['unscored']} "
                                                 f"applicable guided turns ({overall['not_applicable']} NA)", "rollouts": per_rollout},
            "local_effect_vs_original": {name: {k: (local[name].get(t["case_id"]) or {}).get(k) for k in
                                                ("majority_reader_doctor_next_message_closer", "majority_reader_discussion_after_closer",
                                                 "majority_reader_final_diagnosis_closer")} for name in ("A", "B")},
            "blind_a_vs_b": {k: dict(Counter(str(v[k]) for v in readings)) for k in ("doctor_next_message", "discussion_after", "final_diagnosis")} if readings else None,
            "R_E": {"original_failed_turn": delta("original", rid, True), "single_turn_frozen_response": delta("A", rid, True),
                    "persistent_failed_turn": delta("B", rid, True), "persistent_minus_single_failed_turn":
                        (delta("B", rid, True) - delta("A", rid, True)) if delta("A", rid, True) is not None and delta("B", rid, True) is not None else None,
                    "original_later_turns": delta("original", rid, False), "single_turn_later_turns": delta("A", rid, False),
                    "persistent_later_turns": delta("B", rid, False)}})
    every = adherence([x for r in rollouts if r["status"] == "completed" for x in r["guided_turns"]])
    summary = {
        "single_turn": {"feasible": sum(c["single_turn"]["intervention_feasible"] for c in cases),
                        "causal_pass": sum(c["single_turn"]["causal_result"] == "pass" for c in cases),
                        "final_success_reruns": sum(c["single_turn"]["final_success"] or 0 for c in cases),
                        "reruns": sum(len(c["single_turn"]["final_diagnoses"]) for c in cases)},
        "persistent": {"cases_with_usable_rollout": sum(c["persistent"]["usable_rollouts"] > 0 for c in cases),
                       "usable_rollouts": sum(c["persistent"]["usable_rollouts"] for c in cases),
                       "causal_pass": sum(c["persistent"]["causal_result"] == "pass" for c in cases),
                       "final_success_reruns": sum(c["persistent"]["final_success"] for c in cases), **every,
                       "failure_modes": dict(Counter(str(c["persistent"]["failure_mode"]) for c in cases)),
                       "rollouts_correct_diagnosis_appeared": sum(r["correct_diagnosis_appeared"] for c in cases for r in c["persistent"]["rollouts"]),
                       "rollouts_doctor_reached_correct": sum(r["doctor_reached_correct"] for c in cases for r in c["persistent"]["rollouts"]),
                       "rollouts_agent_assessment_reached_correct": sum(r["agent_assessment_reached_correct"] for c in cases for r in c["persistent"]["rollouts"])},
        "blind_a_vs_b": {k: dict(sum((Counter(c["blind_a_vs_b"][k]) for c in cases if c["blind_a_vs_b"]), Counter()))
                         for k in ("doctor_next_message", "discussion_after", "final_diagnosis")},
        "adherence_classes": ADHERENCE,
        "note": "diagnostic pilot; nothing here enters any run. The criterion (>= 2 of 3 correct final diagnoses), judge, predictor and weights are unchanged."}
    (out / "paired_comparison.json").write_text(json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "persistent_turns.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(turn_rows[0]))
        writer.writeheader()
        writer.writerows(turn_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for c in cases:
        p = c["persistent"]
        print(c["case_id"][-4:], c["failed_turn"], "| A", c["single_turn"]["causal_result"], "| B", p["causal_result"], p["final_success"], "/3 |",
              [r["scores_by_turn"] for r in p["rollouts"]], "|", p["failure_mode"], "|", p["final_diagnoses"])
        print("     local vs original", c["local_effect_vs_original"], "| A vs B", c["blind_a_vs_b"], "| R_E", {k: round(v, 3) if v is not None else None for k, v in c["R_E"].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targeted-pilot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("run", "report"), default="run")
    args = parser.parse_args()
    (run if args.stage == "run" else report)(args)


if __name__ == "__main__":
    main()
