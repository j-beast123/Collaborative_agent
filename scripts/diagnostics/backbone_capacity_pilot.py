"""Diagnostic pilot (not the method, not read by any run): does a larger Agent backbone change the picture on the six
audit-recoverable failures? Only the Agent model differs; Doctor, prompts, decoding, judge (majority-v1 mapping), rubrics
(those of the interaction-targeted pilot) and the 2-of-3 criterion are those of the clean run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.diagnostics.failure_audit import names
from src.agentclinic.pipeline import Appender
from src.collaboration.rerun import intervention_instruction
from src.collaboration.runtime import run_with_retries
from src.common import usage
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.failure_analysis import load_latest
from src.judges.rubric_scoring import build_scoring_items, rubric_view

ROOT = Path(__file__).resolve().parents[2]
RERUNS, MIN_SUCCESSES = 3, 2
CONDITIONS = ("full", "unguided", "persistent", "decision")


def load_any(path: str, policy: dict):
    """The Agent model in bf16 over every visible GPU (device_map auto): a 14B model does not fit one 24 GB GPU.
    Generation only; the prompts, chat template (thinking off) and decoding are those of the run."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, policy["torch_dtype"]), device_map="auto")
    model.eval()
    return model, tokenizer


def adherence(scores: list) -> dict:
    applicable = [s for s in scores if s in (0, 1, 2)]
    return {"agent_turns": len(scores), **{f"score_{k}": sum(s == k for s in scores) for k in (2, 1, 0)},
            "not_applicable": sum(s == "NA" for s in scores),
            "score2_rate": sum(s == 2 for s in scores) / len(scores) if scores else None,
            "score2_rate_among_applicable": sum(s == 2 for s in applicable) / len(applicable) if applicable else None,
            "mean_score": sum(applicable) / len(applicable) if applicable else None}


def run(args) -> None:
    from src.agentclinic.protocol import continue_dialogue
    from src.agentclinic.runtime import build_clinic
    from src.agentclinic.trajectory import agent_state, collaborate, dialogue_of
    from src.grpo.trainer import build_scorer, spec_of

    pilot, out = ROOT / args.targeted_pilot, ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = dict(json.loads((pilot / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(pilot / "base/trajectories.jsonl")}
    records = sorted(load_latest(pilot / "iter0/rubrics/failure_analyses.jsonl").values(), key=lambda r: r["source_case_id"])
    assert len(records) == 6
    model, tokenizer = load_any(args.model, config["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    spec, scorer = spec_of(clinic.source), build_scorer(config, ROOT)
    (out / f"backbone_{args.backbone}.json").write_text(json.dumps({"backbone": args.backbone, "model": args.model,
                                                                     "revision": args.revision, "placement": str(getattr(model, "hf_device_map", None))},
                                                                    indent=1), encoding="utf-8")
    rows = Appender(out / f"rollouts_{args.backbone}.jsonl", key=lambda r: (r["condition"], r["rubric_id"], r["rerun"]))
    conditions = [c for c in CONDITIONS if not (c == "persistent" and args.persistent_pilot) and not (c == "decision" and not args.decision_rubrics)]
    decision = {}
    if args.decision_rubrics:
        checks = {r["case_id"]: r for r in read_jsonl(ROOT / args.decision_rubrics / "decision_rubric_checks.jsonl")}
        decision = {r["source_case_id"]: r for r in load_latest(ROOT / args.decision_rubrics / "failure_analyses.jsonl").values()
                    if checks.get(r["source_case_id"], {}).get("status") == "valid"}

    def one(job: tuple) -> None:
        condition, record, index = job
        if condition == "decision":
            if record["source_case_id"] not in decision:
                return
            record = decision[record["source_case_id"]]
        if (condition, record["rubric_id"], index) in rows.keys:
            return
        t, turn = trajectories[record["source_trajectory_id"]], record["analysis"]["intervention_turn_id"]
        case, gold = clinic.cases[t["case_id"]], t["ground_truth"]["correct_diagnosis"]
        row = {"backbone": args.backbone, "condition": condition, "case_id": t["case_id"], "rubric_id": record["rubric_id"], "rerun": index}
        if condition == "full":
            trajectory = collaborate(case, dialogue_of(t, 1), clinic.runtime, clinic.judge, spec=spec,
                                     max_attempts=config["rollout"]["max_attempts"], trajectory_id=f"{t['case_id']}:{args.backbone}:full:{index}")
            dialogue = trajectory["turns"]
            row.update(status=trajectory["status"], final_diagnosis=trajectory["final_diagnosis"], correct=bool(trajectory["is_correct"]),
                       standalone_diagnosis=trajectory["agent_initial_position"], standalone_correct=trajectory["agent_initial_correct"],
                       scores_by_turn=None, from_turn=1)
        else:
            instruction = intervention_instruction(record["analysis"]["ideal_response_behavior"]) if condition in ("persistent", "decision") else ""

            def once() -> dict:
                states = {}
                with usage.case(case["case_id"]):
                    result = continue_dialogue(case, dialogue_of(t, turn), clinic.runtime, guideline=instruction, states=states,
                                               spec=spec, persistent_guideline=bool(instruction))
                return dict(result, states=states)

            result = run_with_retries(once, format_attempts=config["validation"].get("max_rerun_attempts", 1))
            verdict = clinic.judge.equivalent(gold, result["final_diagnosis"] or "") if result["status"] == "completed" else {"equivalent": None}
            dialogue = result["dialogue"]
            state = lambda i, m: t["turns"][turn]["pre_response_state"] if i == turn else agent_state(case, result["states"][m["stage"]])  # noqa: E731
            scored = dict(t, turns=[dict(m, pre_response_state=state(i, m)) if m["speaker"] == "agent" and i >= turn
                                    else (t["turns"][i] if i < turn else m) for i, m in enumerate(dialogue)])
            labels = {}
            for item in build_scoring_items(scored, [rubric_view(record)], include_initial_turn=True):
                if item.turn_id >= turn:
                    labels[dialogue[item.turn_id]["stage"]] = scorer.score(item).labels.get(record["rubric_id"])
            row.update(status=result["status"], final_diagnosis=result["final_diagnosis"], correct=bool(verdict["equivalent"]),
                       instruction=instruction, scores_by_turn=labels, from_turn=turn)
        later = dialogue[row["from_turn"]:]
        correct = lambda text: bool(text) and bool(clinic.judge.equivalent(gold, text)["equivalent"])  # noqa: E731
        row.update(correct_diagnosis_appeared=any(names(gold, m["text"]) for m in later),
                   doctor_reached_correct=any(correct(m.get("provisional_diagnosis") or m.get("final_diagnosis") or "")
                                              for m in later if m["speaker"] == "doctor"),
                   agent_assessment_reached_correct=any(correct(m.get("assessment") or "") for m in later if m["speaker"] == "agent"),
                   agent_turns=sum(m["speaker"] == "agent" for m in dialogue),
                   dialogue=[{k: m.get(k) for k in ("speaker", "stage", "text", "assessment", "provisional_diagnosis", "final_diagnosis") if m.get(k)}
                             for m in dialogue])
        rows.add(row)

    with ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(one, [(c, r, i) for c in conditions for r in records for i in range(RERUNS)]))
    print(Counter((r["condition"], r["status"], r["correct"]) for r in rows.rows))
    effectiveness(config, rows.rows, clinic.cases, out / f"effectiveness_{args.backbone}.jsonl")


def effectiveness(config: dict, rows: list[dict], cases: dict, path: Path) -> None:
    """R_E of every Agent response from the failed turn on, from one thread (the predictor service)."""
    from src.agentclinic.adapter import agent_view, render
    from src.agentclinic.pipeline import QUESTION
    from src.effectiveness.service import build_effectiveness
    from src.grpo.trainer import effectiveness_settings

    provider = build_effectiveness(effectiveness_settings(config, 0), 1.0, ROOT)
    effects = Appender(path, key=lambda r: r["key"])
    for row in rows:
        if row["condition"] == "full" or row["status"] != "completed":
            continue
        dialogue = row["dialogue"]
        for i, m in enumerate(dialogue):
            key = f"{row['backbone']}:{row['condition']}:{row['rubric_id']}:{row['rerun']}:{i}"
            if m["speaker"] != "agent" or i < row["from_turn"] or key in effects.keys or not m.get("text"):
                continue
            history = {"history_id": key, "intervention_turn": m["stage"], "question": QUESTION, "options": {},
                       "scenario": render(agent_view(cases[row["case_id"]])),
                       "turns": [{"speaker": x["speaker"], "text": x["text"]} for x in dialogue[:i]]}
            prediction = provider.predict(history, m["text"], m["stage"], trajectory_id=key)
            effects.add({"key": key, "backbone": row["backbone"], "condition": row["condition"], "case_id": row["case_id"],
                         "rubric_id": row["rubric_id"], "rerun": row["rerun"], "turn_id": i, "stage": m["stage"],
                         "failed_turn": i == row["from_turn"], "delta": prediction.delta})


def report(args) -> None:
    pilot, out = ROOT / args.targeted_pilot, ROOT / args.output_dir
    records = sorted(load_latest(pilot / "iter0/rubrics/failure_analyses.jsonl").values(), key=lambda r: r["source_case_id"])
    decision = {r["source_case_id"]: r for r in load_latest(ROOT / args.decision_rubrics / "failure_analyses.jsonl").values()} if args.decision_rubrics else {}
    checks = {r["case_id"]: r for r in read_jsonl(ROOT / args.decision_rubrics / "decision_rubric_checks.jsonl")} if args.decision_rubrics else {}
    rows = [r for path in sorted(out.glob("rollouts_*.jsonl")) for r in read_jsonl(path)]
    effects = [e for path in sorted(out.glob("effectiveness_*.jsonl")) for e in read_jsonl(path)]
    if args.persistent_pilot:  # the 4B persistent rollouts of the earlier pilot, as they are
        for r in read_jsonl(ROOT / args.persistent_pilot / "persistent_rollouts.jsonl"):
            rows.append({"backbone": "4b", "condition": "persistent", "case_id": r["case_id"], "rubric_id": r["rubric_id"], "rerun": r["rerun"],
                         "status": r["status"], "final_diagnosis": r["final_diagnosis"], "correct": r["correct"],
                         "scores_by_turn": {x["stage"]: x["score"] for x in r["guided_turns"]}, "reused_from": args.persistent_pilot,
                         "correct_diagnosis_appeared": None, "doctor_reached_correct": None, "agent_assessment_reached_correct": None,
                         "from_turn": r["intervention_turn_id"]})
        effects += [dict(e, backbone="4b", condition="persistent") for e in read_jsonl(ROOT / args.persistent_pilot / "effectiveness.jsonl")
                    if e["condition"] == "B"]
    groups = [(b, c) for b in ("4b", "14b") for c in CONDITIONS if any(r["backbone"] == b and r["condition"] == c for r in rows)]
    mean = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731

    def stats(selected: list[dict], b: str, c: str, case_ids: set) -> dict:
        n = {"rollouts": len(selected), "completed": sum(r["status"] == "completed" for r in selected),
             "final_success": sum(r["correct"] for r in selected),
             "case_pass": sum(sum(r["correct"] for r in selected if r["case_id"] == k) >= MIN_SUCCESSES for k in case_ids),
             "doctor_reached_correct": sum(bool(r.get("doctor_reached_correct")) for r in selected),
             "agent_assessment_reached_correct": sum(bool(r.get("agent_assessment_reached_correct")) for r in selected),
             "correct_diagnosis_appeared": sum(bool(r.get("correct_diagnosis_appeared")) for r in selected)}
        if c == "full":
            n["standalone_correct"] = sum(bool(r.get("standalone_correct")) for r in selected)
        else:
            n["adherence"] = adherence([s for r in selected for s in (r["scores_by_turn"] or {}).values()])
            n["adherence"]["score_ge1_rate"] = ((n["adherence"]["score_2"] + n["adherence"]["score_1"]) / n["adherence"]["agent_turns"]
                                                if n["adherence"]["agent_turns"] else None)
            mine = [e for e in effects if e["backbone"] == b and e["condition"] == c and e["case_id"] in case_ids]
            n["R_E_failed_turn"] = mean([e["delta"] for e in mine if e["failed_turn"]])
            n["R_E_later_turns"] = mean([e["delta"] for e in mine if not e["failed_turn"]])
        return n

    all_cases = {r["source_case_id"] for r in records}
    summary = {f"{b}_{c}": stats([r for r in rows if r["backbone"] == b and r["condition"] == c], b, c, all_cases) for b, c in groups}
    pairs = {"capacity": [("4b_full", "14b_full"), ("4b_unguided", "14b_unguided"), ("4b_persistent", "14b_persistent")],
             "rubric_abstraction_14b": [("14b_unguided", "14b_persistent"), ("14b_persistent", "14b_decision"), ("14b_unguided", "14b_decision")]}
    comparisons = {name: [{"left": a, "right": b, "left_stats": summary.get(a), "right_stats": summary.get(b)} for a, b in pair]
                   for name, pair in pairs.items()}
    cases = []
    for record in records:
        k = record["source_case_id"]
        row = {"case_id": k, "existing_rubric": record["candidate_rubric"]["name"],
               "existing_rubric_score_2": record["candidate_rubric"]["scores"]["2"]["behavior"],
               "decision_rubric": decision[k]["candidate_rubric"]["name"] if k in decision else None,
               "decision_rubric_score_2": decision[k]["candidate_rubric"]["scores"]["2"]["behavior"] if k in decision else None,
               "decision_rubric_status": (checks.get(k) or {}).get("status")}
        for b, c in groups:
            mine = sorted((r for r in rows if r["backbone"] == b and r["condition"] == c and r["case_id"] == k), key=lambda r: r["rerun"])
            if not mine:
                continue
            row[f"{b}_{c}"] = {**stats(mine, b, c, {k}), "final_diagnoses": [r["final_diagnosis"] for r in mine],
                               **({"standalone_diagnoses": [r.get("standalone_diagnosis") for r in mine]} if c == "full" else {})}
        cases.append(row)
    result = {"summary": summary, "comparisons": comparisons, "cases": cases,
              "note": ("diagnostic pilot; only the Agent backbone differs between 4b and 14b, only the rubric between persistent and decision. "
                       "Criterion (>= 2 of 3 correct), judge, Doctor, prompts, decoding unchanged. The dialogue before the failed turn was "
                       "written with 4B in every unguided / persistent / decision rollout. 4b_persistent is the earlier pilot's (reused). "
                       "Six cases: an indication, not an estimate.")}
    (out / "backbone_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for c in cases:
        cells = []
        for key, v in c.items():
            if isinstance(v, dict):
                cell = f"{key} {v['final_success']}/3"
                if "standalone_correct" in v:
                    cell += f" (alone {v['standalone_correct']}/3)"
                if "adherence" in v:
                    a = v["adherence"]
                    cell += f" s2 {a['score_2']}/{a['agent_turns'] - a['not_applicable']}"
                cell += f" A{v['agent_assessment_reached_correct']} D{v['doctor_reached_correct']}"
                cells.append(cell)
        print(c["case_id"][-4:], " | ".join(cells))
        for key, v in c.items():
            if isinstance(v, dict) and key.startswith("14b"):
                print("      ", key, v["final_diagnoses"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targeted-pilot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--persistent-pilot", help="4B: reuse its persistent rollouts instead of sampling them again")
    parser.add_argument("--decision-rubrics", help="directory of scripts/diagnostics/decision_rubric_pilot.py: condition decision")
    parser.add_argument("--stage", choices=("run", "report"), default="run")
    args = parser.parse_args()
    (run if args.stage == "run" else report)(args)


if __name__ == "__main__":
    main()
