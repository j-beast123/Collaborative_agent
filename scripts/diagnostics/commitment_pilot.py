"""Diagnostic pilot (not the method, not read by any run): does a generic final-diagnosis commitment instruction turn the
4B FULL near misses into strict successes? The judge (majority-v1 mapping), rubrics, cases and the FULL condition of
scripts/diagnostics/backbone_capacity_pilot.py are unchanged; only the instruction below is added. No gold diagnosis, no case hint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.diagnostics.backbone_capacity_pilot import load_any
from scripts.diagnostics.failure_audit import names
from src.agentclinic.pipeline import Appender
from src.common import usage
from src.common.io import read_jsonl
from src.judges.failure_analysis import load_latest

ROOT = Path(__file__).resolve().parents[2]
RERUNS = 3
STEPS = """1. Commit to ONE primary diagnosis that best explains the overall evidence.
2. Use the most specific diagnosis that is actually supported by the available evidence.
3. Do not replace a specific disease/process with a broader syndrome or manifestation unless the broader label is genuinely the intended primary diagnosis.
4. If one diagnosis is a manifestation, complication, or associated syndrome of another, identify which one should be the primary diagnosis for this case.
5. Do not list multiple alternatives as the final answer.
6. If some uncertainty remains, still choose the single best-supported primary diagnosis; uncertainty may be stated after the diagnosis.
7. Base this only on evidence already available in the conversation. Do not introduce new medical facts."""
INSTRUCTION = {
    "agent": ("[FINAL DIAGNOSIS COMMITMENT INSTRUCTION]\nBefore giving the diagnosis you recommend:\n" + STEPS +
              "\nYour Current assessment line names that single primary diagnosis."),
    "doctor": ("[FINAL DIAGNOSIS COMMITMENT INSTRUCTION]\nBefore giving your final diagnosis:\n" + STEPS +
               "\nThe final_diagnosis field holds that single primary diagnosis."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=("agent", "doctor"), required=True)
    parser.add_argument("--backbone-pilot", required=True)
    parser.add_argument("--model", required=True, help="local Qwen3-4B checkpoint")
    args = parser.parse_args()
    import src.agentclinic.protocol as protocol
    from src.agentclinic.runtime import build_clinic
    from src.agentclinic.trajectory import dialogue_of
    from src.grpo.trainer import spec_of

    out = ROOT / args.backbone_pilot
    targeted = (out / "run.sh").read_text(encoding="utf-8").split("--targeted-pilot ")[1].split()[0]
    config = dict(json.loads((ROOT / targeted / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.backbone_pilot)
    trajectories = {t["trajectory_id"]: t for t in read_jsonl(ROOT / targeted / "base/trajectories.jsonl")}
    records = sorted(load_latest(ROOT / targeted / "iter0/rubrics/failure_analyses.jsonl").values(), key=lambda r: r["source_case_id"])
    if args.side == "doctor":  # this process only: the Doctor's finalizing prompts carry the instruction
        original = protocol.doctor_prompt
        protocol.doctor_prompt = lambda case, dialogue, stage: original(case, dialogue, stage) + (
            "\n\n" + INSTRUCTION["doctor"] if stage in ("FINAL", "NEXT") else "")
    model, tokenizer = load_any(args.model, config["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    spec = spec_of(clinic.source)
    rows = Appender(out / f"rollouts_4b_commit_{args.side}.jsonl", key=lambda r: (r["rubric_id"], r["rerun"]))
    correct = lambda gold, text: bool(text) and bool(clinic.judge.equivalent(gold, text)["equivalent"])  # noqa: E731

    def one(job: tuple) -> None:
        record, index = job
        if (record["rubric_id"], index) in rows.keys:
            return
        t = trajectories[record["source_trajectory_id"]]
        case, gold = clinic.cases[t["case_id"]], t["ground_truth"]["correct_diagnosis"]
        guideline = INSTRUCTION["agent"] if args.side == "agent" else ""
        for attempt in range(1, config["rollout"]["max_attempts"] + 1):  # as collaborate(): a format failure is rerun
            with usage.case(case["case_id"]):
                result = protocol.continue_dialogue(case, dialogue_of(t, 1), clinic.runtime, guideline=guideline, spec=spec,
                                                    persistent_guideline=bool(guideline))
            if result["status"] == "completed":
                break
        dialogue = result["dialogue"]
        rows.add({"backbone": "4b", "condition": f"full_commit_{args.side}", "case_id": t["case_id"], "rubric_id": record["rubric_id"],
                  "rerun": index, "status": result["status"], "attempts": attempt, "final_diagnosis": result["final_diagnosis"],
                  "correct": correct(gold, result["final_diagnosis"]) if result["status"] == "completed" else False,
                  "instruction": INSTRUCTION[args.side], "side": args.side, "from_turn": 1,
                  "correct_diagnosis_appeared": any(names(gold, m["text"]) for m in dialogue[1:]),
                  "doctor_reached_correct": any(correct(gold, m.get("provisional_diagnosis") or m.get("final_diagnosis") or "")
                                                for m in dialogue[1:] if m["speaker"] == "doctor"),
                  "agent_assessment_reached_correct": any(correct(gold, m.get("assessment") or "") for m in dialogue if m["speaker"] == "agent"),
                  "dialogue": [{k: m.get(k) for k in ("speaker", "stage", "text", "assessment", "provisional_diagnosis", "final_diagnosis") if m.get(k)}
                               for m in dialogue]})

    with ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(one, [(r, i) for r in records for i in range(RERUNS)]))
    print(Counter((r["status"], r["correct"]) for r in rows.rows))


if __name__ == "__main__":
    main()
