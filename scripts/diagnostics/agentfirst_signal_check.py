"""Pre-main signal check of the agent-first / frontier-Doctor protocol (environment v14): does it give a non-degenerate
outcome signal on TRAIN30? One natural collaboration per TRAIN case with the base Agent (Qwen3-4B), and, for evaluation
only, the same Doctor's diagnosis of the case alone (no Agent message). No method component is run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agentclinic.pipeline import Appender
from src.common import usage
from src.common.config import read_config

ROOT = Path(__file__).resolve().parents[2]
ENV = "configs/effectiveness/v14_agentclinic_agentfirst_frontierdoctor_v1.json"
BASE = "outputs/serl_clean/clean30_4b_ours_20261003-085508_34594ece/resolved_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", type=int, help="only the first n TRAIN cases (smoke test)")
    args = parser.parse_args()
    from src.agentclinic.protocol import doctor_prompt, final_diagnosis, parse_transition, protocol_spec
    from src.agentclinic.runtime import build_clinic, load_policy
    from src.agentclinic.trajectory import collaborate

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / BASE).read_text(encoding="utf-8"))
    config = dict(config, output_dir=args.output_dir, environment=dict(config["environment"], config=ENV))
    source = read_config(ROOT / ENV)
    spec = protocol_spec(source["protocol"])
    model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    cases = sorted(k for k, c in clinic.cases.items() if c["split"] == "train")[: args.cases]
    rows = Appender(out / "collaborations.jsonl", key=lambda r: r["case_id"])
    alone = Appender(out / "doctor_alone.jsonl", key=lambda r: r["case_id"])
    correct = lambda gold, text: bool(text) and bool(clinic.judge.equivalent(gold, text)["equivalent"])  # noqa: E731

    def one(case_id: str) -> None:
        case = clinic.cases[case_id]
        if case_id not in rows.keys:
            t = collaborate(case, [], clinic.runtime, clinic.judge, spec=spec, max_attempts=3, trajectory_id=f"{case_id}:agentfirst:base")
            a1 = next((m for m in t["turns"] if m["speaker"] == "agent"), {})
            rows.add({"case_id": case_id, "gold": case["gold"], "a1_assessment": a1.get("assessment"), "a1_correct": correct(case["gold"], a1.get("assessment")),
                      "final_diagnosis": t["final_diagnosis"], "final_correct": bool(t["is_correct"]), "status": t["status"],
                      "trajectory_status": t["trajectory_status"], "num_turns": t["num_turns"], "stop": t.get("stop"),
                      "doctor_first_diagnosis": t["doctor_initial_claim"], "standalone": t["agent_initial_position"], "trajectory": t})
        if case_id not in alone.keys:
            with usage.case(case_id):
                text = clinic.runtime.doctor(clinic.runtime.doctor_system, doctor_prompt(case, [], "FINAL", spec.doctor_full_view))
            diagnosis = final_diagnosis(text) or ((parse_transition(text) or {}).get("final_diagnosis") or "")
            alone.add({"case_id": case_id, "doctor_alone_diagnosis": diagnosis or None, "doctor_alone_correct": correct(case["gold"], diagnosis),
                       "raw": text})

    with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(one, cases))
    by = {r["case_id"]: r for r in rows.rows}
    da = {r["case_id"]: r for r in alone.rows}
    ids = [k for k in cases if k in by and k in da]
    n = len(ids)
    rate = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    cells = Counter((by[k]["a1_correct"], by[k]["final_correct"]) for k in ids)
    summary = {"protocol": spec.name, "doctor": source["models"]["doctor"]["model_id"], "agent": config["policy"]["model_id"], "cases": n,
               "valid_collaborations": sum(by[k]["trajectory_status"] == "valid" for k in ids),
               "A_agent_a1_correct": rate([by[k]["a1_correct"] for k in ids]),
               "B_doctor_final_correct": rate([by[k]["final_correct"] for k in ids]),
               "C_final_given_a1_correct": rate([by[k]["final_correct"] for k in ids if by[k]["a1_correct"]]),
               "C_final_given_a1_wrong": rate([by[k]["final_correct"] for k in ids if not by[k]["a1_correct"]]),
               "D_doctor_alone_correct": rate([da[k]["doctor_alone_correct"] for k in ids]),
               "cells": {"agent_correct_final_correct": cells[(True, True)], "agent_correct_final_wrong": cells[(True, False)],
                         "agent_wrong_final_correct": cells[(False, True)], "agent_wrong_final_wrong": cells[(False, False)]},
               "E_collab_vs_alone": dict(Counter(f"alone_{'ok' if da[k]['doctor_alone_correct'] else 'wrong'}->collab_{'ok' if by[k]['final_correct'] else 'wrong'}" for k in ids)),
               "turns": dict(Counter(by[k]["num_turns"] for k in ids)), "stop": dict(Counter(str(by[k]["stop"]) for k in ids))}
    (out / "signal_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
