"""The controlled-evidence inputs of RQ2 for a set of reference trajectories, built and audited without any Agent model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts import run_eval as E
from src.agentclinic.adapter import doctor_view, render
from src.agentclinic.evaluation import ClinicEvidenceBuilder, controlled_items
from src.agentclinic.runtime import build_judge
from src.common.config import read_config
from src.common.io import read_jsonl
from src.common.main_config import load_main_config

ROOT = E.ROOT
CONDITIONS = ("correct_strong", "correct_weak", "incorrect_strong_looking", "incorrect_none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    config["systems"] = {name: system for name, system in config["systems"].items() if system}
    classifier = E.clinic_classifier(config)
    ev, run = config["evidence"], dict(load_main_config(ROOT / config["run_config"], ROOT), output_dir=config["cache_dir"])
    builder = ClinicEvidenceBuilder(E.chat(config, ev["generator"], "evidence_generator"), E.evidence_validator(config),
                                    generator_prompt=(ROOT / ev["generator_prompt"]).read_text(encoding="utf-8"),
                                    validator_prompt=(ROOT / ev["validator_prompt"]).read_text(encoding="utf-8"),
                                    prompt_version=ev["prompt_version"], params=ev["params"], max_attempts=int(ev.get("max_attempts", 1)),
                                    retry_params=ev.get("retry_params"), validator_prompt_version=ev.get("validator_prompt_version"),
                                    pair_prompt=(ROOT / ev["pair_validator_prompt"]).read_text(encoding="utf-8") if ev.get("pair_validator_prompt") else None)
    judge = build_judge(run, ROOT, read_config(ROOT / run["environment"]["config"]))
    items = controlled_items(read_jsonl(ROOT / args.reference), classifier.cases, builder, judge)
    E.write_json(out / "rq2_controlled_items.prepared.json", [{k: v for k, v in item.items() if k != "trajectory"} | {"case_id": item["trajectory"]["case_id"]} for item in items])
    summary = {}
    for condition in CONDITIONS:
        members = [i for i in items if i["condition_id"] == condition]
        accepted = [i for i in members if i["validation_status"] == "accepted"]
        summary[condition] = {
            "attempted": len(members), "accepted": len(accepted), "rejected": len(members) - len(accepted),
            "accepted_at_attempt": dict(Counter(i["evidence"]["attempt"] for i in accepted)),
            "rejection_reasons": [{"case_id": i["trajectory"]["case_id"], "attempts": [a["reason"] for a in i["evidence"]["attempts"]],
                                   "final_reason": i["evidence"]["reason"]} for i in members if i["validation_status"] != "accepted"],
            "representative_accepted": [{"case_id": i["trajectory"]["case_id"], "claimed_diagnosis": i["claimed_diagnosis"],
                                         "evidence": i["evidence"]["evidence"]} for i in accepted[:3]]}
    summary["roles"] = {"generator": ev["generator"].get("model_id"), "validator": ev["validator"].get("model_id"), "max_attempts": ev.get("max_attempts", 1)}
    E.write_json(out / "rq2_controlled_validation_audit.json", summary)
    by_case = {}
    for item in items:
        if item["condition_id"] in ("correct_strong", "correct_weak") and item["validation_status"] == "accepted":
            by_case.setdefault(item["trajectory"]["case_id"], {})[item["condition_id"]] = item
    with (out / "rq2_controlled_pairs_for_manual_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "correct_diagnosis", "doctor_patient_information", "strong_statement", "weak_statement",
                         "validator_strong", "validator_weak", "human_both_support(y/n)", "human_strong_more_discriminative(y/n)",
                         "human_weak_is_not_hedged_strong(y/n)", "human_keep(y/n)", "human_note"])
        for case_id, pair in sorted(by_case.items()):
            if set(pair) != {"correct_strong", "correct_weak"}:
                continue
            case = classifier.cases[case_id]
            writer.writerow([case_id, case["gold"], render(doctor_view(case)), pair["correct_strong"]["evidence"]["evidence"],
                             pair["correct_weak"]["evidence"]["evidence"], json.dumps(pair["correct_strong"]["evidence"].get("validation")),
                             json.dumps(pair["correct_weak"]["evidence"].get("validation")), "", "", "", "", ""])
    pairs = sum(set(p) == {"correct_strong", "correct_weak"} for p in by_case.values())
    print(json.dumps({"items": len(items), "retained_strong_weak_pairs": pairs,
                      **{c: {k: summary[c][k] for k in ("attempted", "accepted", "rejected", "accepted_at_attempt")} for c in CONDITIONS}}, indent=2))


if __name__ == "__main__":
    main()
