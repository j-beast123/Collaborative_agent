"""Audit sheet of the frozen diagnosis-equivalence decisions before TEST is opened."""

from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.config import read_config
from src.common.io import read_jsonl
from src.common.main_config import load_main_config
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--main-config", default="configs/serl_agentclinic_qwen4b_max7_train30.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = read_config(args.config)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    decisions = read_jsonl(ROOT / config["diagnosis_equivalence_frozen"])
    conflicted = [d for d in decisions if str(d.get("source", "")).startswith("majority")]
    others = [d for d in decisions if d not in conflicted]
    rng = random.Random(args.seed)
    sample = []
    for value in (True, False):
        pool = sorted((d for d in others if d["equivalent"] is value), key=lambda d: (d["reference"], d["predicted"]))
        sample += rng.sample(pool, min(args.sample // 2, len(pool)))
    auditor = load_main_config(ROOT / args.main_config, ROOT)["roles"]["rubric_manager"]
    chat = CachedChat(build_chat_backend(auditor), out / "cache_auditor.jsonl")
    prompt = (ROOT / config["prompts"]["diagnosis_equivalence"]).read_text(encoding="utf-8")

    def audit(row: dict) -> dict:
        user = json.dumps({"reference_diagnosis": row["raw_reference"], "predicted_diagnosis": row["raw_predicted"]}, ensure_ascii=False)
        result = chat.complete(ChatRequest(prompt, user, "equivalence-audit-v1", {"temperature": 0, "max_tokens": 200}))
        parsed, _ = parse_json_object(result.text) if result.status == "ok" else (None, None)
        return {"group": "conflicted" if row in conflicted else "sample", "reference": row["raw_reference"], "predicted": row["raw_predicted"],
                "frozen_decision": row["equivalent"], "frozen_source": row.get("source"), "votes": row.get("votes"), "observed": row.get("observed"),
                "auditor_decision": (parsed or {}).get("equivalent"), "auditor_reason": (parsed or {}).get("reason")}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(audit, conflicted + sample))
    with (out / "equivalence_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "correct_diagnosis", "predicted_diagnosis", "frozen_decision", "frozen_source", "votes", "observed_before_freeze",
                         "auditor_decision", "auditor_reason", "agree", "human_decision(equivalent y/n)", "human_note"])
        for r in rows:
            writer.writerow([r["group"], r["reference"], r["predicted"], r["frozen_decision"], r["frozen_source"], json.dumps(r["votes"]),
                             json.dumps(r["observed"]), r["auditor_decision"], r["auditor_reason"], r["auditor_decision"] == r["frozen_decision"], "", ""])
    summary = {"frozen_decisions": len(decisions), "auditor": auditor["model_id"], "judge": config["models"]["judge"]["model_id"]}
    for group in ("conflicted", "sample"):
        members = [r for r in rows if r["group"] == group and isinstance(r["auditor_decision"], bool)]
        summary[group] = {"pairs": len(members), "agree": sum(r["auditor_decision"] == r["frozen_decision"] for r in members),
                          "frozen_true_auditor_false": sum(r["frozen_decision"] and not r["auditor_decision"] for r in members),
                          "frozen_false_auditor_true": sum(not r["frozen_decision"] and r["auditor_decision"] for r in members)}
    summary["disagreements"] = [{k: r[k] for k in ("group", "reference", "predicted", "frozen_decision", "auditor_decision", "auditor_reason")}
                                for r in rows if isinstance(r["auditor_decision"], bool) and r["auditor_decision"] != r["frozen_decision"]]
    (out / "equivalence_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, indent=2))


if __name__ == "__main__":
    main()
