"""Human adjudication of diagnosis equivalence, and the immutable mapping made from it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from src.agentclinic.outcome import normalize
from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
QUALIFIER = r"\b(possible|possibly|probable|probably|likely|suspected|suspicion|suspicious|presumed|presumptive|suggestive|concern|concerning|consistent with)\b"
NON_COMMITTAL = r"\b(rule out|r o|ruling out|versus|vs|differential|pending|or|consider|consideration|cannot exclude|exclude)\b"
GENERIC = {"syndrome", "disease", "disorder", "of", "the", "with", "and", "acute", "chronic", "type"}


def names_reference(reference: str, predicted: str) -> bool:
    words = [w for w in reference.split() if w not in GENERIC]
    return bool(words) and all(w in predicted.split() for w in words)


def sheet(config: dict, out: Path) -> dict:
    base = ROOT / config["output_dir"]
    decisions = read_jsonl(ROOT / config["diagnosis_equivalence_frozen"])
    use = Counter()
    for name in ("before_rollouts.jsonl", "after_rollouts.jsonl"):
        for r in read_jsonl(ROOT / config["labels_dir"] / name, missing_ok=True):
            use[(normalize(r["gold"]), normalize(r["final_diagnosis"] or ""), "label_continuations")] += 1
    for t in read_jsonl(base / "base_trajectories.jsonl"):
        use[(normalize(t["ground_truth"]["correct_diagnosis"]), normalize(t["final_diagnosis"] or ""), f"base_{t['split']}")] += 1
    rows = []
    for d in decisions:
        named = names_reference(d["reference"], d["predicted"])
        qualified, hedged = bool(re.search(QUALIFIER, d["predicted"])), bool(re.search(NON_COMMITTAL, d["predicted"]))
        conflicted = str(d.get("source", "")).startswith("majority")
        if not (conflicted or (not d["equivalent"] and named) or (d["equivalent"] and hedged)):
            continue
        if named and qualified and not hedged:
            suggestion = "equivalent (principle 1: the diagnosis is stated with lowered confidence)"
        elif hedged:
            suggestion = "check commitment (principle 2: a differential or 'rule out' is not a commitment)"
        elif named:
            suggestion = "names the correct diagnosis: check whether it is the final diagnosis or a complication / part of another one"
        else:
            suggestion = "no suggestion"
        rows.append({"correct_diagnosis": d["raw_reference"], "predicted_diagnosis": d["raw_predicted"], "automatic_decision": "y" if d["equivalent"] else "n",
                     "automatic_source": d.get("source"), "reason_listed": "conflicted" if conflicted else "not equivalent but names the diagnosis" if not d["equivalent"] else "equivalent but non-committal wording",
                     "label_continuations": use[(d["reference"], d["predicted"], "label_continuations")],
                     "base_train": use[(d["reference"], d["predicted"], "base_train")], "base_dev": use[(d["reference"], d["predicted"], "base_dev")],
                     "rule_suggestion": suggestion, "human_decision(equivalent y/n)": "", "human_note": ""})
    rows.sort(key=lambda r: (-r["label_continuations"], r["correct_diagnosis"], r["predicted_diagnosis"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "adjudication_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"frozen_decisions": len(decisions), "rows_for_a_human": len(rows), "by_reason": dict(Counter(r["reason_listed"] for r in rows)),
               "by_suggestion": dict(Counter(r["rule_suggestion"].split(" (")[0] for r in rows)),
               "label_continuations_on_these_pairs": sum(r["label_continuations"] for r in rows),
               "base_collaborations_on_these_pairs": {"train": sum(r["base_train"] for r in rows), "dev": sum(r["base_dev"] for r in rows)},
               "frozen_file_sha256": hashlib.sha256((ROOT / config["diagnosis_equivalence_frozen"]).read_bytes()).hexdigest(),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "adjudication_sheet_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def finalize(sheet_path: Path, frozen: Path, target: Path) -> dict:
    if target.exists():
        raise SystemExit(f"{target} exists: a mapping is written once")
    human = {}
    with sheet_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            decision = row["human_decision(equivalent y/n)"].strip().lower()
            if decision not in ("y", "n"):
                raise SystemExit(f"no human decision for: {row['correct_diagnosis']} <> {row['predicted_diagnosis']} (every row needs y or n; no model fills it)")
            human[(normalize(row["correct_diagnosis"]), normalize(row["predicted_diagnosis"]))] = decision == "y"
    # the mapping holds the human decisions only: every other pair is decided by the frozen automatic judge of the
    # environment that uses the mapping, not by the judge whose decisions were being audited
    automatic = {(d["reference"], d["predicted"]): d for d in read_jsonl(frozen)}
    rows, changed = [], 0
    for key, decision in sorted(human.items()):
        before = automatic.get(key) or {}
        changed += before.get("equivalent") is not None and before["equivalent"] != decision
        rows.append({"reference": key[0], "predicted": key[1], "equivalent": decision, "source": "human",
                     "raw_reference": before.get("raw_reference"), "raw_predicted": before.get("raw_predicted"),
                     "automatic_decision_before": before.get("equivalent"), "automatic_source_before": before.get("source")})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    target.chmod(0o444)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    summary = {"mapping": str(target), "sha256": digest, "pairs": len(rows), "human_decisions": len(human), "decisions_changed_by_a_human": changed,
               "sheet": str(sheet_path), "sheet_sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    target.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sheet", "finalize"))
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--sheet")
    parser.add_argument("--frozen")
    parser.add_argument("--target")
    args = parser.parse_args()
    if args.command == "sheet":
        result = sheet(read_config(args.config), ROOT / args.output_dir)
    else:
        result = finalize(ROOT / args.sheet, ROOT / args.frozen, ROOT / args.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
