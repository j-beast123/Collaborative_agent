"""Freeze one diagnosis-equivalence decision per normalized (reference, predicted) pair, and recompute the Effectiveness
labels of an environment with it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.agentclinic.outcome import DiagnosisJudge, normalize
from src.common.config import read_config
from src.common.io import read_jsonl
from src.judges.llm import openrouter_chat

ROOT = Path(__file__).resolve().parents[1]
VOTES = 5


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True, help="new predictor directory; its labels/ is written here")
    parser.add_argument("--also", nargs="*", default=[], help="more trajectory files whose judged outcomes count as observed decisions")
    args = parser.parse_args()
    config = read_config(args.config)
    base, labels, target = ROOT / config["output_dir"], ROOT / config["labels_dir"], ROOT / args.target / "labels"
    frozen_path = base / "diagnosis_equivalence.frozen.jsonl"
    if frozen_path.exists() or target.exists():
        raise SystemExit(f"{frozen_path} or {target} exists: the decisions are frozen once")
    k = config["k"]
    rollouts = {name: read_jsonl(labels / name) for name in ("before_rollouts.jsonl", "after_rollouts.jsonl")}
    observed = defaultdict(Counter)   # normalized pair -> decisions seen
    wording = defaultdict(Counter)    # normalized pair -> raw (gold, predicted) seen
    for rows in rollouts.values():
        for r in rows:
            if r["outcome_method"] == "judge" and r["success"] is not None:
                key = (normalize(r["gold"]), normalize(r["final_diagnosis"]))
                observed[key][bool(r["success"])] += 1
                wording[key][(r["gold"], r["final_diagnosis"])] += 1
    for path in [base / "base_trajectories.jsonl"] + [ROOT / p for p in args.also]:
        for t in read_jsonl(path, missing_ok=True):
            if t.get("outcome_method") == "judge" and t.get("is_correct") is not None and t.get("final_diagnosis"):
                gold = t["ground_truth"]["correct_diagnosis"]
                key = (normalize(gold), normalize(t["final_diagnosis"]))
                observed[key][bool(t["is_correct"])] += 1
                wording[key][(gold, t["final_diagnosis"])] += 1
    prompt = (ROOT / config["prompts"]["diagnosis_equivalence"]).read_text(encoding="utf-8")
    decisions, unstable = [], []
    for key in sorted(observed):
        counts = observed[key]
        gold, predicted = wording[key].most_common(1)[0][0]
        if len(counts) == 1:
            decisions.append({"reference": key[0], "predicted": key[1], "equivalent": next(iter(counts)), "source": "observed_unanimous",
                              "observed": {str(d): n for d, n in counts.items()}, "raw_reference": gold, "raw_predicted": predicted})
            continue
        votes = []
        for vote in range(VOTES):  # no cache: each call is a new answer of the same judge and prompt
            judge = DiagnosisJudge(openrouter_chat(config["models"]["judge"]["model_id"], None), prompt)
            votes.append(judge.judge(gold, predicted)["equivalent"])
        valid = [v for v in votes if isinstance(v, bool)]
        if len(valid) < VOTES:
            raise SystemExit(f"the judge did not answer {VOTES} times for {key}: {votes}")
        row = {"reference": key[0], "predicted": key[1], "equivalent": sum(valid) * 2 > len(valid), "source": f"majority_of_{VOTES}_calls",
               "votes": votes, "observed": {str(d): n for d, n in counts.items()}, "raw_reference": gold, "raw_predicted": predicted}
        decisions.append(row)
        unstable.append(row)
    frozen = {(d["reference"], d["predicted"]): d["equivalent"] for d in decisions}
    write_rows(frozen_path, decisions)

    # the label directory with outcomes from the frozen decisions
    target.mkdir(parents=True)
    for path in sorted(labels.iterdir()):
        if path.is_file() and path.name not in ("before_rollouts.jsonl", "after_rollouts.jsonl", "labels.jsonl", "labels_train.jsonl",
                                               "labels_dev.jsonl", "integrity_report.json", "split_manifest.json",
                                               "turn_sanity_full.json", "turn_leakage_audit_full.json"):
            shutil.copy2(path, target / path.name)
    split = {t["case_id"]: t["split"] for t in read_jsonl(base / "base_trajectories.jsonl")}
    changed_rows = Counter()
    for name, rows in rollouts.items():
        out = []
        for r in rows:
            new = r
            if r["outcome_method"] == "judge" and r["success"] is not None:
                decision = frozen[(normalize(r["gold"]), normalize(r["final_diagnosis"]))]
                if decision != bool(r["success"]):
                    changed_rows[(name, split[r["case_id"]])] += 1
                new = dict(r, success=decision, outcome_method="frozen", success_before_freeze=r["success"],
                           outcome_reason="frozen decision of this normalized diagnosis pair")
            out.append(new)
        rollouts[name] = out
        write_rows(target / name, out)
    before = defaultdict(list)
    for r in rollouts["before_rollouts.jsonl"]:
        before[r["history_id"]].append(r)
    after = defaultdict(list)
    for r in rollouts["after_rollouts.jsonl"]:
        after[(r["history_id"], r["candidate_id"], r.get("candidate_attempt", 1))].append(r)
    new_labels, changed_labels, changed_delta = [], Counter(), Counter()
    for label in read_jsonl(labels / "labels.jsonl"):
        b = [r for r in before[label["history_id"]] if r["status"] == "completed"]
        a = [r for r in after[(label["history_id"], label["candidate_id"], label.get("candidate_attempt", 1))] if r["status"] == "completed"]
        if len(a) != k or len(b) != k:
            raise SystemExit(f"{label['history_id']}/{label['candidate_id']}: not {k} continuations on both sides")
        counts = {"before": {"successes": sum(r["success"] for r in b), "trials": k}, "after": {"successes": sum(r["success"] for r in a), "trials": k}}
        rates = {"before_rate": counts["before"]["successes"] / k, "after_rate": counts["after"]["successes"] / k}
        row = dict(label, **counts, **rates, delta_effectiveness=rates["after_rate"] - rates["before_rate"])
        if (row["before"], row["after"]) != (label["before"], label["after"]):
            changed_labels[label["split"]] += 1
        if row["delta_effectiveness"] != label["delta_effectiveness"]:
            changed_delta[label["split"]] += 1
        new_labels.append(row)
    write_rows(target / "labels.jsonl", new_labels)
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "frozen_decisions": str(frozen_path.relative_to(ROOT)), "pairs": len(decisions),
              "unstable_pairs": len(unstable), "unstable": unstable, "votes_per_unstable_pair": VOTES,
              "continuations": sum(len(v) for v in rollouts.values()),
              "continuations_whose_outcome_changed": {f"{name}:{s}": n for (name, s), n in sorted(changed_rows.items())},
              "labels": len(new_labels), "labels_whose_counts_changed": dict(changed_labels), "labels_whose_delta_changed": dict(changed_delta),
              "source_labels_dir": config["labels_dir"], "target_labels_dir": str(target.relative_to(ROOT)),
              "nonzero_delta_pairs_before": dict(Counter(l["split"] for l in read_jsonl(labels / "labels.jsonl") if l["delta_effectiveness"] != 0)),
              "nonzero_delta_pairs_after": dict(Counter(l["split"] for l in new_labels if l["delta_effectiveness"] != 0))}
    (target / "equivalence_freeze_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "unstable"}, indent=2))
    for row in unstable:
        print(row["reference"], "|", row["predicted"], "| observed", row["observed"], "| votes", row["votes"], "->", row["equivalent"])


if __name__ == "__main__":
    main()
