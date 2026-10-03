"""Per iteration of a run: how many rubrics every stage of the rubric method left, and how the kept ones were supported."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def iteration_row(directory: Path) -> dict | None:
    rubrics = directory / "rubrics"
    if not (rubrics / "summary.json").is_file():
        return None
    summary = json.loads((rubrics / "summary.json").read_text(encoding="utf-8"))
    row = {"iteration": int(directory.name[4:]), "rubric_schedule": summary.get("rubric_schedule"), "selection_method": summary.get("selection_method"),
           "analysed_failures": (summary.get("selection") or {}).get("failed"), "generated_candidates": summary.get("raw_candidates"),
           "passed_causal_validation": summary.get("validated_candidates"), "consolidated": summary.get("consolidated_categories"),
           "kept": summary.get("retained_rubrics"), "removed_by_reason": summary.get("removed"), "rubric_set_version": summary.get("rubric_set_version"),
           "initial_rubrics_source": json.loads((rubrics / "initial_rubrics_source.json").read_text())["source"] if (rubrics / "initial_rubrics_source.json").is_file() else None}
    if (rubrics / "rubric_table.json").is_file():
        table = [r for r in json.loads((rubrics / "rubric_table.json").read_text(encoding="utf-8"))["rows"] if r["level"] == "category"]
        row["positive_gap"] = sum((r.get("effectiveness_gap") or 0) > 0 for r in table)
        row["rubrics"] = [{"category": r["category"], "rubric": r["rubric"], "source_failures": len(r.get("source_failures") or []),
                           "validation_success_rate": r.get("validation_success_rate"), "applicable_n": r.get("applicable_n"),
                           "score_0_n": r.get("low_n"), "score_1_n": r.get("partial_n"), "score_2_n": r.get("high_n"),
                           "effectiveness_gap": r.get("effectiveness_gap"), "weight": r.get("final_weight"), "kept": r.get("retained"),
                           "removal_reason": r.get("removal_reason")} for r in table]
    if (rubrics / "rubric_set.json").is_file() and (rubrics / "gap_scores.jsonl").is_file():
        rubric_set = json.loads((rubrics / "rubric_set.json").read_text(encoding="utf-8"))
        kept = {r["rubric_id"] for r in rubric_set["rubrics"]}
        scores = read_jsonl(rubrics / "gap_scores.jsonl")
        applicable = sum(any(label not in (None, "NA") for rid, label in (s.get("labels") or {}).items() if rid in kept) for s in scores)
        row.update(selection=rubric_set.get("selection"), gap_population_responses=len(scores),
                   gap_population_with_an_applicable_kept_rubric=applicable,
                   gap_population_applicability=applicable / len(scores) if scores else None)
    log = directory / "reward_log.jsonl"
    if log.is_file():
        rows = [r for r in read_jsonl(log) if r.get("status") == "completed"]
        per_rubric: dict[str, Counter] = {}
        with_rubric = 0
        for r in rows:
            any_applicable = False
            for stage in (r.get("rubric_scores") or {}).values():
                for rid, label in (stage.get("labels") or {}).items():
                    per_rubric.setdefault(rid, Counter())[str(label)] += 1
                    any_applicable |= label not in (None, "NA")
            with_rubric += any_applicable
        row["training"] = {"sampled_responses": len(rows), "with_an_applicable_rubric": with_rubric,
                           "applicability": with_rubric / len(rows) if rows else None,
                           "scores_by_rubric": {rid: dict(counts) for rid, counts in per_rubric.items()}}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run = ROOT / args.run
    rows = [row for directory in sorted(run.glob("iter*"), key=lambda p: int(p.name[4:])) if (row := iteration_row(directory))]
    text = json.dumps({"run": args.run, "iterations": rows}, indent=2, ensure_ascii=False)
    if args.output:
        (ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
