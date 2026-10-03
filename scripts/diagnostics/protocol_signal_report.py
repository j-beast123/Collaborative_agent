"""Signal check of a protocol's base collaborations (an environment's base_trajectories.jsonl): Doctor final success,
the Agent's A1 correctness (its first stated assessment, the run's diagnosis-equivalence judge), the A1 x final table,
dialogue length and early-stop rate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.agentclinic.runtime import build_judge
from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    source = read_config(ROOT / args.env)
    out = ROOT / source["output_dir"]
    judge = build_judge({"output_dir": source["output_dir"]}, ROOT, source)
    rows = read_jsonl(out / "base_trajectories.jsonl")
    table = []
    for t in rows:
        a1 = next((m for m in t["turns"] if m["speaker"] == "agent"), {})
        gold = t["ground_truth"]["correct_diagnosis"]
        ok = bool(a1.get("assessment")) and bool(judge.equivalent(gold, a1["assessment"])["equivalent"])
        table.append({"case_id": t["case_id"], "gold": gold, "a1": a1.get("assessment"), "a1_correct": ok, "final": t["final_diagnosis"],
                      "final_correct": bool(t["is_correct"]), "turns": t["num_turns"], "stop": t.get("stop")})
    n = len(table)
    cells = Counter((r["a1_correct"], r["final_correct"]) for r in table)
    summary = {"protocol": rows[0]["protocol"] if rows else None, "cases": n, "final_success": sum(r["final_correct"] for r in table),
               "a1_correct": sum(r["a1_correct"] for r in table),
               "table": {"a1_correct": {"final_correct": cells[(True, True)], "final_wrong": cells[(True, False)]},
                         "a1_wrong": {"final_correct": cells[(False, True)], "final_wrong": cells[(False, False)]}},
               "failures": n - sum(r["final_correct"] for r in table),
               "mean_agent_turns": round(sum(r["turns"] for r in table) / n, 2) if n else None,
               "early_stop_rate": round(sum(r["stop"] == "early" for r in table) / n, 3) if n else None,
               "turns": dict(Counter(r["turns"] for r in table))}
    if args.compare:
        summary["compare"] = json.loads((ROOT / args.compare).read_text(encoding="utf-8"))
    (out / "signal_check.json").write_text(json.dumps({"summary": summary, "cases": table}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "compare"}, ensure_ascii=False, indent=1))
    for r in table:
        print(r["case_id"][-4:], "A1", "Y" if r["a1_correct"] else "N", "final", "Y" if r["final_correct"] else "N", r["turns"], "|", r["gold"], "| A1:", r["a1"], "| final:", r["final"])


if __name__ == "__main__":
    main()
