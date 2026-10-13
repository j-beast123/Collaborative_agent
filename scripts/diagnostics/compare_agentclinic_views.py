"""Compare two prepare runs on the same cases (e.g. two Doctor views): who is right before A1 and
what the natural collaboration changes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.common.io import read_jsonl
from src.evaluation.collaboration_states import STATES


def summary(rows: list[dict]) -> dict:
    n = len(rows)

    def count(test) -> int:
        return sum(bool(test(r)) for r in rows)

    def recovered(state: str) -> str:
        return f"{count(lambda r: r['collaboration_state'] == state and r['natural_final_correct'])}/{count(lambda r: r['collaboration_state'] == state)}"

    d1, final = count(lambda r: r["d1_correct"]), count(lambda r: r["natural_final_correct"])
    out = {"cases": n, "Doctor D1 correct": f"{d1} ({d1 / n:.0%})",
           "Agent standalone correct": f"{count(lambda r: r['agent_standalone_correct'])}",
           "natural final correct": f"{final} ({final / n:.0%})", "natural final - D1": f"{final - d1:+d}",
           "gained by collaboration (D1 wrong -> final right)": count(lambda r: not r["d1_correct"] and r["natural_final_correct"]),
           "lost by collaboration (D1 right -> final wrong)": count(lambda r: r["d1_correct"] and not r["natural_final_correct"]),
           "D1-Agent disagreement": f"{count(lambda r: r['d1_agent_agree'] is False)} ({count(lambda r: r['d1_agent_agree'] is False) / n:.0%})"}
    for state in STATES.values():
        k = count(lambda r: r["collaboration_state"] == state)
        out[state] = f"{k} ({k / n:.0%})"
    for state in STATES.values():
        out[f"{state}: natural final correct"] = recovered(state)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--output")
    a = p.parse_args()
    rows = {name: {r["case_id"]: r for r in read_jsonl(Path(name) / "prepare_diagnostics.jsonl")} for name in (a.run_a, a.run_b)}
    shared = sorted(set(rows[a.run_a]) & set(rows[a.run_b]))
    first, second = ([rows[name][c] for c in shared] for name in (a.run_a, a.run_b))
    sa, sb = summary(first), summary(second)
    lines = [f"# Doctor-view comparison on {len(shared)} shared cases", "",
             f"- A: {a.run_a}", f"- B: {a.run_b}", "", "| | A | B |", "|---|---|---|"]
    lines += [f"| {key} | {sa[key]} | {sb[key]} |" for key in sa]
    moves = Counter((x["collaboration_state"], y["collaboration_state"]) for x, y in zip(first, second))
    lines += ["", "## State transitions A -> B", "| A | B | cases |", "|---|---|---|"]
    lines += [f"| {x} | {y} | {n} |" for (x, y), n in sorted(moves.items(), key=lambda item: -item[1])]
    same_agent = sum(x["agent_standalone_diagnosis"] == y["agent_standalone_diagnosis"] for x, y in zip(first, second))
    lines += ["", f"Agent standalone assessment identical in {same_agent} of {len(shared)} cases (shared when the runs use one standalone file)."]
    text = "\n".join(lines) + "\n"
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
