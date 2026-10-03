"""Is there an effectiveness signal to learn? Diagnostics of a labelled run (e.g. the pilot)."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.common.io import read_jsonl, write_jsonl
from src.judges.llm import ChatRequest, openrouter_chat
from src.judges.option_extraction import openrouter_extractor

ROOT = Path(__file__).resolve().parents[1]
MAINTAIN_PAIR = ("evidence_grounded_maintain", "unsupported_maintain")


def group_of(response_id: str) -> str:
    return "natural" if response_id == "original" or response_id.startswith("sample_") else response_id


def share(values) -> float | None:
    values = list(values)
    return sum(bool(v) for v in values) / len(values) if values else None


def spread(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "max": max(values), "share_nonzero": share(v != 0 for v in values),
            "counts": dict(sorted(Counter(round(v, 2) for v in values).items()))}


def excerpt(rollout: dict, **extra) -> dict:
    return {"rollout_id": rollout["rollout_id"], "history_id": rollout["history_id"],
            **extra,
            "dialogue": [{k: t.get(k) for k in ("speaker", "round", "text", "intervention", "final") if t.get(k)}
                         for t in rollout["dialogue"]]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--judge-model", required=True, help="OpenRouter model for the option extractor and evidence check")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--recommendation-file")
    p.add_argument("--max-examples", type=int, default=20)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = read_jsonl(a.labels)
    candidates = read_jsonl(a.candidates)
    by_pair = {(r["history_id"], r["response_id"]): r for r in labels}
    by_history = defaultdict(list)
    for row in labels:
        by_history[row["history_id"]].append(row)
    rollouts = [r for r in read_jsonl(a.labels + ".rollouts.jsonl") if r["status"] == "completed"]
    k = labels[0]["after"]["trials"] if labels else None

    # A, B, F
    deltas = [r["delta_effectiveness"] for r in labels]
    spreads = [max(r["after_rate"] for r in rows) - min(r["after_rate"] for r in rows) for rows in by_history.values()]
    saturation = defaultdict(list)
    for rows in by_history.values():
        before = rows[0]["before_rate"]
        level = "before = 0" if before == 0 else "before = 1" if before == 1 else "0 < before < 1"
        saturation[level].append(rows)
    by_behavior = defaultdict(list)
    for row in labels:
        by_behavior[group_of(row["response_id"])].append(row["delta_effectiveness"])

    # C: same option, different behavior
    pair_diffs = [by_pair[(h, MAINTAIN_PAIR[0])]["after_rate"] - by_pair[(h, MAINTAIN_PAIR[1])]["after_rate"]
                  for h in by_history if all((h, name) in by_pair for name in MAINTAIN_PAIR)]

    # compliance of the controlled generations (all of them, labelled or not)
    compliance = defaultdict(Counter)
    for c in candidates:
        if c.get("intended_behavior"):
            compliance[c["intended_behavior"]][str(c.get("compliance"))] += 1
            compliance[c["intended_behavior"]]["observed:" + str(c.get("observed_behavior"))] += 1

    # D, E: read the rollouts
    positions = {c["history_id"]: c["meta"] for c in candidates if (c.get("meta") or {}).get("doctor_claim")}
    extractor = openrouter_extractor(a.judge_model, out / "signal_option_cache.jsonl", ROOT)
    evidence_chat = openrouter_chat(a.judge_model, out / "signal_evidence_cache.jsonl")
    evidence_prompt = (ROOT / "prompts/evidence_check.txt").read_text(encoding="utf-8")

    def new_evidence(case: dict, dialogue: list, index: int) -> bool | None:
        earlier = "\n\n".join(f"{t['speaker'].title()}: {t['text']}" for t in dialogue[:index])
        user = json.dumps({"question": case["question"], "options": case["options"], "discussion_so_far": earlier,
                           "new_clinician_message": dialogue[index]["text"]}, ensure_ascii=False)
        result = evidence_chat.complete(ChatRequest(evidence_prompt, user, "evidence-check-v1",
                                                    {"temperature": 0, "max_tokens": 3}))
        reply = (result.text or "").strip().upper()
        return None if result.status != "ok" or reply[:3] not in ("YES", "NO") else reply.startswith("YES")

    def read_rollout(rollout: dict) -> dict | None:
        after = ":after:" in rollout["state_id"]
        response_id = rollout["state_id"].rsplit(":after:", 1)[1] if after else None
        label = by_pair.get((rollout["history_id"], response_id)) if after \
            else next(iter(by_history.get(rollout["history_id"], [])), None)
        if label is None:
            return None
        meta = positions.get(rollout["history_id"], {})
        dialogue = rollout["dialogue"]
        found = {"rollout": rollout, "label": label, "group": group_of(response_id) if response_id else "(before)",
                 "doctor_revised": None, "revised_to_agent_answer": None, "reversal": None, "new_evidence": None}
        if meta.get("doctor_claim") and rollout.get("final_option"):
            found["doctor_revised"] = rollout["final_option"] != meta["doctor_claim"]
            found["revised_to_agent_answer"] = found["doctor_revised"] and rollout["final_option"] == meta.get("agent_initial_answer")
        fixed = next((i for i, t in enumerate(dialogue) if t.get("intervention")), None)
        later = next((i for i, t in enumerate(dialogue) if fixed is not None and i > fixed
                      and t["speaker"] == "agent" and not t.get("final")), None)
        if response_id and later is not None:
            case = label["history"]
            before_option = extractor.stated_option(case, dialogue[fixed]["text"])
            after_option = extractor.stated_option(case, dialogue[later]["text"])
            if before_option and after_option:
                found["reversal"] = before_option != after_option
                if found["reversal"]:
                    found["new_evidence"] = new_evidence(case, dialogue, later - 1)
        return found

    with ThreadPoolExecutor(12) as pool:
        read = [r for r in pool.map(read_rollout, rollouts) if r]
    revision, reversal = defaultdict(list), defaultdict(list)
    for r in read:
        if r["doctor_revised"] is not None:
            revision[r["group"]].append(r)
        if r["reversal"] is not None:
            reversal[r["group"]].append(r)
    revision_rates = {g: {"rollouts": len(rs), "doctor_revised": share(r["doctor_revised"] for r in rs),
                          "revised_to_agent_answer": share(r["revised_to_agent_answer"] for r in rs),
                          "success": share(r["rollout"]["correct"] for r in rs)} for g, rs in sorted(revision.items())}
    reversal_rates = {g: {"rollouts": len(rs), "agent_reversed": share(r["reversal"] for r in rs),
                          "reversed_without_new_evidence": share(r["reversal"] and r["new_evidence"] is False for r in rs)}
                      for g, rs in sorted(reversal.items())}
    write_jsonl(out / "doctor_revision_examples.jsonl",
                [excerpt(r["rollout"], group=r["group"], positions=positions.get(r["rollout"]["history_id"]),
                         final_option=r["rollout"]["final_option"], correct=r["rollout"]["correct"])
                 for r in read if r["doctor_revised"]][: a.max_examples])
    write_jsonl(out / "agent_reversal_examples.jsonl",
                [excerpt(r["rollout"], group=r["group"], positions=positions.get(r["rollout"]["history_id"]),
                         new_evidence=r["new_evidence"])
                 for r in read if r["reversal"] and r["new_evidence"] is False][: a.max_examples])

    report = {
        "histories": len(by_history), "pairs": len(labels), "k": k, "rollouts_completed": len(rollouts),
        "candidates_generated": len(candidates),
        "nonzero_delta": {"pairs": share(d != 0 for d in deltas),
                          "histories_with_any": share(any(r["delta_effectiveness"] != 0 for r in rows)
                                                      for rows in by_history.values())},
        "within_history_variation": spread(spreads),
        "delta_by_behavior": {g: {"n": len(v), "mean": statistics.fmean(v), "share_nonzero": share(x != 0 for x in v)}
                              for g, v in sorted(by_behavior.items())},
        "same_answer_behavior_effect": {**spread(pair_diffs), "pair": list(MAINTAIN_PAIR),
                                        "evidence_better": sum(d > 0 for d in pair_diffs),
                                        "equal": sum(d == 0 for d in pair_diffs),
                                        "unsupported_better": sum(d < 0 for d in pair_diffs)},
        "compliance": {name: dict(counter) for name, counter in sorted(compliance.items())},
        "doctor_revision_rate": revision_rates,
        "agent_reversal_rate": reversal_rates,
        "before_saturation": {level: {"histories": len(groups), "share": len(groups) / len(by_history),
                                      "delta": spread([r["delta_effectiveness"] for rows in groups for r in rows])}
                              for level, groups in sorted(saturation.items())},
    }
    (out / "signal_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    recommendation = Path(a.recommendation_file).read_text(encoding="utf-8").strip() if a.recommendation_file \
        else "Not written yet: read the numbers above, then pass --recommendation-file."
    (out / "signal_report.md").write_text(markdown(report, recommendation), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("histories", "pairs", "nonzero_delta", "same_answer_behavior_effect")},
                     indent=2, default=str))


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def markdown(report: dict, recommendation: str) -> str:
    lines = ["# v4 Effectiveness Signal Check", "",
             f"- histories: {report['histories']}",
             f"- candidates: {report['pairs']} labelled pairs ({report['candidates_generated']} generated)",
             f"- K: {report['k']} ({report['rollouts_completed']} completed rollouts)", "",
             "## Non-zero delta",
             f"- pairs with delta != 0: {pct(report['nonzero_delta']['pairs'])}",
             f"- histories with at least one candidate delta != 0: {pct(report['nonzero_delta']['histories_with_any'])}", "",
             "| candidate | pairs | mean delta | delta != 0 |", "|---|---|---|---|"]
    lines += [f"| {g} | {v['n']} | {v['mean']:+.2f} | {pct(v['share_nonzero'])} |" for g, v in report["delta_by_behavior"].items()]
    variation = report["within_history_variation"]
    lines += ["", "## Within-history variation", "max(after) - min(after) per history:",
              f"- mean {variation.get('mean', 0):.2f}, median {variation.get('median', 0):.2f}, "
              f"histories with any spread {pct(variation.get('share_nonzero'))}",
              f"- distribution: {variation.get('counts')}", ""]
    same = report["same_answer_behavior_effect"]
    lines += ["## Same-answer behavior effect", "evidence_grounded_maintain minus unsupported_maintain (AFTER rate, same option):",
              f"- histories with both: {same['n']}"]
    if same["n"]:
        lines += [f"- mean difference {same['mean']:+.2f}; evidence better in {same['evidence_better']}, "
                  f"equal in {same['equal']}, unsupported better in {same['unsupported_better']}"]
    lines += ["", "## Candidate compliance", "| intended | complied | did not | observed |", "|---|---|---|---|"]
    for name, counts in report["compliance"].items():
        observed = ", ".join(f"{key[9:]} {n}" for key, n in counts.items() if key.startswith("observed:"))
        lines.append(f"| {name} | {counts.get('True', 0)} | {counts.get('False', 0)} | {observed} |")
    lines += ["", "## Doctor revision rate", "Final answer differs from the doctor's option before the response:", "",
              "| after | rollouts | doctor revised | revised to the agent's answer | success |", "|---|---|---|---|---|"]
    lines += [f"| {g} | {v['rollouts']} | {pct(v['doctor_revised'])} | {pct(v['revised_to_agent_answer'])} | {pct(v['success'])} |"
              for g, v in report["doctor_revision_rate"].items()]
    lines += ["", "## Agent reversal rate", "The continuation agent's next turn states a different option than the fixed response:", "",
              "| after | rollouts | reversed | reversed without new clinician evidence |", "|---|---|---|---|"]
    lines += [f"| {g} | {v['rollouts']} | {pct(v['agent_reversed'])} | {pct(v['reversed_without_new_evidence'])} |"
              for g, v in report["agent_reversal_rate"].items()]
    lines += ["", "## BEFORE saturation", "| BEFORE | histories | share | mean delta | delta != 0 |", "|---|---|---|---|---|"]
    lines += [f"| {level} | {v['histories']} | {pct(v['share'])} | {v['delta'].get('mean', 0):+.2f} | {pct(v['delta'].get('share_nonzero'))} |"
              for level, v in report["before_saturation"].items()]
    lines += ["", "## Recommendation", recommendation, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
