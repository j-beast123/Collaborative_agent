"""Signal report of a v5 labelling run: does changing A1 change the Doctor's downstream success?"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from src.common.io import read_jsonl, write_jsonl

from src.evaluation.collaboration_states import STATES

from .candidates import CANDIDATES
from .outcome import DiagnosisJudge

MAINTAIN = ("evidence_grounded_maintain", "unsupported_maintain")


def share(values) -> float | None:
    values = list(values)
    return sum(bool(v) for v in values) / len(values) if values else None


def mean(values) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def distribution(values: list[float]) -> dict:
    return dict(sorted(Counter(round(v, 2) for v in values).items()))


def build_report(out: Path, dataset: dict, judge: DiagnosisJudge, new_evidence: Callable[[list, int], bool | None],
                 max_examples: int = 20) -> dict:
    histories = {r["history_id"]: r for r in read_jsonl(out / "histories.jsonl")}
    candidates = {(r["history_id"], r["candidate_id"]): r for r in read_jsonl(out / "candidates.jsonl", missing_ok=True)}
    compliance = read_jsonl(out / "candidate_compliance.jsonl", missing_ok=True)
    labels = read_jsonl(out / "labels.jsonl", missing_ok=True)
    before = [r for r in read_jsonl(out / "before_rollouts.jsonl", missing_ok=True) if r["status"] == "completed"]
    after = [r for r in read_jsonl(out / "after_rollouts.jsonl", missing_ok=True) if r["status"] == "completed"]
    after = [r for r in after if (r["history_id"], r["candidate_id"]) in candidates  # continuations of the chosen generation
             and r.get("candidate_attempt", 1) == candidates[(r["history_id"], r["candidate_id"])].get("attempt", 1)]
    by_history = defaultdict(dict)
    for row in labels:
        by_history[row["history_id"]][row["response_id"]] = row
    k = labels[0]["after"]["trials"] if labels else None

    def same(a: str, b: str) -> bool | None:
        return judge.equivalent(a, b)["equivalent"] if a and b else None

    # who is right before the intervention (evaluator-side reading; never used to generate anything)
    agent_right = {h: same(r["gold"], r["agent_standalone_diagnosis"]) for h, r in histories.items()}
    doctor_right = {h: same(r["gold"], r["doctor_provisional_diagnosis"]) for h, r in histories.items()}

    # compliance
    by_candidate = defaultdict(list)
    for row in compliance:
        by_candidate[row["candidate_id"]].append(row)
    compliance_table = {name: {
        "generated": len(rows), "compliant": sum(bool(r["behavior_compliant"]) for r in rows),
        "observed": dict(Counter(str(r["observed_behavior"]) for r in rows)),
        "uses_private_evidence": share(r["uses_private_evidence"] for r in rows),
        "fabricated_evidence": share(r["fabricated_evidence"] for r in rows),
        "attempts": dict(Counter(r.get("attempt", 1) for r in rows)),
    } for name, rows in sorted(by_candidate.items())}

    # effectiveness signal
    deltas = [r["delta_effectiveness"] for r in labels]
    ranges = [max(r["after_rate"] for r in rows.values()) - min(r["after_rate"] for r in rows.values())
              for rows in by_history.values() if rows]
    before_rates = [next(iter(rows.values()))["before_rate"] for rows in by_history.values() if rows]
    after_by_candidate = defaultdict(list)
    for row in labels:
        after_by_candidate[row["response_id"]].append(row)

    def paired(a: str, b: str, keep=lambda h: True) -> dict:
        diffs = [rows[a]["after_rate"] - rows[b]["after_rate"] for h, rows in by_history.items()
                 if a in rows and b in rows and keep(h)]
        return {"histories": len(diffs), "mean_difference": mean(diffs), "first_better": sum(d > 0 for d in diffs),
                "equal": sum(d == 0 for d in diffs), "second_better": sum(d < 0 for d in diffs)}

    # doctor responsiveness and A2 reversal, read from the AFTER rollouts
    def read(rollout: dict) -> dict:
        history = histories[rollout["history_id"]]
        dialogue = rollout["dialogue"]
        stage = {m["stage"]: m for m in dialogue}
        candidate = candidates[(rollout["history_id"], rollout["candidate_id"])]
        d1 = history["doctor_provisional_diagnosis"]
        revised = None if not d1 or not rollout["final_diagnosis"] else not same(d1, rollout["final_diagnosis"])
        a1, a2 = stage["A1"].get("assessment"), stage["A2"].get("assessment")
        reversed_stance = None if not a1 or not a2 else not same(a1, a2)
        d2_index = next(i for i, m in enumerate(dialogue) if m["stage"] == "D2")
        return {"rollout": rollout, "candidate": rollout["candidate_id"], "doctor_revised": revised,
                "toward_agent": bool(revised) and bool(candidate.get("assessment")) and same(candidate["assessment"], rollout["final_diagnosis"]),
                "a2_reversed": reversed_stance,
                "d2_new_evidence": new_evidence(dialogue, d2_index) if reversed_stance else None}

    with ThreadPoolExecutor(12) as pool:
        read_rows = list(pool.map(read, after))
    private = {(c["history_id"], c["candidate_id"]): bool(c["uses_private_evidence"]) for c in compliance}
    responsiveness, reversal = {}, {}
    for name in CANDIDATES:
        rows = [r for r in read_rows if r["candidate"] == name]
        known = [r for r in rows if r["doctor_revised"] is not None]
        responsiveness[name] = {
            "rollouts": len(known), "doctor_revised": share(r["doctor_revised"] for r in known),
            "revised_toward_agent": share(r["toward_agent"] for r in known),
            "agent_communicated_private_evidence": share(private.get((r["rollout"]["history_id"], name)) for r in known),
            "success": share(r["rollout"]["success"] for r in rows)}
        stance = [r for r in rows if r["a2_reversed"] is not None]
        reversal[name] = {
            "rollouts": len(stance), "a2_reversed": share(r["a2_reversed"] for r in stance),
            "reversed_without_new_evidence": share(r["a2_reversed"] and r["d2_new_evidence"] is False for r in stance),
            "reversed_with_new_evidence": share(r["a2_reversed"] and r["d2_new_evidence"] is True for r in stance)}

    def excerpt(r: dict, **extra) -> dict:
        ro = r["rollout"]
        return {"history_id": ro["history_id"], "candidate_id": ro["candidate_id"], "rollout_index": ro["rollout_index"],
                "gold": ro["gold"], "final_diagnosis": ro["final_diagnosis"], "success": ro["success"], **extra,
                "dialogue": [{"stage": m["stage"], "text": m["text"]} for m in ro["dialogue"]]}

    write_jsonl(out / "doctor_revision_examples.jsonl",
                [excerpt(r, toward_agent=r["toward_agent"]) for r in read_rows if r["doctor_revised"]][:max_examples])
    write_jsonl(out / "agent_reversal_examples.jsonl",
                [excerpt(r, d2_new_evidence=r["d2_new_evidence"]) for r in read_rows if r["a2_reversed"]][:max_examples])

    def direction(flag: dict, value: bool) -> dict:
        keep = lambda h: flag.get(h) is value  # noqa: E731
        hs = [h for h in by_history if keep(h)]
        return {"histories": len(hs),
                "before": mean(next(iter(by_history[h].values()))["before_rate"] for h in hs if by_history[h]),
                "after": {name: mean(by_history[h][name]["after_rate"] for h in hs if name in by_history[h])
                          for name in CANDIDATES},
                "evidence_vs_unsupported": paired(*MAINTAIN, keep), "evidence_vs_accept": paired(MAINTAIN[0], "accept_revise", keep)}

    def state_of(h: str) -> str:
        return STATES.get((agent_right.get(h), doctor_right.get(h)), "undefined")

    by_state = defaultdict(list)
    for h, rows in by_history.items():
        by_state[state_of(h)].append(rows)
    signal_by_state = {state: {
        "histories": len(groups),
        "with_nonzero_delta": sum(any(r["delta_effectiveness"] != 0 for r in rows.values()) for rows in groups),
        "all_candidates_identical": sum(len({r["after_rate"] for r in rows.values()}) == 1 for rows in groups),
        "mean_range": mean(max(r["after_rate"] for r in rows.values()) - min(r["after_rate"] for r in rows.values())
                           for rows in groups),
        "before": mean(next(iter(rows.values()))["before_rate"] for rows in groups),
        "after": {name: mean(rows[name]["after_rate"] for rows in groups if name in rows) for name in CANDIDATES},
    } for state, groups in sorted(by_state.items())}

    report = {
        "dataset": dataset, "histories": len(by_history), "pairs": len(labels), "k": k,
        "before_rollouts": len(before), "after_rollouts": len(after),
        "who_is_right_before_a1": {"agent_standalone_correct": share(v for v in agent_right.values() if v is not None),
                                   "doctor_d1_correct": share(v for v in doctor_right.values() if v is not None)},
        "compliance": compliance_table,
        "outcome_distribution": {
            "before": distribution(before_rates),
            "after": {name: {"mean": mean(r["after_rate"] for r in rows), "distribution": distribution([r["after_rate"] for r in rows])}
                      for name, rows in sorted(after_by_candidate.items())}},
        "signal": {
            "pairs_nonzero_delta": share(d != 0 for d in deltas),
            "histories_with_nonzero_delta": share(any(r["delta_effectiveness"] != 0 for r in rows.values())
                                                  for rows in by_history.values()),
            "within_history_range": {"mean": mean(ranges), "median": statistics.median(ranges) if ranges else None,
                                     "distribution": distribution(ranges)},
            "mean_abs_delta": mean(abs(d) for d in deltas),
            "median_abs_delta": statistics.median([abs(d) for d in deltas]) if deltas else None,
            "histories_all_candidates_identical": sum(r == 0 for r in ranges),
        },
        "by_split": {split: {"histories": len({r["history_id"] for r in labels if r["split"] == split}),
                             "usable_pairs": sum(r["split"] == split for r in labels)}
                     for split in sorted({r["split"] for r in labels})},
        "usable_pairs_by_state": {state: {name: sum(name in rows for rows in groups) for name in CANDIDATES}
                                  for state, groups in sorted(by_state.items())},
        "saturated_histories": {
            "all_zero (BEFORE and every AFTER are 0)": sum(next(iter(rows.values()))["before_rate"] == 0
                                                           and all(r["after_rate"] == 0 for r in rows.values())
                                                           for rows in by_history.values() if rows),
            "all_one (BEFORE and every AFTER are 1)": sum(next(iter(rows.values()))["before_rate"] == 1
                                                          and all(r["after_rate"] == 1 for r in rows.values())
                                                          for rows in by_history.values() if rows)},
        "delta_by_behavior": {name: {"pairs": len(rows), "mean": mean(r["delta_effectiveness"] for r in rows),
                                     "distribution": distribution([r["delta_effectiveness"] for r in rows])}
                              for name, rows in sorted(after_by_candidate.items())},
        "signal_by_state": signal_by_state,
        "behavior_comparisons": {
            "evidence_grounded_vs_unsupported": paired(*MAINTAIN),
            "maintain_vs_accept": paired(MAINTAIN[0], "accept_revise"),
            "evidence_only_vs_evidence_grounded": paired("evidence_only_report", MAINTAIN[0]),
            "evidence_only_vs_natural_before": {
                "histories": len(after_by_candidate.get("evidence_only_report", [])),
                "mean_difference": mean(r["delta_effectiveness"] for r in after_by_candidate.get("evidence_only_report", [])),
                "first_better": sum(r["delta_effectiveness"] > 0 for r in after_by_candidate.get("evidence_only_report", [])),
                "equal": sum(r["delta_effectiveness"] == 0 for r in after_by_candidate.get("evidence_only_report", [])),
                "second_better": sum(r["delta_effectiveness"] < 0 for r in after_by_candidate.get("evidence_only_report", []))},
            "request_evidence_delta": {"pairs": len(after_by_candidate.get("request_evidence", [])),
                                       "mean_delta": mean(r["delta_effectiveness"] for r in after_by_candidate.get("request_evidence", []))},
        },
        "doctor_responsiveness": responsiveness, "continuation_agent": reversal,
        "outcome_direction": {"agent_correct_before_a1": direction(agent_right, True),
                              "agent_incorrect_before_a1": direction(agent_right, False)},
    }
    (out / "signal_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "signal_report.md").write_text(markdown(report), encoding="utf-8")
    return report


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def num(value) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def markdown(r: dict) -> str:
    d, s = r["dataset"], r["signal"]
    lines = ["# v5 AgentClinic Effectiveness Signal Check", "",
             "## Dataset", f"- source cases: {d['source_cases']}; eligible: {d['eligible']}; sampled: {d['sampled']}",
             f"- missing-field counts: {d['ineligible_reasons']}", f"- selection: {d['selection']}",
             f"- sampled cases whose consultant-only evidence names the diagnosis: {d['gold_named_in_private_evidence']}",
             f"- labelled: {r['histories']} histories, {r['pairs']} pairs, K = {r['k']} "
             f"({r['before_rollouts']} BEFORE and {r['after_rollouts']} AFTER continuations)",
             f"- before A1: Agent standalone correct {pct(r['who_is_right_before_a1']['agent_standalone_correct'])}, "
             f"Doctor D1 correct {pct(r['who_is_right_before_a1']['doctor_d1_correct'])}", "",
             "## Candidate compliance",
             "| candidate | histories | compliant | observed | uses consultant-only evidence | fabricated | chosen attempt |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {n} | {c['generated']} | {c['compliant']} | {c['observed']} | {pct(c['uses_private_evidence'])} | "
              f"{pct(c['fabricated_evidence'])} | {c['attempts']} |" for n, c in r["compliance"].items()]
    lines += ["", "## Outcome distribution", f"- BEFORE success rate per history: {r['outcome_distribution']['before']}",
              "", "| candidate | mean AFTER | AFTER distribution |", "|---|---|---|"]
    lines += [f"| {n} | {v['mean']:.2f} | {v['distribution']} |" for n, v in r["outcome_distribution"]["after"].items()]
    rng = s["within_history_range"]
    lines += ["", "## Effectiveness signal",
              f"- candidate-history pairs with delta != 0: {pct(s['pairs_nonzero_delta'])}",
              f"- histories with at least one nonzero delta: {pct(s['histories_with_nonzero_delta'])}",
              f"- within-history range max(AFTER) - min(AFTER): mean {num(rng['mean'])}, median {num(rng['median'])}, "
              f"distribution {rng['distribution']}",
              f"- absolute delta: mean {num(s['mean_abs_delta'])}, median {num(s['median_abs_delta'])}",
              f"- histories where all candidates have identical AFTER: {s['histories_all_candidates_identical']} of {r['histories']}",
              f"- saturated histories: {r['saturated_histories']}",
              f"- by split: {r['by_split']}",
              "", "## Usable labels by state", "| state | " + " | ".join(CANDIDATES) + " |", "|---|" + "---|" * len(CANDIDATES)]
    lines += [f"| {state} | " + " | ".join(str(v[n]) for n in CANDIDATES) + " |" for state, v in r["usable_pairs_by_state"].items()]
    lines += ["", "## Delta by behavior", "| candidate | pairs | mean delta | delta distribution |", "|---|---|---|---|"]
    lines += [f"| {n} | {v['pairs']} | {num(v['mean'])} | {v['distribution']} |" for n, v in r["delta_by_behavior"].items()]
    lines += ["", "## Signal by collaboration state (D1 vs Agent standalone)",
              "| state | histories | with nonzero delta | all candidates identical | mean range | BEFORE | AFTER by candidate |",
              "|---|---|---|---|---|---|---|"]
    for state, v in r["signal_by_state"].items():
        after = ", ".join(f"{n} {v['after'][n]:.2f}" for n in CANDIDATES if v["after"][n] is not None)
        lines.append(f"| {state} | {v['histories']} | {v['with_nonzero_delta']} | {v['all_candidates_identical']} | "
                     f"{num(v['mean_range'])} | {num(v['before'])} | {after} |")
    lines += ["", "## Behavior comparisons (AFTER rate, same history)",
              "| comparison | histories | mean difference | first better | equal | second better |", "|---|---|---|---|---|---|"]
    b = r["behavior_comparisons"]
    for title, v in (("evidence_grounded_maintain - unsupported_maintain", b["evidence_grounded_vs_unsupported"]),
                     ("evidence_grounded_maintain - accept_revise", b["maintain_vs_accept"]),
                     ("evidence_only_report - evidence_grounded_maintain", b["evidence_only_vs_evidence_grounded"]),
                     ("evidence_only_report - natural A1 (BEFORE)", b["evidence_only_vs_natural_before"])):
        lines.append(f"| {title} | {v['histories']} | {num(v['mean_difference'])} | {v['first_better']} | {v['equal']} | {v['second_better']} |")
    lines += [f"- request_evidence: mean delta {num(b['request_evidence_delta']['mean_delta'])} over {b['request_evidence_delta']['pairs']} pairs",
              "", "## Doctor responsiveness",
              "| after | continuations | Doctor revised D1 diagnosis | revised toward the Agent | Agent gave private evidence | success |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {v['rollouts']} | {pct(v['doctor_revised'])} | {pct(v['revised_toward_agent'])} | "
              f"{pct(v['agent_communicated_private_evidence'])} | {pct(v['success'])} |" for n, v in r["doctor_responsiveness"].items()]
    lines += ["", "## Continuation Agent (A2)",
              "| after | continuations | A2 reversed A1 stance | without new Doctor evidence | with new Doctor evidence |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {v['rollouts']} | {pct(v['a2_reversed'])} | {pct(v['reversed_without_new_evidence'])} | "
              f"{pct(v['reversed_with_new_evidence'])} |" for n, v in r["continuation_agent"].items()]
    lines += ["", "## Outcome direction"]
    for title, v in (("Agent's standalone diagnosis correct", r["outcome_direction"]["agent_correct_before_a1"]),
                     ("Agent's standalone diagnosis incorrect", r["outcome_direction"]["agent_incorrect_before_a1"])):
        after = ", ".join(f"{n} {v['after'][n]:.2f}" if v["after"][n] is not None else f"{n} n/a" for n in CANDIDATES)
        lines += [f"- {title} ({v['histories']} histories): BEFORE {num(v['before'])}; AFTER {after}",
                  f"  - evidence_grounded - unsupported: {num(v['evidence_vs_unsupported']['mean_difference'])} "
                  f"({v['evidence_vs_unsupported']['histories']} histories); evidence_grounded - accept_revise: "
                  f"{num(v['evidence_vs_accept']['mean_difference'])} ({v['evidence_vs_accept']['histories']} histories)"]
    lines += ["", "## Diagnostics for the go / no-go decision",
              "The question: does changing the Agent response produce reproducible within-history variation in the "
              "Doctor's downstream success?",
              f"- histories that stay completely invariant: {s['histories_all_candidates_identical']} of {r['histories']}",
              "- candidate-specific success rates: see Outcome distribution",
              f"- grounded vs unsupported: mean difference {num(b['evidence_grounded_vs_unsupported']['mean_difference'])} "
              f"over {b['evidence_grounded_vs_unsupported']['histories']} histories",
              f"- sampling noise: with K = {r['k']}, one continuation changes a rate by {1 / r['k']:.2f}" if r["k"] else "",
              "", "Full labelling stays disabled (run_full_labeling: false) until this report has been reviewed.", ""]
    return "\n".join(lines)
