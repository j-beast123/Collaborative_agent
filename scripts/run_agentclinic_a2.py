"""A2 Effectiveness labels on AgentClinic-MedQA (v6): the Agent's second response as the intervention."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.run_agentclinic import build_runtime, protocol_identity, select, write_split_files
from src.agentclinic.adapter import agent_view, assert_no_leakage, render
from src.agentclinic.outcome import normalize
from src.agentclinic.pipeline import a2_histories, build_labels, generate_candidates, integrity_check
from src.agentclinic.protocol import agent_prompt, doctor_prompt
from src.common.config import read_config
from src.common.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
STAGE = "A2"


def scope_histories(config: dict, cases: dict[str, dict], scope: str) -> tuple[list[dict], list[dict]]:
    """A2 histories of the scope from the A1 run's frozen histories (read only)."""
    source = [h for h in read_jsonl(ROOT / config["source_histories"]) if h["case_id"] in cases]
    rows, skipped = a2_histories(source)
    if scope == "smoke":
        smoke = config["smoke"]
        pool = [r for r in rows if cases[r["case_id"]]["split"] == smoke["split"]]
        rows = sorted(random.Random(smoke["seed"]).sample(pool, min(smoke["histories"], len(pool))), key=lambda r: r["case_id"])
    return rows, skipped


def information_section(prompt: str) -> str:
    return prompt.split("Conversation so far:")[0]


def leakage_audit(out: Path, cases: dict[str, dict], histories: list[dict]) -> dict:
    """Correct_Diagnosis reaches nothing but the outcome judgment: not the A2 prompt, the Doctor's FINAL prompt, a
    candidate instruction or the predictor input.
    """
    by_history = {h["history_id"]: h for h in histories}
    labels = [r for r in read_jsonl(out / "labels.jsonl", missing_ok=True) if r["history_id"] in by_history]
    attempts = [a for a in read_jsonl(out / "candidate_attempts.jsonl", missing_ok=True) if a["history_id"] in by_history]
    problems = []
    for row in histories:
        case, gold = cases[row["case_id"]], normalize(cases[row["case_id"]]["gold"])
        assert_no_leakage(case)
        for name, prompt in (("A2 prompt", agent_prompt(case, row["history"], STAGE)),
                             ("Doctor FINAL prompt", doctor_prompt(case, row["history"] + [
                                 {"speaker": "agent", "stage": "A2", "text": "x"}], "FINAL"))):
            if gold and gold in normalize(information_section(prompt)):
                problems.append(f"{row['history_id']}: the correct diagnosis is in the information section of the {name}")
            if "Correct_Diagnosis" in prompt:
                problems.append(f"{row['history_id']}: evaluator-only field in the {name}")
    for attempt in attempts:
        row = by_history[attempt["history_id"]]
        gold, own = normalize(cases[row["case_id"]]["gold"]), normalize(row["agent_assessment_before"] or "")
        if gold and gold in normalize(attempt["instruction"]) and gold not in own:
            problems.append(f"{attempt['history_id']}/{attempt['candidate_id']}: the instruction names the correct diagnosis")
    for label in labels:
        case = cases[label["case_id"]]
        history = label["history"]
        if history["scenario"] != render(agent_view(case)) or normalize(case["gold"]) in normalize(history["scenario"]):
            problems.append(f"{label['history_id']}: predictor scenario is not the Agent's view or names the correct diagnosis")
        if [t["text"] for t in history["turns"]] != [m["text"] for m in by_history[label["history_id"]]["history"]] \
                or history.get("intervention_turn") != STAGE or label.get("intervention_turn") != STAGE:
            problems.append(f"{label['history_id']}: predictor history is not D1 -> A1 -> D2 marked as A2")
        if any(k in json.dumps(label["history"]) for k in ("Correct_Diagnosis", "final_diagnosis", "ground_truth")):
            problems.append(f"{label['history_id']}: outcome or evaluator field in the predictor input")
        if label["split"] != case["split"] or case["split"] not in ("train", "dev"):
            problems.append(f"{label['history_id']}: split {label['split']} is not the case's TRAIN/DEV split")
    splits = defaultdict(set)
    for label in labels:
        splits[label["split"]].add(label["case_id"])
    if splits["train"] & splits["dev"]:
        problems.append("a case has labels in TRAIN and in DEV")
    return {"passed": not problems, "problems": problems, "histories": len(histories), "labels": len(labels),
            "candidate_generations": len(attempts), "cases_by_split": {s: len(c) for s, c in splits.items()}}


def sanity(out: Path, histories: list[dict], k: int, names: list[str]) -> dict:
    """Are the A2 labels informative? Nothing here is a predictor threshold."""
    wanted = {h["history_id"] for h in histories}
    labels = [r for r in read_jsonl(out / "labels.jsonl", missing_ok=True) if r["history_id"] in wanted]
    compliance = [c for c in read_jsonl(out / "candidate_compliance.jsonl", missing_ok=True) if c["history_id"] in wanted]
    candidates = {(c["history_id"], c["candidate_id"]): c for c in read_jsonl(out / "candidates.jsonl", missing_ok=True)
                  if c["history_id"] in wanted}
    after = [r for r in read_jsonl(out / "after_rollouts.jsonl", missing_ok=True) if r["history_id"] in wanted]
    by_history = defaultdict(list)
    for row in labels:
        by_history[row["history_id"]].append(row)
    before = {hid: rows[0]["before"]["successes"] for hid, rows in by_history.items()}
    finals = defaultdict(lambda: defaultdict(set))  # history -> candidate -> the Doctor's final diagnoses
    for row in after:
        if row["status"] == "completed":
            finals[row["history_id"]][row["candidate_id"]].add(normalize(row["final_diagnosis"] or ""))
    labelled = {(r["history_id"], r["candidate_id"]) for r in labels}
    responsive = [hid for hid, per in finals.items()
                  if len({frozenset(v) for name, v in per.items() if (hid, name) in labelled}) > 1]
    ordered = [hid for hid, rows in by_history.items() if len({r["after"]["successes"] for r in rows}) > 1]
    texts = defaultdict(list)
    for (hid, name), candidate in candidates.items():
        texts[hid].append(candidate["response"])
    return {
        "histories": len(histories), "histories_with_labels": len(by_history), "labelled_pairs": len(labels), "k": k,
        "candidate_compliance": {name: {"compliant": sum(bool(c["behavior_compliant"]) for c in compliance if c["candidate_id"] == name),
                                        "generated": sum(c["candidate_id"] == name for c in compliance)} for name in names},
        "compliant_share": mean(bool(c["behavior_compliant"]) for c in compliance) if compliance else None,
        "before_rate_distribution": {str(m): sum(v == m for v in before.values()) for m in range(k + 1)},
        "before_floor_share": mean(v == 0 for v in before.values()) if before else None,
        "before_ceiling_share": mean(v == k for v in before.values()) if before else None,
        "after_floor_or_ceiling_share": mean(r["after"]["successes"] in (0, k) for r in labels) if labels else None,
        "nonzero_delta_pairs": sum(r["delta_effectiveness"] != 0 for r in labels),
        "nonzero_delta_pair_share": mean(r["delta_effectiveness"] != 0 for r in labels) if labels else None,
        "histories_with_a_nonzero_delta": sum(any(r["delta_effectiveness"] != 0 for r in rows) for rows in by_history.values()),
        "histories_with_candidate_ordering": len(ordered),
        "histories_where_the_final_diagnosis_depends_on_the_candidate": len(responsive),
        "mean_abs_delta": mean(abs(r["delta_effectiveness"]) for r in labels) if labels else None,
        "delta_by_candidate": {name: {"n": len(rows), "mean_delta": mean(r["delta_effectiveness"] for r in rows),
                                      "mean_after_rate": mean(r["after_rate"] for r in rows)}
                               for name in names if (rows := [r for r in labels if r["candidate_id"] == name])},
        "response_words_by_candidate": {name: round(mean(len(c["response"].split()) for (_, n), c in candidates.items() if n == name), 1)
                                        for name in names if any(n == name for _, n in candidates)},
        "histories_with_identical_candidate_texts": sum(len(set(v)) < len(v) for v in texts.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--scope", choices=("smoke", "full"), required=True)
    p.add_argument("stages", nargs="+", choices=("candidates", "labels", "report"))
    a = p.parse_args()
    config = read_config(a.config)
    if config.get("intervention_turn") != STAGE:
        raise SystemExit("this script labels A2 interventions: the config must set intervention_turn to A2")
    all_cases, dataset = select(config, "full")  # TRAIN and DEV of the case-level split; TEST is never selected
    cases = {c["case_id"]: c for c in all_cases}
    if any(c["split"] not in ("train", "dev") for c in all_cases):
        raise SystemExit("a case outside TRAIN and DEV was selected")
    out = ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    histories, skipped = scope_histories(config, cases, a.scope)
    names = tuple(config["candidates"]["names"])
    existing = out / "histories_a2.jsonl"
    known = {r["history_id"]: r for r in read_jsonl(existing, missing_ok=True)}
    if any(known[h["history_id"]]["history"] != h["history"] for h in histories if h["history_id"] in known):
        raise SystemExit("an A2 history differs from the one already labelled in this directory")
    write_jsonl(existing, sorted({**known, **{h["history_id"]: h for h in histories}}.values(), key=lambda r: r["history_id"]))
    (out / f"run_config_{a.scope}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    results = {"scope": a.scope, "histories": len(histories), "skipped": skipped,
               "by_split": {s: sum(cases[h["case_id"]]["split"] == s for h in histories) for s in ("train", "dev")}}
    workers = config["labels"]["workers"]
    if any(stage in ("candidates", "labels") for stage in a.stages):
        runtime, diagnosis, compliance = build_runtime(config)
        try:
            if "candidates" in a.stages:
                results["candidates"] = generate_candidates(cases, histories, runtime, compliance, out, workers,
                                                            max_attempts=config["candidates"]["max_attempts"],
                                                            stage=STAGE, names=names)
            if "labels" in a.stages:
                results["labels"] = build_labels(
                    cases, histories, runtime, diagnosis, out, k=config["k"], workers=workers,
                    require_compliance=config["labels"]["require_compliance"],
                    max_attempts=config["labels"]["max_rollout_attempts"], stage=STAGE,
                    metadata={"version": config["version"], "k": config["k"], "models": config["models"],
                              "decoding": config["decoding"], "intervention_turn": STAGE, **protocol_identity(config),
                              "split": "mixed"})
        finally:
            if runtime.close:
                runtime.close()
    if "report" in a.stages:
        report = sanity(out, histories, config["k"], list(names))
        audit = leakage_audit(out, cases, histories)
        (out / f"a2_sanity_{a.scope}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / f"a2_leakage_audit_{a.scope}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        results["sanity"], results["leakage_audit"] = report, {k: v for k, v in audit.items() if k != "problems"}
        stop = []
        if not audit["passed"]:
            stop.append(f"leakage audit failed: {audit['problems'][:3]}")
        if a.scope == "smoke":
            rule = config["smoke"]["stop_if"]
            for name, value in (("labelled_pairs", report["labelled_pairs"]),
                                ("histories_with_candidate_ordering", report["histories_with_candidate_ordering"]),
                                ("nonzero_delta_pairs", report["nonzero_delta_pairs"])):
                if value < rule[f"{name}_below"]:
                    stop.append(f"{name} = {value} < {rule[f'{name}_below']}")
        else:
            results["label_files"] = write_split_files(out, dataset, config)
            results["integrity"] = integrity_check(out, k=config["k"], splits=dataset["splits"], thresholds=config["integrity"])
            if not results["integrity"]["passed"]:
                stop.append("integrity check failed: " + ", ".join(results["integrity"]["failures"]))
        results["proceed"] = not stop
        results["stop_reasons"] = stop
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    if results.get("stop_reasons"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
