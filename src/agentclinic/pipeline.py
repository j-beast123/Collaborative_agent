"""Effectiveness-label rollouts on AgentClinic-MedQA (v5)."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.common.config import IntegrationUnavailable
from src.common.io import read_jsonl, write_jsonl
from src.evaluation.collaboration_states import STATES

from .adapter import agent_view, render
from .candidates import CANDIDATES, ComplianceJudge, candidate_type, guideline
from .candidates import PROMPT_VERSION as COMPLIANCE_VERSION
from .outcome import DiagnosisJudge
from .protocol import (ASSESSMENT, TWO_TURN, ClinicRuntime, ProtocolSpec, agent_message, continue_dialogue, continue_from_a1,
                       doctor_opening, standalone_prompt, stated)

QUESTION = "What is the most likely diagnosis?"
FROZEN = "histories.frozen.sha256"


def freeze_histories(out: Path) -> str:
    """Fix the prepared histories: later stages check this digest and prepare refuses to run."""
    digest = hashlib.sha256((out / "histories.jsonl").read_bytes()).hexdigest()
    (out / FROZEN).write_text(digest + "\n", encoding="utf-8")
    return digest


def check_frozen(out: Path) -> None:
    marker = out / FROZEN
    if marker.exists() and marker.read_text(encoding="utf-8").strip() != hashlib.sha256(
            (out / "histories.jsonl").read_bytes()).hexdigest():
        raise RuntimeError("histories.jsonl differs from the frozen snapshot")


class Appender:
    """Append-only JSONL with a key index; safe to share across threads."""

    def __init__(self, path: Path, key) -> None:
        self.path, self.key, self.lock = Path(path), key, threading.Lock()
        self.rows = read_jsonl(self.path, missing_ok=True)
        self.keys = {key(r) for r in self.rows}

    def add(self, row: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows.append(row)
            self.keys.add(self.key(row))


def success(case: dict, final_diagnosis: Optional[str], judge: DiagnosisJudge) -> dict:
    """Outcome of one dialogue: the equivalence judge sees only the two diagnosis strings."""
    verdict = judge.equivalent(case["gold"], final_diagnosis or "")
    return {"success": verdict["equivalent"], "outcome_method": verdict["method"], "outcome_reason": verdict["reason"]}


def prepare(cases: list[dict], runtime: ClinicRuntime, judge: DiagnosisJudge, out: Path, workers: int,
            standalone_file: Optional[Path] = None) -> dict:
    """histories.jsonl: one row per case with D1, the standalone assessment and the natural collaboration."""
    if (out / FROZEN).exists():
        raise IntegrationUnavailable(f"histories are frozen ({out / FROZEN}): D1, the standalone assessments and the "
                                     "natural dialogues are not sampled again")
    histories = Appender(out / "histories.jsonl", key=lambda r: r["case_id"])
    shared = Appender(standalone_file, key=lambda r: r["case_id"]) if standalone_file else None

    def standalone_assessment(case) -> str:
        if shared and case["case_id"] in shared.keys:
            return next(r["agent_standalone_diagnosis"] for r in shared.rows if r["case_id"] == case["case_id"])
        diagnosis = stated((runtime.agent_greedy or runtime.agent)(runtime.agent_system, standalone_prompt(case)),
                           ASSESSMENT)
        if shared:
            shared.add({"case_id": case["case_id"], "agent_standalone_diagnosis": diagnosis})
        return diagnosis

    def run(case):
        history = doctor_opening(case, runtime)
        standalone = standalone_assessment(case)
        natural = continue_from_a1(case, history, runtime)
        histories.add({
            "history_id": f"{case['case_id']}:A1", "case_id": case["case_id"], "history": history,
            "doctor_provisional_diagnosis": history[0]["provisional_diagnosis"],
            "agent_standalone_diagnosis": standalone,
            "natural": {"dialogue": natural["dialogue"], "final_diagnosis": natural["final_diagnosis"],
                        "status": natural["status"], **success(case, natural["final_diagnosis"], judge)},
            "gold": case["gold"],
        })

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, [c for c in cases if c["case_id"] not in histories.keys]))
    return {"histories": len(histories.rows)}


def prepare_diagnostics(out: Path, judge: DiagnosisJudge) -> dict:
    """Who is right before A1? Per case (prepare_diagnostics.jsonl) and in total (prepare_summary.json)."""
    def same(a, b):
        return judge.equivalent(a, b)["equivalent"] if a and b else None

    rows = []
    for h in read_jsonl(out / "histories.jsonl"):
        doctor_ok, agent_ok = same(h["gold"], h["doctor_provisional_diagnosis"]), same(h["gold"], h["agent_standalone_diagnosis"])
        rows.append({
            "case_id": h["case_id"], "gold": h["gold"],
            "d1_diagnosis": h["doctor_provisional_diagnosis"], "d1_correct": doctor_ok,
            "agent_standalone_diagnosis": h["agent_standalone_diagnosis"], "agent_standalone_correct": agent_ok,
            "d1_agent_agree": same(h["doctor_provisional_diagnosis"], h["agent_standalone_diagnosis"]),
            "natural_final_diagnosis": h["natural"]["final_diagnosis"],
            "natural_final_correct": same(h["gold"], h["natural"]["final_diagnosis"]),
            "collaboration_state": STATES.get((agent_ok, doctor_ok)),
        })
    rows.sort(key=lambda r: r["case_id"])
    (out / "prepare_diagnostics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                   encoding="utf-8")

    def count(test) -> int:
        return sum(bool(test(r)) for r in rows)

    summary = {
        "cases": len(rows),
        "doctor_d1_correct": count(lambda r: r["d1_correct"]),
        "agent_standalone_correct": count(lambda r: r["agent_standalone_correct"]),
        "natural_final_correct": count(lambda r: r["natural_final_correct"]),
        "doctor_and_agent_correct (Preserve)": count(lambda r: r["collaboration_state"] == "Preserve"),
        "agent_only_correct (Challenge)": count(lambda r: r["collaboration_state"] == "Challenge"),
        "doctor_only_correct (Accept)": count(lambda r: r["collaboration_state"] == "Accept"),
        "both_wrong (Recover)": count(lambda r: r["collaboration_state"] == "Recover"),
        "state_undefined (a diagnosis was not stated or not judged)": count(lambda r: r["collaboration_state"] is None),
        "d1_and_agent_disagree": count(lambda r: r["d1_agent_agree"] is False),
        "natural_final_minus_d1": count(lambda r: r["natural_final_correct"]) - count(lambda r: r["d1_correct"]),
        "gained_by_collaboration (D1 wrong, final right)": count(lambda r: not r["d1_correct"] and r["natural_final_correct"]),
        "lost_by_collaboration (D1 right, final wrong)": count(lambda r: r["d1_correct"] and not r["natural_final_correct"]),
        "natural_final_by_state": {state: f"{count(lambda r: r['collaboration_state'] == state and r['natural_final_correct'])}"
                                          f"/{count(lambda r: r['collaboration_state'] == state)}"
                                   for state in STATES.values()},
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def a2_histories(histories: list[dict]) -> tuple[list[dict], list[dict]]:
    """(A2 label histories, skipped) from prepared A1 histories: h2 = D1 -> A1 -> D2 of the natural collaboration."""
    rows, skipped = [], []
    for row in sorted(histories, key=lambda r: r["case_id"]):
        dialogue = row["natural"]["dialogue"]
        if [m["stage"] for m in dialogue[:3]] != ["D1", "A1", "D2"] or not dialogue[1]["text"] or not dialogue[2]["text"]:
            skipped.append({"case_id": row["case_id"], "reason": "the natural collaboration has no D1 -> A1 -> D2"})
            continue
        rows.append({
            "history_id": f"{row['case_id']}:A2", "source_history_id": row["history_id"], "case_id": row["case_id"],
            "intervention_turn": "A2", "history": dialogue[:3],
            "agent_assessment_before": dialogue[1].get("assessment") or row["agent_standalone_diagnosis"],
            "agent_standalone_diagnosis": row["agent_standalone_diagnosis"],
            "doctor_provisional_diagnosis": dialogue[2].get("provisional_diagnosis") or row["doctor_provisional_diagnosis"],
            "gold": row["gold"]})
    return rows, skipped


def turn_histories(trajectories: list[dict]) -> list[dict]:
    """Label histories of every Agent turn that the base collaborations reached (the main protocol): for a trajectory
    that stopped after A3, the histories before A1, A2 and A3 and no others.
    """
    rows = []
    for trajectory in sorted(trajectories, key=lambda t: t["case_id"]):
        if trajectory["trajectory_status"] != "valid":
            continue
        turns, held = trajectory["turns"], trajectory["agent_initial_position"]
        for index, turn in enumerate(turns):
            if turn["speaker"] != "agent":
                continue
            history = [{"speaker": t["speaker"], "stage": t["stage"], "text": t["text"],
                        **{k: t[k] for k in ("assessment", "provisional_diagnosis") if t.get(k)}} for t in turns[:index]]
            rows.append({"history_id": f"{trajectory['case_id']}:{turn['stage']}", "source_trajectory_id": trajectory["trajectory_id"],
                         "case_id": trajectory["case_id"], "intervention_turn": turn["stage"], "history": history,
                         "agent_assessment_before": held, "agent_standalone_diagnosis": trajectory["agent_initial_position"],
                         "doctor_provisional_diagnosis": turns[index - 1].get("provisional_diagnosis"),
                         "gold": trajectory["ground_truth"]["correct_diagnosis"]})
            held = turn.get("assessment") or held
    return rows


def select_label_histories(histories: list[dict], judge: DiagnosisJudge, *, n: int, seed: int) -> dict:
    """Histories for the K-rollout signal pilot: those where the Doctor's D1 diagnosis and the Agent's standalone
    diagnosis differ, then a seeded random sample of n.
    """
    ordered = sorted(histories, key=lambda h: h["case_id"])
    pool = [h for h in ordered if h["doctor_provisional_diagnosis"] and h["agent_standalone_diagnosis"]
            and judge.equivalent(h["doctor_provisional_diagnosis"], h["agent_standalone_diagnosis"])["equivalent"] is False]
    chosen = sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda h: h["case_id"])
    return {"criterion": "D1 diagnosis != Agent standalone diagnosis (equivalence judge); seeded random sample",
            "seed": seed, "requested": n, "pool_case_ids": [h["case_id"] for h in pool],
            "selected_case_ids": [h["case_id"] for h in chosen]}


def integrity_check(out: Path, *, k: int, splits: dict[str, str], thresholds: dict) -> dict:
    """Checks before any training on the labels."""
    import math

    labels = {split: read_jsonl(out / f"labels_{split}.jsonl", missing_ok=True) for split in ("train", "dev")}
    rows = labels["train"] + labels["dev"]
    compliance = read_jsonl(out / "candidate_compliance.jsonl", missing_ok=True)
    cases = {split: {r["case_id"] for r in labels[split]} for split in labels}

    def well_formed(r: dict) -> bool:
        values = (r.get("before_rate"), r.get("after_rate"), r.get("delta_effectiveness"))
        return (r["before"]["trials"] == r["after"]["trials"] == k
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                and 0 <= r["before"]["successes"] <= k and 0 <= r["after"]["successes"] <= k
                and isinstance(r.get("response"), str) and bool(r["response"].strip())
                and r.get("behavior_compliant") is True)

    before = {}
    for r in rows:
        before.setdefault(r["history_id"], set()).add(r["before"]["successes"])
    compliant_share = sum(bool(c["behavior_compliant"]) for c in compliance) / len(compliance) if compliance else 0.0
    dev_nonzero = sum(r["delta_effectiveness"] != 0 for r in labels["dev"])
    checks = {
        "train_and_dev_label_files_not_empty": bool(labels["train"]) and bool(labels["dev"]),
        f"every_label_has_exactly_k_{k}_before_and_after_and_no_missing_value": all(well_formed(r) for r in rows),
        "before_is_shared_within_a_history": all(len(v) == 1 for v in before.values()),
        "no_case_in_both_train_and_dev": not (cases["train"] & cases["dev"]),
        "every_label_case_is_in_its_split_and_none_is_test": all(splits.get(r["case_id"]) == r["split"] for r in rows),
        "no_duplicate_pair": len({(r["history_id"], r["response_id"]) for r in rows}) == len(rows),
        f"train_pairs_at_least_{thresholds['min_train_pairs']}": len(labels["train"]) >= thresholds["min_train_pairs"],
        f"dev_pairs_at_least_{thresholds['min_dev_pairs']}": len(labels["dev"]) >= thresholds["min_dev_pairs"],
        f"dev_nonzero_delta_pairs_at_least_{thresholds['min_dev_nonzero_delta_pairs']}":
            dev_nonzero >= thresholds["min_dev_nonzero_delta_pairs"],
        f"compliant_share_at_least_{thresholds['min_compliant_share']}": compliant_share >= thresholds["min_compliant_share"],
    }
    report = {"passed": all(checks.values()), "checks": checks, "failures": [name for name, ok in checks.items() if not ok],
              "numbers": {"train_pairs": len(labels["train"]), "dev_pairs": len(labels["dev"]),
                          "train_cases": len(cases["train"]), "dev_cases": len(cases["dev"]),
                          "dev_nonzero_delta_pairs": dev_nonzero,
                          "train_nonzero_delta_pairs": sum(r["delta_effectiveness"] != 0 for r in labels["train"]),
                          "compliant_share": round(compliant_share, 3)}}
    (out / "integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_candidates(cases: dict[str, dict], histories: list[dict], runtime: Optional[ClinicRuntime],
                        compliance: ComplianceJudge, out: Path, workers: int, max_attempts: int = 3,
                        stage: Optional[str] = "A1", names: tuple = tuple(CANDIDATES), spec: ProtocolSpec = TWO_TURN) -> dict:
    """Candidates per history, each regenerated until it complies (at most max_attempts)."""
    attempts = defaultdict(list)
    for row in read_jsonl(out / "candidate_attempts.jsonl", missing_ok=True):
        attempts[(row["history_id"], row["candidate_id"])].append(row)
    if not attempts:  # runs made before attempts were logged: the stored response is attempt 1
        for row in read_jsonl(out / "candidates.jsonl", missing_ok=True):
            attempts[(row["history_id"], row["candidate_id"])].append(dict(row, attempt=1))
    lock = threading.Lock()

    def run(job):
        row, name = job
        case, key = cases[row["case_id"]], (row["history_id"], name)
        standalone = row.get("agent_assessment_before", row["agent_standalone_diagnosis"])
        mine = attempts[key]
        for attempt in mine:  # judgments of an older judge version are redone on the stored response
            if attempt.get("judge_version") != getattr(compliance, "version", COMPLIANCE_VERSION):
                attempt.update(compliance.judge(case, row["history"], standalone, attempt["response"], name))
        while runtime and not any(a["behavior_compliant"] for a in mine) and len(mine) < max_attempts:
            instruction = guideline(name, standalone or "not stated")
            message = agent_message(case, row["history"], stage or row["intervention_turn"], runtime, instruction, spec)
            generated = {"history_id": row["history_id"], "case_id": row["case_id"], "candidate_id": name,
                         "attempt": len(mine) + 1, "candidate_type": candidate_type(name), "intended_behavior": name,
                         "instruction": instruction, "response": message["text"], "assessment": message["assessment"],
                         "raw": message["raw"],
                         **compliance.judge(case, row["history"], standalone, message["text"], name)}
            with lock:
                mine.append(generated)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, [(row, name) for row in histories for name in names]))
    wanted = {(row["history_id"], name) for row in histories for name in names}
    chosen = [next((a for a in rows if a["behavior_compliant"]), rows[-1]) for key, rows in sorted(attempts.items())
              if rows and key in wanted]
    judgment_fields = ("intended_behavior", "intended_behavior_class", "observed_behavior", "uses_private_evidence",
                       "fabricated_evidence", "stance", "judge_rationale", "judge_status", "judge_version", "behavior_compliant")
    write_jsonl(out / "candidate_attempts.jsonl", [a for _, rows in sorted(attempts.items()) for a in rows])
    write_jsonl(out / "candidates.jsonl", [{k: v for k, v in a.items() if k not in judgment_fields or k == "intended_behavior"}
                                           for a in chosen])
    write_jsonl(out / "candidate_compliance.jsonl",
                [{"history_id": a["history_id"], "case_id": a["case_id"], "candidate_id": a["candidate_id"],
                  "attempt": a["attempt"], **{k: a.get(k) for k in judgment_fields}} for a in chosen])
    return {"candidates": len(chosen), "generations": sum(len(rows) for rows in attempts.values()),
            "compliant": {name: f"{sum(bool(a['behavior_compliant']) for a in chosen if a['candidate_id'] == name)}"
                                f"/{sum(a['candidate_id'] == name for a in chosen)}" for name in names}}


def rescore(out: Path, judge: DiagnosisJudge) -> dict:
    """Judge every stored continuation again (the dialogues are kept as they are) and drop labels.jsonl,
    which the labels stage then rebuilds from the rollouts. Previous files are kept with the given tag."""
    changed = {}
    for name in ("before_rollouts.jsonl", "after_rollouts.jsonl"):
        path = out / name
        rows = read_jsonl(path, missing_ok=True)
        if not rows:
            continue
        flips = 0
        for row in rows:
            verdict = judge.equivalent(row["gold"], row["final_diagnosis"] or "")
            flips += verdict["equivalent"] != row["success"]
            row.update(success=verdict["equivalent"], outcome_method=verdict["method"], outcome_reason=verdict["reason"])
        write_jsonl(path, rows)
        changed[name] = {"continuations": len(rows), "outcome_changed": flips}
    labels = out / "labels.jsonl"
    if labels.exists():
        labels.replace(out / "labels.before_rescore.jsonl")
    return changed


def build_labels(cases: dict[str, dict], histories: list[dict], runtime: ClinicRuntime, judge: DiagnosisJudge,
                 out: Path, *, k: int, workers: int, require_compliance: bool, metadata: dict,
                 max_attempts: int = 3, stage: Optional[str] = "A1", spec: ProtocolSpec = TWO_TURN) -> dict:
    """before_rollouts.jsonl, after_rollouts.jsonl (every continuation) and labels.jsonl (one row per pair)."""
    before = Appender(out / "before_rollouts.jsonl", key=lambda r: (r["history_id"], r["rollout_index"]))
    # a continuation belongs to one generation of a candidate (rows written before attempts existed are attempt 1)
    after = Appender(out / "after_rollouts.jsonl",
                     key=lambda r: (r["history_id"], r["candidate_id"], r.get("candidate_attempt", 1), r["rollout_index"]))
    labels = Appender(out / "labels.jsonl", key=lambda r: (r["history_id"], r["response_id"]))
    candidates = {(c["history_id"], c["candidate_id"]): c for c in read_jsonl(out / "candidates.jsonl")}
    compliance = {(c["history_id"], c["candidate_id"]): c for c in read_jsonl(out / "candidate_compliance.jsonl")}

    def rollout(job):
        row, name, index = job
        case = cases[row["case_id"]]
        fixed = candidates[(row["history_id"], name)] if name else {}
        log, key = (before, (row["history_id"], index)) if name is None \
            else (after, (row["history_id"], name, fixed.get("attempt", 1), index))
        if key in log.keys:
            return
        for attempt in range(1, max_attempts + 1):  # a FINAL that is not valid JSON is a format failure, rerun
            if stage == "A1":
                result = continue_from_a1(case, row["history"], runtime, fixed_a1=fixed.get("response"),
                                          fixed_assessment=fixed.get("assessment") or "")
            else:
                result = continue_dialogue(case, row["history"], runtime, spec=spec, fixed={
                    "text": fixed["response"], "assessment": fixed.get("assessment") or ""} if name else None)
            if result["status"] == "completed":
                break
        log.add({"history_id": row["history_id"], "case_id": row["case_id"],
                 **({} if name is None else {"candidate_id": name, "candidate_attempt": fixed.get("attempt", 1)}),
                 "rollout_index": index, "attempt": attempt, "turns": result.get("turns"), "stop": result.get("stop"),
                 "status": result["status"], "final_diagnosis": result["final_diagnosis"], "gold": case["gold"],
                 **success(case, result["final_diagnosis"], judge), "dialogue": result["dialogue"]})

    def labelled(row) -> list[str]:
        return [name for name in CANDIDATES if (row["history_id"], name) in candidates
                and not (require_compliance and not compliance.get((row["history_id"], name), {}).get("behavior_compliant"))]

    jobs = [(row, name, index) for row in histories for name in [None] + labelled(row) for index in range(k)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(rollout, jobs))

    def counts(rows: list[dict]) -> Optional[dict]:
        """{successes, trials}; None unless all K continuations completed with a judged outcome."""
        done = [r for r in rows if r["status"] == "completed" and r["success"] is not None]
        return {"successes": sum(r["success"] for r in done), "trials": len(done)} if len(done) == k else None

    incomplete = 0
    for row in histories:
        case = cases[row["case_id"]]
        b = counts([r for r in before.rows if r["history_id"] == row["history_id"]])
        for name in labelled(row):
            if (row["history_id"], name) in labels.keys:
                continue
            chosen = candidates[(row["history_id"], name)].get("attempt", 1)
            a = counts([r for r in after.rows if r["history_id"] == row["history_id"] and r["candidate_id"] == name
                        and r.get("candidate_attempt", 1) == chosen])
            if a is None or b is None:
                incomplete += 1
                continue
            judged = compliance.get((row["history_id"], name), {})
            labels.add({
                "case_id": row["case_id"], "original_case_id": row["case_id"], "history_id": row["history_id"],
                "response_id": name, "candidate_id": name, "split": case.get("split") or metadata["split"],
                **({} if stage == "A1" else {"intervention_turn": stage or row["intervention_turn"]}),
                # predictor input: what the Agent has when it writes the turn (no ground truth, no future dialogue)
                "history": {**({} if stage == "A1" else {"intervention_turn": stage or row["intervention_turn"]}),
                            "question": QUESTION, "options": {}, "scenario": render(agent_view(case)),
                            "turns": [{"speaker": m["speaker"], "text": m["text"]} for m in row["history"]]},
                "response": candidates[(row["history_id"], name)]["response"],
                "before": b, "after": a,
                "before_rate": b["successes"] / k, "after_rate": a["successes"] / k,
                "delta_effectiveness": a["successes"] / k - b["successes"] / k,
                "candidate_type": candidate_type(name), "intended_behavior": name, "candidate_attempt": chosen,
                **{key: judged.get(key) for key in ("intended_behavior_class", "observed_behavior", "behavior_compliant",
                                                    "uses_private_evidence", "fabricated_evidence")},
                "meta": {"agent_standalone_diagnosis": row["agent_standalone_diagnosis"],
                         "doctor_provisional_diagnosis": row["doctor_provisional_diagnosis"],
                         **({"agent_assessment_before": row["agent_assessment_before"]} if "agent_assessment_before" in row else {})},
                "metadata": metadata,
            })
    return {"before_rollouts": len(before.rows), "after_rollouts": len(after.rows), "labels": len(labels.rows),
            "incomplete_pairs": incomplete}
