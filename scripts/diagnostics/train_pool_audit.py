"""Feasibility audit (diagnostic; not the method, not read by any run): how many cases of the whole original TRAIN pool can
the Qwen3-14B Agent solve alone but not in the collaboration, and how many of those failures are attributable to the
interaction? DEV and TEST are never read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agentclinic.adapter import agent_view, render
from src.agentclinic.pipeline import Appender
from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
POOL_ENV = "configs/effectiveness/v10_qwen4b_gemma12b_max7.json"
REPEATS = 3
VERSION = "train-pool-audit-v1"
AUDIT_PROMPT = """You audit failed clinical collaborations of one case for a research analysis. A consulting Agent (who alone
has the examination and test findings) talks with a primary Doctor (who has only the history); the Doctor's final diagnosis
was wrong in each collaboration you receive. Asked alone, before any conversation, the Agent named the correct diagnosis.
You receive the correct diagnosis, the Agent's information, and the failed collaborations with turn ids, the Agent's stated
assessment per turn and its state before each turn.

Use only what is written. Do not credit the Agent with a hypothesis it did not state in the collaboration. The question is
whether the failure is attributable to the interaction: the Agent held the correct diagnosis in the collaboration and lost it
or failed to carry it through the interaction.

Answer over the collaborations together (say "in k of n" where they differ):
- correct_hypothesis_in_collaboration: did the Agent state the correct diagnosis (or an equivalent) in the collaboration?
  "yes", "no" or "some"
- first_turn_with_correct_hypothesis: the turn id where it first did, per collaboration (null when never)
- doctor_competing_diagnosis: the Doctor's competing diagnosis
- agent_moved_from_correct_to_wrong: did the Agent move from the correct diagnosis to the Doctor's or another wrong one?
- premature_agreement, did_not_surface_contradictory_evidence, demoted_correct_to_secondary, no_challenge_or_request:
  each "yes", "no" or "some"
- observable_interaction_bottleneck: one concrete sentence with turn ids, or "none"
- category: "INTERACTION_ATTRIBUTABLE" (the Agent held the correct diagnosis in the collaboration and an observable
  interaction action lost it), "KNOWLEDGE_REASONING_DESPITE_STANDALONE_SUCCESS" (in the collaboration the Agent itself did
  not hold the correct diagnosis, or reasoned to a wrong one, so a different interaction behavior alone would not carry it),
  or "AMBIGUOUS"
- confidence: "high", "medium" or "low"
- reason: two sentences

Return exactly one JSON object and no Markdown with these keys."""


def setup(args, with_model: bool):
    from src.agentclinic.selection import split_eligible_cases

    native = ROOT / args.native_run
    config = dict(json.loads((native / "resolved_config.json").read_text(encoding="utf-8")), output_dir=args.output_dir)
    pool_env = read_config(ROOT / POOL_ENV)
    cases = split_eligible_cases(pool_env, ROOT, allowed_splits=("train",))
    cases = {k: c for k, c in cases.items() if c["split"] == "train"}
    histories = {r["case_id"]: r for r in read_jsonl(ROOT / pool_env["output_dir"] / "histories.jsonl") if r["case_id"] in cases}
    # the TRAIN30 D1 of the clean environment must be the pool's D1 of the same case
    clean = {r["case_id"]: r for r in read_jsonl(ROOT / read_config(ROOT / config["environment"]["config"])["output_dir"] / "histories.jsonl")}
    mismatch = [k for k in clean if k in histories and json.dumps(clean[k]["history"]) != json.dumps(histories[k]["history"])]
    if mismatch:
        raise SystemExit(f"D1 differs between the pool and the clean environment for {mismatch}")
    clinic = None
    if with_model:
        from src.agentclinic.runtime import build_clinic, load_policy

        model, tokenizer = load_policy(config["policy"], None, trainable=False, device=config["devices"]["policy"])
        clinic = build_clinic(config, ROOT, model, tokenizer)
    return config, cases, histories, clinic


def collaboration(clinic, case, history, trajectory_id):
    from src.agentclinic.trajectory import collaborate
    from src.grpo.trainer import spec_of

    return collaborate(case, history["history"], clinic.runtime, clinic.judge, spec=spec_of(clinic.source),
                       max_attempts=3, trajectory_id=trajectory_id)


def row_of(t: dict, source: str, index: int) -> dict:
    return {"case_id": t["case_id"], "rerun": index, "source": source, "trajectory_status": t["trajectory_status"],
            "standalone_diagnosis": t["agent_initial_position"], "standalone_correct": bool(t["agent_initial_correct"]),
            "doctor_initial_diagnosis": t["doctor_initial_claim"], "doctor_initial_correct": bool(t["doctor_initial_correct"]),
            "final_diagnosis": t["final_diagnosis"], "collaboration_correct": bool(t["is_correct"]), "num_turns": t.get("num_turns"),
            "stop": t.get("stop"), "collaboration_state": t.get("collaboration_state"), "trajectory": t}


def screen(args) -> None:
    config, cases, histories, clinic = setup(args, with_model=True)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = Appender(out / "pool_collaborations.jsonl", key=lambda r: (r["case_id"], r["rerun"]))
    native = {t["case_id"]: t for t in read_jsonl(ROOT / args.native_run / "base/trajectories.jsonl")}
    for case_id, t in native.items():  # TRAIN30: the 14B-native run's base collaborations, as they are
        if case_id in cases and (case_id, 0) not in rows.keys:
            rows.add(row_of(t, f"{args.native_run}/base/trajectories.jsonl", 0))

    def one(case_id):
        if (case_id, 0) not in rows.keys:
            rows.add(row_of(collaboration(clinic, cases[case_id], histories[case_id], f"{case_id}:14b:pool:0"), "generated", 0))

    with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(one, sorted(cases)))
    print(len(rows.rows), Counter((r["standalone_correct"], r["collaboration_correct"]) for r in rows.rows if r["rerun"] == 0))


def group2(rows: list[dict]) -> list[str]:
    first = {r["case_id"]: r for r in rows if r["rerun"] == 0}
    return sorted(k for k, r in first.items() if r["standalone_correct"] and not r["collaboration_correct"])


def robust(args) -> None:
    from src.agentclinic.protocol import ASSESSMENT, stated, standalone_prompt

    config, cases, histories, clinic = setup(args, with_model=True)
    out = ROOT / args.output_dir
    rows = Appender(out / "pool_collaborations.jsonl", key=lambda r: (r["case_id"], r["rerun"]))
    sampled = Appender(out / "sampled_standalone.jsonl", key=lambda r: (r["case_id"], r["sample"]))
    targets = group2(rows.rows)

    def more(job):
        case_id, index = job
        if (case_id, index) not in rows.keys:
            rows.add(row_of(collaboration(clinic, cases[case_id], histories[case_id], f"{case_id}:14b:pool:{index}"), "generated", index))

    def alone(job):
        case_id, index = job
        if (case_id, index) in sampled.keys:
            return
        diagnosis = stated(clinic.runtime.agent(clinic.runtime.agent_system, standalone_prompt(cases[case_id])), ASSESSMENT)
        sampled.add({"case_id": case_id, "sample": index, "diagnosis": diagnosis,
                     "correct": bool(diagnosis) and bool(clinic.judge.equivalent(cases[case_id]["gold"], diagnosis)["equivalent"])})

    with clinic.generation(), ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
        list(pool.map(more, [(k, i) for k in targets for i in range(1, REPEATS)]))
        list(pool.map(alone, [(k, i) for k in targets for i in range(REPEATS)]))
    print("group 2", targets)


def classify(rows: list[dict], sampled: list[dict]) -> dict:
    result = {}
    for case_id in group2(rows):
        mine = [r for r in rows if r["case_id"] == case_id]
        alone = [s for s in sampled if s["case_id"] == case_id]
        success = sum(r["collaboration_correct"] for r in mine)
        competence = sum(s["correct"] for s in alone)
        result[case_id] = {"collaborations": len(mine), "collaboration_success": success, "sampled_standalone": len(alone),
                           "sampled_standalone_correct": competence,
                           "level": "ROBUST" if len(alone) == REPEATS and len(mine) == REPEATS and competence >= 2 and success <= 1 else "BORDERLINE"}
    return result


def audit(args) -> None:
    from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object

    config, cases, histories, _ = setup(args, with_model=False)
    out = ROOT / args.output_dir
    rows, sampled = read_jsonl(out / "pool_collaborations.jsonl"), read_jsonl(out / "sampled_standalone.jsonl", missing_ok=True)
    levels = classify(rows, sampled)
    reader = CachedChat(build_chat_backend(config["roles"]["rubric_manager"]), out / "cache_reader.jsonl")
    audits = Appender(out / "robust_audit.jsonl", key=lambda r: r["case_id"])

    def one(case_id):
        if case_id in audits.keys:
            return
        failed = [r["trajectory"] for r in rows if r["case_id"] == case_id and not r["collaboration_correct"]]
        user = json.dumps({"correct_diagnosis": cases[case_id]["gold"], "everything_the_agent_knew": render(agent_view(cases[case_id])),
                           "standalone_diagnosis": failed[0]["agent_initial_position"],
                           "failed_collaborations": [{"final_diagnosis": t["final_diagnosis"], "turns": [
                               {"turn_id": i, "speaker": m["speaker"], "stage": m["stage"], "text": m["text"],
                                **({"assessment": m.get("assessment"), "agent_state_before_this_turn": {k: (m.get("pre_response_state") or {}).get(k)
                                    for k in ("agent_current_assessment", "doctor_current_assessment", "better_supported_position", "evidence_sufficient")}}
                                   if m["speaker"] == "agent" else {})} for i, m in enumerate(t["turns"])]} for t in failed]}, ensure_ascii=False)
        answer = reader.complete(ChatRequest(AUDIT_PROMPT, user, VERSION, {"temperature": 0, "max_tokens": 900}))
        parsed, _ = parse_json_object(answer.text) if answer.status == "ok" else (None, None)
        audits.add({"case_id": case_id, "failed_collaborations": len(failed), **(parsed or {"category": None})})

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, [k for k, v in levels.items() if v["level"] == "ROBUST"]))


def report(args) -> None:
    out = ROOT / args.output_dir
    rows, sampled = read_jsonl(out / "pool_collaborations.jsonl"), read_jsonl(out / "sampled_standalone.jsonl", missing_ok=True)
    audits = {r["case_id"]: r for r in read_jsonl(out / "robust_audit.jsonl", missing_ok=True)}
    train30 = {json.loads(line)["case_id"] for line in (ROOT / args.native_run / "base/trajectories.jsonl").read_text().splitlines() if line.strip()}
    first = {r["case_id"]: r for r in rows if r["rerun"] == 0}
    levels = classify(rows, sampled)

    def groups(ids: set) -> dict:
        g = Counter()
        for k in ids:
            r = first[k]
            g[{(True, True): "G1_standalone_correct_collab_correct", (True, False): "G2_standalone_correct_collab_fail",
               (False, True): "G3_standalone_wrong_collab_correct", (False, False): "G4_standalone_wrong_collab_fail"}[
                (r["standalone_correct"], r["collaboration_correct"])]] += 1
        n = len(ids)
        return {"cases": n, **dict(sorted(g.items())),
                "standalone_correct": sum(first[k]["standalone_correct"] for k in ids),
                "collaboration_fail": sum(not first[k]["collaboration_correct"] for k in ids),
                "collaboration_sensitive_rate": round(g["G2_standalone_correct_collab_fail"] / n, 3) if n else None,
                "standalone_rate": round(sum(first[k]["standalone_correct"] for k in ids) / n, 3) if n else None,
                "collaboration_success_rate": round(sum(first[k]["collaboration_correct"] for k in ids) / n, 3) if n else None}

    g2 = group2(rows)
    robust_ids = sorted(k for k, v in levels.items() if v["level"] == "ROBUST")
    attributable = sorted(k for k in robust_ids if (audits.get(k) or {}).get("category") == "INTERACTION_ATTRIBUTABLE")
    total = len(first)
    pct = lambda n: f"{n} ({100 * n / total:.1f}%)" if total else str(n)  # noqa: E731
    summary = {
        "pool": groups(set(first)), "train30": groups(set(first) & train30), "rest_of_pool": groups(set(first) - train30),
        "funnel": {"total_train_pool": total, "standalone_correct": pct(sum(r["standalone_correct"] for r in first.values())),
                   "collaboration_fail": pct(sum(not r["collaboration_correct"] for r in first.values())),
                   "standalone_correct_and_collaboration_fail": pct(len(g2)), "robust_collaboration_sensitive": pct(len(robust_ids)),
                   "interaction_attributable": pct(len(attributable))},
        "group2_case_ids": [k[-4:] for k in g2],
        "group2_doctor_initial": {"wrong": sum(not first[k]["doctor_initial_correct"] for k in g2),
                                  "correct": sum(first[k]["doctor_initial_correct"] for k in g2),
                                  "stronger_candidates_doctor_wrong": [k[-4:] for k in g2 if not first[k]["doctor_initial_correct"]]},
        "robustness": {k[-4:]: v for k, v in levels.items()},
        "robust_case_ids": [k[-4:] for k in robust_ids],
        "audit": {k[-4:]: {f: (audits.get(k) or {}).get(f) for f in ("category", "confidence", "correct_hypothesis_in_collaboration",
                                                                          "agent_moved_from_correct_to_wrong", "observable_interaction_bottleneck", "reason")}
                  for k in robust_ids},
        "interaction_attributable_case_ids": [k[-4:] for k in attributable],
        "note": "diagnostic; 14B Agent; strict majority-v1; standalone = the protocol's greedy diagnosis in the screen; DEV/TEST not read"}
    (out / "train_pool_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    table = [{k: r[k] for k in ("case_id", "standalone_diagnosis", "standalone_correct", "doctor_initial_diagnosis", "doctor_initial_correct",
                                "final_diagnosis", "collaboration_correct", "num_turns", "stop", "collaboration_state", "source")} for r in first.values()]
    (out / "pool_cases.json").write_text(json.dumps(sorted(table, key=lambda r: r["case_id"]), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("screen", "robust", "audit", "report"), required=True)
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    {"screen": screen, "robust": robust, "audit": audit, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
