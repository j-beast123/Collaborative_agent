"""An environment that differs from its parent only in how a final diagnosis is judged (the equivalence prompt and/or the
judge model, and a human-adjudicated mapping): every generated text of the parent is kept and only outcomes are read again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.run_base_collaborations import FROZEN, identity, identity_parts
from src.agentclinic.outcome import DiagnosisJudge, normalize
from src.common.config import read_config
from src.common.io import read_jsonl
from src.evaluation.collaboration_states import STATES
from src.judges.llm import openrouter_chat

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def differing(a, b, prefix="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [name for key in sorted(set(a) | set(b)) for name in differing(a.get(key), b.get(key), f"{prefix}{key}.")]
    return [] if a == b else [prefix.rstrip(".")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--human-mapping", required=True)
    parser.add_argument("--decisions", help="decisions made before by this environment's judge under its policy (scripts/majority_equivalence.py); "
                                            "pairs in it are not judged again")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    config, parent = read_config(args.config), read_config(args.parent)
    changed = differing(identity_parts(config), identity_parts(parent))
    if not changed or any(not (name == "prompts.diagnosis_equivalence" or name.startswith("models.judge")) for name in changed):
        raise SystemExit(f"the environments must differ in the diagnosis judge only (differences: {changed})")
    source, out = ROOT / parent["output_dir"], ROOT / config["output_dir"]
    parent_labels, labels = ROOT / parent["labels_dir"], ROOT / config["labels_dir"]
    decisions_path = ROOT / config["diagnosis_equivalence_frozen"]
    if out.exists():
        raise SystemExit(f"{out} exists: a clean environment is derived once")
    frozen = json.loads((source / FROZEN).read_text(encoding="utf-8"))
    for name in ("histories", "base_trajectories"):
        if sha(source / f"{name}.jsonl") != frozen[f"{name}_sha256"]:
            raise SystemExit(f"the parent's frozen {name} changed")
    human = {(r["reference"], r["predicted"]): bool(r["equivalent"]) for r in read_jsonl(ROOT / args.human_mapping)}
    base = read_jsonl(source / "base_trajectories.jsonl")
    rollouts = {name: read_jsonl(parent_labels / name) for name in ("before_rollouts.jsonl", "after_rollouts.jsonl")}

    # every (correct diagnosis, stated diagnosis) pair whose equivalence is read anywhere
    wording = defaultdict(Counter)
    def need(gold, stated):
        if stated and stated.strip() and normalize(gold) != normalize(stated):
            wording[(normalize(gold), normalize(stated))][(gold, stated)] += 1
    for t in base:
        gold = t["ground_truth"]["correct_diagnosis"]
        for stated in (t.get("final_diagnosis"), t.get("agent_initial_position"), t.get("doctor_initial_claim")):
            need(gold, stated)
    for rows in rollouts.values():
        for r in rows:
            need(r["gold"], r.get("final_diagnosis"))
    judge = DiagnosisJudge(openrouter_chat(config["models"]["judge"]["model_id"], None),
                           (ROOT / config["prompts"]["diagnosis_equivalence"]).read_text(encoding="utf-8"),
                           params=config.get("diagnosis_judge_params"))
    given = read_jsonl(ROOT / args.decisions) if args.decisions else []
    known = {(r["reference"], r["predicted"]) for r in given}
    judge.policy = config.get("diagnosis_judge_policy")  # a pair not decided before is decided under the same policy
    todo = sorted(key for key in wording if key not in human and key not in known)

    def decide(key):
        gold, stated = wording[key].most_common(1)[0][0]
        for _ in range(3):
            verdict = judge.judge(gold, stated)
            if isinstance(verdict["equivalent"], bool):
                return {"reference": key[0], "predicted": key[1], "equivalent": verdict["equivalent"], "source": "judge",
                        "reason": verdict.get("reason"), "raw_reference": gold, "raw_predicted": stated,
                        **({"votes": verdict["votes"], "policy": verdict["policy"]} if verdict.get("votes") else {})}
        raise SystemExit(f"the judge gave no decision for {key}")

    with ThreadPoolExecutor(args.workers) as pool:
        automatic = list(pool.map(decide, todo))
    out.mkdir(parents=True)
    human_rows = read_jsonl(ROOT / args.human_mapping)
    write_rows(decisions_path, human_rows + given + automatic)
    decision = {**{(r["reference"], r["predicted"]): r["equivalent"] for r in given + automatic}, **human}

    def correct(gold, stated):
        if not (stated or "").strip():
            return None, None
        if normalize(gold) == normalize(stated):
            return True, "exact"
        return decision[(normalize(gold), normalize(stated))], "frozen"

    new_identity, flips = identity(config), Counter()
    trajectories = []
    for t in base:
        gold = t["ground_truth"]["correct_diagnosis"]
        final_ok, method = correct(gold, t.get("final_diagnosis"))
        agent_ok, doctor_ok = correct(gold, t.get("agent_initial_position"))[0], correct(gold, t.get("doctor_initial_claim"))[0]
        row = dict(t, is_correct=bool(final_ok), outcome_method=method or "empty", agent_initial_correct=agent_ok, doctor_initial_correct=doctor_ok,
                   collaboration_state=STATES.get((agent_ok, doctor_ok)), protocol_identity=new_identity,
                   in_parent={k: t.get(k) for k in ("is_correct", "agent_initial_correct", "doctor_initial_correct", "collaboration_state", "protocol_identity")})
        flips[(t["split"], "final outcome")] += row["is_correct"] != t["is_correct"]
        flips[(t["split"], "collaboration state")] += row["collaboration_state"] != t.get("collaboration_state")
        trajectories.append(row)
    shutil.copyfile(source / "histories.jsonl", out / "histories.jsonl")
    write_rows(out / "base_trajectories.jsonl", trajectories)
    (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    labels.mkdir(parents=True)
    copied = []
    for name in ("candidates.jsonl", "candidate_attempts.jsonl", "candidate_compliance.jsonl", "histories_turns.jsonl"):
        shutil.copyfile(parent_labels / name, labels / name)
        copied.append(name)
    split = {t["case_id"]: t["split"] for t in base}
    k = config["k"]
    for name, rows in rollouts.items():
        relabelled = []
        for r in rows:
            ok, method = correct(r["gold"], r.get("final_diagnosis"))
            new = dict(r, success=bool(ok), outcome_method=method or "empty", success_in_parent=r["success"],
                       outcome_reason="decision of this environment's diagnosis-equivalence mapping")
            new.pop("success_before_freeze", None)
            flips[(split[r["case_id"]], name.split("_")[0] + " continuation")] += new["success"] != bool(r["success"])
            relabelled.append(new)
        rollouts[name] = relabelled
        write_rows(labels / name, relabelled)
    before, after = defaultdict(list), defaultdict(list)
    for r in rollouts["before_rollouts.jsonl"]:
        before[r["history_id"]].append(r)
    for r in rollouts["after_rollouts.jsonl"]:
        after[(r["history_id"], r["candidate_id"], r.get("candidate_attempt", 1))].append(r)
    new_labels = []
    for label in read_jsonl(parent_labels / "labels.jsonl"):
        b = [r for r in before[label["history_id"]] if r["status"] == "completed"]
        a = [r for r in after[(label["history_id"], label["candidate_id"], label.get("candidate_attempt", 1))] if r["status"] == "completed"]
        if len(a) != k or len(b) != k:
            raise SystemExit(f"{label['history_id']}/{label['candidate_id']}: not {k} continuations on both sides")
        counts = {"before": {"successes": sum(r["success"] for r in b), "trials": k}, "after": {"successes": sum(r["success"] for r in a), "trials": k}}
        row = dict(label, **counts, before_rate=counts["before"]["successes"] / k, after_rate=counts["after"]["successes"] / k,
                   metadata=dict(label["metadata"], protocol_identity=new_identity, continuation_protocol_hash=new_identity))
        row["delta_effectiveness"] = row["after_rate"] - row["before_rate"]
        flips[(label["split"], "label counts")] += (row["before"], row["after"]) != (label["before"], label["after"])
        flips[(label["split"], "label delta")] += row["delta_effectiveness"] != label["delta_effectiveness"]
        new_labels.append(row)
    write_rows(labels / "labels.jsonl", new_labels)
    report = {
        "derived": time.strftime("%Y-%m-%d %H:%M:%S"), "procedure": "scripts/derive_clean_environment.py", "parent_config": args.parent,
        "parent_protocol_identity": identity(parent), "protocol_identity": new_identity, "identity_parts_that_differ": changed,
        "judge": {"model": config["models"]["judge"]["model_id"], "prompt": config["prompts"]["diagnosis_equivalence"],
                  "prompt_sha256": sha(ROOT / config["prompts"]["diagnosis_equivalence"]), "params": judge.params},
        "human_mapping": args.human_mapping, "human_mapping_sha256": sha(ROOT / args.human_mapping), "human_decisions": len(human),
        "pairs_needed": len(wording), "pairs_decided_by_a_human": sum(key in human for key in wording), "pairs_decided_by_the_judge": len(automatic),
        "pairs_taken_from_given_decisions": sum(key in known for key in wording), "given_decisions": args.decisions,
        "given_decisions_sha256": sha(ROOT / args.decisions) if args.decisions else None, "outcome_judge_policy": config.get("diagnosis_judge_policy"),
        "decisions_file": config["diagnosis_equivalence_frozen"], "decisions_sha256_at_derivation": sha(decisions_path),
        "reused_unchanged": ["histories.jsonl (D1)", "every dialogue and final diagnosis of base_trajectories.jsonl",
                             "every dialogue and final diagnosis of before_rollouts.jsonl and after_rollouts.jsonl", *copied],
        "relabelled": ["base_trajectories.jsonl: is_correct, agent_initial_correct, doctor_initial_correct, collaboration_state",
                       "before_rollouts.jsonl / after_rollouts.jsonl: success", "labels.jsonl: before, after, rates, delta"],
        "changed": {f"{split_name}: {what}": n for (split_name, what), n in sorted(flips.items())},
        "train_base": {"correct": sum(t["is_correct"] for t in trajectories if t["split"] == "train"),
                       "correct_in_parent": sum(t["is_correct"] for t in base if t["split"] == "train"), "cases": sum(t["split"] == "train" for t in base)},
        "nonzero_delta_labels": dict(Counter(l["split"] for l in new_labels if l["delta_effectiveness"] != 0)),
        "nonzero_delta_labels_in_parent": dict(Counter(l["split"] for l in read_jsonl(parent_labels / "labels.jsonl") if l["delta_effectiveness"] != 0))}
    (out / "provenance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "clean_derivation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
