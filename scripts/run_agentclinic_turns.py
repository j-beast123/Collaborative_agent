"""Effectiveness labels of the main protocol: every Agent turn a base collaboration reached is an intervention point."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from scripts.run_agentclinic import build_runtime, select, write_split_files, candidate_generator
from dataclasses import replace
from scripts.run_agentclinic_a2 import information_section, sanity
from scripts.run_base_collaborations import FROZEN, identity
from src.agentclinic.adapter import agent_view, assert_no_leakage, render
from src.agentclinic.adapter import load_cases
from src.agentclinic.selection import load_manifest
from src.agentclinic.outcome import normalize
from src.agentclinic.pipeline import build_labels, generate_candidates, integrity_check, turn_histories
from src.agentclinic.protocol import agent_prompt, doctor_prompt, protocol_spec
from src.common.config import read_config
from src.common.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def select_train_dev(config: dict) -> tuple[list[dict], dict]:
    """Read only TRAIN/DEV case objects; TEST content is never deserialized."""
    dataset = config["dataset"]
    manifest = load_manifest(ROOT / dataset["manifest"], ROOT / dataset["path"])
    splits = {cid: row["split"] for cid, row in manifest["cases"].items()}
    allowed = {cid for cid, split in splits.items() if split in ("train", "dev")}
    cases = load_cases(ROOT / dataset["path"], (config.get("doctor_view") or {}).get("patient_fields"), case_ids=allowed)
    if {c["case_id"] for c in cases} != allowed:
        raise ValueError("missing TRAIN/DEV source cases")
    for case in cases:
        case["split"] = splits[case["case_id"]]
        assert_no_leakage(case)
    return cases, {"splits": splits}


def frozen_protocol(config: dict) -> dict:
    """The run's frozen protocol; labels are only made under it."""
    path = ROOT / config["output_dir"] / FROZEN
    if not path.is_file():
        raise SystemExit(f"{path} does not exist: freeze the protocol first (scripts/run_base_collaborations.py freeze)")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen["protocol_identity"] != identity(config):
        raise SystemExit("the config no longer gives the frozen protocol identity (a prompt, model, decoding or the dataset changed)")
    for name in ("histories", "base_trajectories"):
        if frozen.get(f"{name}_sha256") and hashlib.sha256((path.parent / f"{name}.jsonl").read_bytes()).hexdigest() != frozen[f"{name}_sha256"]:
            raise SystemExit(f"the frozen {name} artifact changed")
    return frozen


def scope_histories(config: dict, cases: dict[str, dict], scope: str) -> list[dict]:
    trajectories = [t for t in read_jsonl(ROOT / config["output_dir"] / "base_trajectories.jsonl") if t["case_id"] in cases]
    if scope == "smoke":
        smoke = config["smoke"]
        pool = sorted(t["case_id"] for t in trajectories if cases[t["case_id"]]["split"] == smoke["split"])
        chosen = set(random.Random(smoke["seed"]).sample(pool, min(smoke["cases"], len(pool))))
        trajectories = [t for t in trajectories if t["case_id"] in chosen]
    return turn_histories(trajectories)


def leakage_audit(out: Path, cases: dict[str, dict], histories: list[dict], spec) -> dict:
    """Correct_Diagnosis reaches nothing but the outcome judgment: not an Agent prompt, not the Doctor's decision or
    final prompt, not a candidate instruction, not the predictor input.
    """
    by_history = {h["history_id"]: h for h in histories}
    labels = [r for r in read_jsonl(out / "labels.jsonl", missing_ok=True) if r["history_id"] in by_history]
    attempts = [a for a in read_jsonl(out / "candidate_attempts.jsonl", missing_ok=True) if a["history_id"] in by_history]
    problems = []
    for row in histories:
        case, gold, turn = cases[row["case_id"]], normalize(cases[row["case_id"]]["gold"]), row["intervention_turn"]
        assert_no_leakage(case)
        after = row["history"] + [{"speaker": "agent", "stage": turn, "text": "x"}]
        doctor_stage = "FINAL" if int(turn[1:]) == spec.max_turns else "NEXT"
        for name, prompt in ((f"{turn} prompt", agent_prompt(case, row["history"], turn, spec=spec)),
                             (f"Doctor prompt after {turn}", doctor_prompt(case, after, doctor_stage))):
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
        case, source, history = cases[label["case_id"]], by_history[label["history_id"]], label["history"]
        if history["scenario"] != render(agent_view(case)) or normalize(case["gold"]) in normalize(history["scenario"]):
            problems.append(f"{label['history_id']}: predictor scenario is not the Agent's view or names the correct diagnosis")
        if [t["text"] for t in history["turns"]] != [m["text"] for m in source["history"]] \
                or history.get("intervention_turn") != source["intervention_turn"] or label.get("intervention_turn") != source["intervention_turn"]:
            problems.append(f"{label['history_id']}: predictor history is not the dialogue before its turn, marked with that turn")
        if any(k in json.dumps(history) for k in ("Correct_Diagnosis", "final_diagnosis", "ground_truth")):
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


def turn_report(out: Path, cases: dict[str, dict], histories: list[dict], k: int, names: list[str], max_turns: int) -> dict:
    """Informativeness of the labels on TRAIN, overall and per turn. DEV is counted and not summarized: no statistic
    of DEV outcomes is produced before the predictor's one evaluation."""
    train = [h for h in histories if cases[h["case_id"]]["split"] == "train"]
    dev = {h["history_id"] for h in histories if cases[h["case_id"]]["split"] == "dev"}
    labels = read_jsonl(out / "labels.jsonl", missing_ok=True)
    turns = [f"A{t}" for t in range(1, max_turns + 1)]
    lengths = defaultdict(list)  # how long the collaboration ran after the intervention (TRAIN, AFTER continuations)
    wanted = {h["history_id"]: h["intervention_turn"] for h in train}
    for row in read_jsonl(out / "after_rollouts.jsonl", missing_ok=True):
        if row["history_id"] in wanted and row["status"] == "completed" and row.get("turns"):
            lengths[wanted[row["history_id"]]].append(row["turns"])
    return {"scope_histories": len(histories), "k": k,
            "train": sanity(out, train, k, names),
            "train_by_turn": {turn: dict(sanity(out, members, k, names),
                                         mean_final_length_after_intervention=sum(lengths[turn]) / len(lengths[turn]) if lengths[turn] else None)
                              for turn in turns if (members := [h for h in train if h["intervention_turn"] == turn])},
            "dev_counts_only": {"histories": len(dev), "labelled_pairs": sum(r["history_id"] in dev for r in labels),
                                "histories_by_turn": {turn: sum(h["intervention_turn"] == turn for h in histories if h["history_id"] in dev)
                                                      for turn in turns}}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("stages", nargs="+", choices=("candidates", "labels", "report"))
    parser.add_argument("--allow-stand-ins", action="store_true",
                        help="full scope although the decided candidate generator / compliance judge is not connected")
    parser.add_argument("--shard", help="I/N: make only the histories of shard I (0-based) of N, in its own directory")
    args = parser.parse_args()
    if args.shard and (args.scope != "full" or "report" in args.stages):
        parser.error("--shard goes with --scope full and the stages candidates / labels; report runs after the merge")
    config = read_config(args.config)
    # execution settings only (how many continuations run at once, how many Agent prompts share a GPU batch): they change
    # the speed, not what is generated, and are not part of the protocol identity. The values used are in run_config_<scope>.json.
    for key, variable in (("workers", "LABEL_WORKERS"), ("agent_batch_size", "LABEL_AGENT_BATCH_SIZE")):
        if os.environ.get(variable):
            config["labels"][key] = int(os.environ[variable])
    if args.scope == "full" and set(args.stages) & {"candidates", "labels"}:
        from src.common.main_config import refuse_stand_ins

        refuse_stand_ins(config, ["effectiveness_candidate_generator", "effectiveness_compliance_judge"],
                         "full Effectiveness labels", args.allow_stand_ins)
    spec = protocol_spec(config["protocol"])
    frozen = frozen_protocol(config)
    selected, dataset = select_train_dev(config) if config["dataset"].get("manifest") else select(config, "full")
    if any(c["split"] not in ("train", "dev") for c in selected):
        raise SystemExit("a case outside TRAIN and DEV was selected")
    cases = {c["case_id"]: c for c in selected}
    out = ROOT / config["labels_dir"]
    out.mkdir(parents=True, exist_ok=True)
    histories = scope_histories(config, cases, args.scope)
    shard = None
    if args.shard:
        from src.agentclinic.shards import assign, seed

        index, total = (int(part) for part in args.shard.split("/"))
        if not 0 <= index < total:
            raise SystemExit("--shard I/N needs 0 <= I < N")
        shard_of = assign(histories, total)
        out = seed(out, index, total, shard_of, {"protocol_identity": frozen["protocol_identity"], "scope": args.scope})
        histories = [h for h in histories if shard_of[h["history_id"]] == index]
        shard = {"shard": index, "shards": total, "directory": str(out.relative_to(ROOT))}
    names = tuple(config["candidates"]["names"])
    known = {r["history_id"]: r for r in read_jsonl(out / "histories_turns.jsonl", missing_ok=True)}
    if any(known[h["history_id"]]["history"] != h["history"] for h in histories if h["history_id"] in known):
        raise SystemExit("a history differs from the one already labelled in this directory")
    write_jsonl(out / "histories_turns.jsonl", sorted({**known, **{h["history_id"]: h for h in histories}}.values(), key=lambda r: r["history_id"]))
    (out / f"run_config_{args.scope}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    results = {"scope": args.scope, **(shard or {}), "protocol_identity": frozen["protocol_identity"], "histories": len(histories),
               "by_split": {s: sum(cases[h["case_id"]]["split"] == s for h in histories) for s in ("train", "dev")},
               "by_turn": {f"A{t}": sum(h["intervention_turn"] == f"A{t}" for h in histories) for t in range(1, spec.max_turns + 1)}}
    workers = config["labels"]["workers"]
    if any(stage in ("candidates", "labels") for stage in args.stages):
        runtime_config = copy.deepcopy(config)
        runtime_config["output_dir"] = str(out.relative_to(ROOT))  # the label directory, or the directory of the shard
        runtime, diagnosis, compliance = build_runtime(runtime_config)
        try:
            def timed(stage: str, started: float) -> None:  # wall time per stage, for the smoke report's estimate
                with (out / f"timing_{args.scope}.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"stage": stage, "scope": args.scope, "histories": len(histories), "started": started,
                                             "ended": time.time(), "seconds": time.time() - started}) + "\n")

            if "candidates" in args.stages:
                started = time.time()
                generator = candidate_generator(config)  # label_roles.candidate_generator, else the Agent model
                results["candidate_generator"] = (config["label_roles"]["candidate_generator"]["model_id"] if generator
                                                  else "agent model (stand-in)")
                results["candidates"] = generate_candidates(cases, histories, replace(runtime, agent=generator) if generator else runtime,
                                                            compliance, out, workers,
                                                            max_attempts=config["candidates"]["max_attempts"], stage=None,
                                                            names=names, spec=spec)
                timed("candidates", started)
            if "labels" in args.stages:
                started = time.time()
                results["labels"] = build_labels(
                    cases, histories, runtime, diagnosis, out, k=config["k"], workers=workers,
                    require_compliance=config["labels"]["require_compliance"],
                    max_attempts=config["labels"]["max_rollout_attempts"], stage=None, spec=spec,
                    metadata={"version": config["version"], "k": config["k"], "models": config["models"], "decoding": config["decoding"],
                              "protocol": spec.name, "protocol_identity": frozen["protocol_identity"],
                              "continuation_protocol_hash": frozen["protocol_identity"],
                              "dataset_manifest": config["dataset"].get("manifest"), "split": "mixed"})
                timed("labels", started)
        finally:
            if runtime.close:
                runtime.close()
    if "report" in args.stages:
        report = turn_report(out, cases, histories, config["k"], list(names), spec.max_turns)
        audit = leakage_audit(out, cases, histories, spec)
        (out / f"turn_sanity_{args.scope}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / f"turn_leakage_audit_{args.scope}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        results["sanity"] = {"train": {key: report["train"][key] for key in ("histories", "labelled_pairs", "nonzero_delta_pairs",
                                                                               "nonzero_delta_pair_share", "histories_with_candidate_ordering",
                                                                               "before_floor_share", "before_ceiling_share", "compliant_share")},
                             "train_by_turn": {turn: {key: row[key] for key in ("histories", "labelled_pairs", "nonzero_delta_pairs",
                                                                                 "histories_with_candidate_ordering")}
                                               for turn, row in report["train_by_turn"].items()},
                             "dev_counts_only": report["dev_counts_only"]}
        results["leakage_audit"] = {key: value for key, value in audit.items() if key != "problems"}
        stop = [] if audit["passed"] else [f"leakage audit failed: {audit['problems'][:3]}"]
        if args.scope == "full":
            results["label_files"] = write_split_files(out, dataset, config)
            results["integrity"] = integrity_check(out, k=config["k"], splits=dataset["splits"], thresholds=config["integrity"])
            if not results["integrity"]["passed"]:
                stop.append("integrity check failed: " + ", ".join(results["integrity"]["failures"]))
        results["proceed"], results["stop_reasons"] = not stop, stop
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    if results.get("stop_reasons"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
