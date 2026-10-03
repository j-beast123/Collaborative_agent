"""Base-policy collaborations of TRAIN and DEV under a protocol, and how far they go."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from scripts.run_agentclinic import build_runtime, select
from src.agentclinic.pipeline import Appender
from src.agentclinic.protocol import NEUTRAL_AGENT_TASK, NEXT_STEP, doctor_opening, protocol_spec
from src.agentclinic.trajectory import generate_trajectories
from src.common.config import read_config
from src.common.io import read_jsonl
from src.data.schema import digest

ROOT = Path(__file__).resolve().parents[1]


def identity_parts(config: dict) -> dict:
    """What defines these collaborations: protocol, transition instruction, models, decoding, prompts, D1 source."""
    prompts = {name: digest((ROOT / path).read_text(encoding="utf-8")) for name, path in sorted(config["prompts"].items())}
    return ({"protocol": {k: config["protocol"][k] for k in ("max_turns", "early_stopping")}, "next_step": digest(NEXT_STEP),
                   "agent_task": digest(NEUTRAL_AGENT_TASK),
                   "models": config["models"], "decoding": config["decoding"], "prompts": prompts,
                   "doctor_view": config.get("doctor_view"),
                   **({"source_histories_sha256": sha256(ROOT / config["source_histories"])} if config.get("source_histories")
                      else {"dataset_manifest_sha256": sha256(ROOT / config["dataset"]["manifest"])})})


def identity(config: dict) -> str:
    return digest(identity_parts(config))[:16]


FROZEN = "protocol.frozen.json"


def freeze(config: dict, out: Path) -> dict:
    """Fix the protocol after the TRAIN reach report. Only degeneracy of the stopping rule blocks it."""
    target = out / FROZEN
    if target.exists():
        frozen = json.loads(target.read_text())
        if frozen["protocol_identity"] != identity(config):
            raise SystemExit(f"{target} was frozen with another protocol identity")
        return frozen
    report = json.loads((out / "turn_reach_report.json").read_text())
    if report["protocol_identity"] != [identity(config)]:
        raise SystemExit("the reach report was made under another protocol identity")
    if report["train_collaborations"] != report["train_cases"]:
        raise SystemExit("the reach report does not cover every TRAIN case")
    degenerate = [name for name in ("almost_all_stop_after_A1", "almost_all_run_to_the_last_turn") if report["degeneracy"][name]]
    if degenerate:
        raise SystemExit(f"the stopping rule is degenerate on TRAIN ({degenerate}): the protocol is not frozen")
    frozen = {"protocol": report["protocol"], "protocol_identity": identity(config), "identity_parts": identity_parts(config),
              "frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "config_version": config["version"],
              "dataset_manifest": config["dataset"].get("manifest"),
              "histories_sha256": sha256(out / "histories.jsonl") if (out / "histories.jsonl").is_file() else None,
              "base_trajectories_sha256": sha256(out / "base_trajectories.jsonl"),
              "basis": {"scope": "TRAIN reach report (turn_reach_report.json); DEV was not read", "degeneracy": report["degeneracy"],
                        "criterion": "frozen unless almost all TRAIN collaborations stop after A1 or almost all run to the last turn; "
                                     "no prompt was tuned toward a reach distribution"}}
    target.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    target.chmod(0o444)
    return frozen



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(config: dict) -> dict:
    """What is taken from the source run instead of being generated again, and the comparison that allows it."""
    if not config.get("source_histories"):
        return {"source_run": None, "d1": {"reused_from_source": False, "generated_here": "histories.jsonl"},
                "agent_standalone_assessment": {"reused_from_source": False},
                "generated_here": "D1, every later message, the Agent-side states and the standalone assessment",
                "dataset_manifest": config["dataset"].get("manifest"), "protocol_identity": identity(config)}
    source_dir = (ROOT / config["source_histories"]).parent
    source, manifest = json.loads((source_dir / "run_config.json").read_text()), json.loads((source_dir / "run_manifest.json").read_text())
    doctor = {"model": config["models"]["doctor"] == source["models"]["doctor"],
              "decoding": config["decoding"]["doctor"] == source["decoding"]["doctor"],
              "system_prompt": sha256(ROOT / config["prompts"]["doctor"]) == manifest["prompt_sha256"]["doctor"],
              "doctor_view": config.get("doctor_view") == source.get("doctor_view"),
              "dataset": sha256(ROOT / config["dataset"]["path"]) == manifest["dataset_sha256"], "split": config["split"] == source["split"]}
    agent = {"model": config["models"]["agent"] == source["models"]["agent"], "decoding": config["decoding"]["agent"] == source["decoding"]["agent"],
             "system_prompt": sha256(ROOT / config["prompts"]["consultant"]) == manifest["prompt_sha256"]["consultant"]}
    if not all(doctor.values()):
        raise SystemExit(f"D1 cannot be reused from {source_dir}: {[k for k, same in doctor.items() if not same]} differ")
    return {"source_run": str(source_dir.relative_to(ROOT)), "source_histories": config["source_histories"],
            "source_histories_sha256": sha256(ROOT / config["source_histories"]), "source_version": source["version"],
            "d1": {"reused_from_source": True, "identical": doctor,
                   "why": "the Doctor's opening message has the same model, decoding, system prompt, case view and prompt text in both protocols"},
            "agent_standalone_assessment": {"reused_from_source": all(agent.values()), "identical": agent,
                                            "why": "reused only when the Agent's model, decoding and system prompt are the source run's; "
                                                   "otherwise asked again under this protocol's system prompt"},
            "generated_here": "every message after D1, the Agent-side states, and the standalone assessment when not reused",
            "protocol_identity": identity(config)}


def opening_histories(cases: dict[str, dict], runtime, path: Path, workers: int) -> list[dict]:
    """D1 of every case, written once: a case that has its D1 keeps it."""
    log = Appender(path, key=lambda r: r["case_id"])

    def run(case):
        history = doctor_opening(case, runtime)
        log.add({"history_id": f"{case['case_id']}:A1", "case_id": case["case_id"], "split": case["split"], "history": history,
                 "doctor_provisional_diagnosis": history[0]["provisional_diagnosis"]})

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, [c for c in cases.values() if c["case_id"] not in log.keys]))
    return sorted((r for r in log.rows if r["case_id"] in cases), key=lambda r: r["case_id"])


def generate(config: dict, cases: dict[str, dict], out: Path) -> dict:
    spec = protocol_spec(config["protocol"])
    origin = provenance(config)
    (out / "provenance.json").write_text(json.dumps(origin, indent=2), encoding="utf-8")
    runtime, judge, _ = build_runtime(config)
    runtime.state_system = (ROOT / config["prompts"]["state"]).read_text(encoding="utf-8")
    try:
        histories = ([h for h in read_jsonl(ROOT / config["source_histories"]) if h["case_id"] in cases] if config.get("source_histories")
                     else opening_histories(cases, runtime, out / "histories.jsonl", config["labels"]["workers"]))
        rows = generate_trajectories(cases, histories, runtime, judge, out / "base_trajectories.jsonl", suffix="base",
                                     workers=config["labels"]["workers"], max_attempts=config["base_trajectories"]["max_attempts"],
                                     spec=spec, reuse_standalone=origin["agent_standalone_assessment"]["reused_from_source"],
                                     iteration=0, checkpoint_id="base",
                                     protocol_identity=identity(config))
    finally:
        if runtime.close:
            runtime.close()
    return {"trajectories": len(rows), "protocol": spec.name, "protocol_identity": identity(config)}


def reach(rows: list[dict], max_turns: int) -> dict:
    valid = [t for t in rows if t["trajectory_status"] == "valid"]

    def share(values) -> float | None:
        values = list(values)
        return mean(values) if values else None

    return {
        "trajectories": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid),
        "invalid_rate": 1 - len(valid) / len(rows) if rows else None,
        "invalid_statuses": dict(Counter(t["status"] for t in rows if t["trajectory_status"] != "valid")),
        "reached": {f"A{turn}": sum(t["num_turns"] >= turn for t in valid) for turn in range(1, max_turns + 1)},
        "stopped_after": {f"A{turn}": sum(t["num_turns"] == turn for t in valid) for turn in range(1, max_turns + 1)},
        "stop_reason": dict(Counter(str(t.get("stop")) for t in valid)),
        "mean_turns": share(t["num_turns"] for t in valid),
        "final_accuracy": share(t["is_correct"] for t in valid),
        "final_accuracy_by_stop_turn": {f"A{turn}": {"n": len(members), "accuracy": share(t["is_correct"] for t in members)}
                                        for turn in range(1, max_turns + 1)
                                        if (members := [t for t in valid if t["num_turns"] == turn])},
        "attempts_needed": dict(Counter(t.get("attempts", 1) for t in rows)),
    }


def report(config: dict, cases: dict[str, dict], out: Path) -> dict:
    """TRAIN only. The DEV collaborations are counted and not read."""
    spec = protocol_spec(config["protocol"])
    everything = [t for t in read_jsonl(out / "base_trajectories.jsonl") if t["case_id"] in cases]
    rows = [t for t in everything if t["split"] == "train"]
    earlier = ({h["case_id"]: h["natural"] for h in read_jsonl(ROOT / config["source_histories"]) if h["case_id"] in cases}
               if config.get("source_histories") else {})  # the fixed two-turn run on the same cases, when this run starts from its D1
    valid = [t for t in rows if t["trajectory_status"] == "valid"]
    same_cases = [t for t in valid if t["case_id"] in earlier and earlier[t["case_id"]]["status"] == "completed"
                  and earlier[t["case_id"]]["success"] is not None]
    train = reach(rows, spec.max_turns)
    share_a1 = train["stopped_after"]["A1"] / train["valid"] if train["valid"] else None
    share_max = train["stopped_after"][f"A{spec.max_turns}"] / train["valid"] if train["valid"] else None
    result = {
        "protocol": spec.name, "max_turns": spec.max_turns, "early_stopping": spec.early_stopping,
        "protocol_identity": sorted({t.get("protocol_identity") for t in everything}),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": "TRAIN cases only",
        "dataset_manifest": config["dataset"].get("manifest"),
        "train_cases": sum(c["split"] == "train" for c in cases.values()), "train_collaborations": len(rows),
        "dev_collaborations_generated_and_not_read": len(everything) - len(rows),
        "train": train,
        "train_by_collaboration_state": {state: reach([t for t in rows if str(t["collaboration_state"]) == state], spec.max_turns)
                                         for state in sorted({str(t["collaboration_state"]) for t in rows})},
        # is the stopping rule working at all? (not a target distribution)
        "degeneracy": {"share_stopped_after_A1": share_a1, f"share_reaching_A{spec.max_turns}": share_max,
                       "almost_all_stop_after_A1": share_a1 is not None and share_a1 >= 0.95,
                       "almost_all_run_to_the_last_turn": share_max is not None and share_max >= 0.95},
        # the same TRAIN cases under the first label run's fixed two-turn protocol (its one natural collaboration per case)
        "compared_with_the_fixed_two_turn_run_on_train": None if not earlier else {
            "cases": len(same_cases),
            "accuracy_this_protocol": mean(t["is_correct"] for t in same_cases) if same_cases else None,
            "accuracy_fixed_two_turn": mean(bool(earlier[t["case_id"]]["success"]) for t in same_cases) if same_cases else None,
            "right_here_wrong_there": sum(t["is_correct"] and not earlier[t["case_id"]]["success"] for t in same_cases),
            "wrong_here_right_there": sum(not t["is_correct"] and bool(earlier[t["case_id"]]["success"]) for t in same_cases)},
        "note": "One natural collaboration per case: the reach of a case varies from sample to sample. An Effectiveness label, "
                "a GRPO sample and a rubric can only be made for a turn that is reached. DEV is not summarized here so that no "
                "decision about the protocol can depend on it.",
    }
    (out / "turn_reach_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    turns = [f"A{t}" for t in range(1, spec.max_turns + 1)]

    def line(name: str, r: dict) -> str:
        accuracy = "n/a" if r["final_accuracy"] is None else f"{r['final_accuracy']:.3f}"
        return f"| {name} | {r['valid']} | " + " | ".join(str(r["reached"][t]) for t in turns) + f" | {accuracy} | {r['invalid']} |"

    lines = [f"# Turn reach of the base Agent on TRAIN: {spec.name}", "",
             f"{len(rows)} TRAIN collaborations, one per case ({len(everything) - len(rows)} DEV collaborations exist and are not read here). "
             "Reached A_t: valid collaborations with at least t Agent turns.", "",
             "| | valid | " + " | ".join(f"reached {t}" for t in turns) + " | final accuracy | invalid |",
             "|---|---|" + "---|" * len(turns) + "---|---|",
             line("TRAIN", train), *[line(f"state {state}", r) for state, r in result["train_by_collaboration_state"].items()], "",
             ("Stopped after: " + ", ".join(f"{t} {train['stopped_after'][t]}" for t in turns)
              + f". Stop reason: {train['stop_reason']}. Mean turns {train['mean_turns']:.2f}.") if valid else "", "",
             f"Accuracy by stop turn: {json.dumps(train['final_accuracy_by_stop_turn'])}", "",
             f"Degeneracy check: {json.dumps(result['degeneracy'])}", "",
             f"Same TRAIN cases under the fixed two-turn run: {json.dumps(result['compared_with_the_fixed_two_turn_run_on_train'])}", ""]
    (out / "turn_reach_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("stages", nargs="+", choices=("generate", "report", "freeze"))
    args = parser.parse_args()
    config = read_config(args.config)
    if config["dataset"].get("manifest"):
        from scripts.run_agentclinic_turns import select_train_dev
        selected, _ = select_train_dev(config)
    else:
        selected, _ = select(config, "full")
    if any(c["split"] not in ("train", "dev") for c in selected):
        raise SystemExit("a case outside TRAIN and DEV was selected")
    cases = {c["case_id"]: c for c in selected}
    out = ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    results = {}
    if "generate" in args.stages:
        if (out / FROZEN).exists():
            raise SystemExit(f"{out / FROZEN} exists: the base collaborations of a frozen protocol are not generated again")
        results["generate"] = generate(config, cases, out)
    if "report" in args.stages:
        results["report"] = {k: v for k, v in report(config, cases, out).items() if k in ("train", "degeneracy", "compared_with_the_fixed_two_turn_run_on_train")}
    if "freeze" in args.stages:
        results["freeze"] = {k: v for k, v in freeze(config, out).items() if k != "identity_parts"}
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
