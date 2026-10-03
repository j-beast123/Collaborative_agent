"""The base collaborations of an environment whose dataset version is a TRAIN subset of another environment's
(scripts/freeze_train_subset.py), taken from that environment instead of being generated again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from scripts.run_agentclinic_turns import select_train_dev
from scripts.run_base_collaborations import FROZEN, identity, identity_parts
from src.agentclinic.pipeline import turn_histories
from src.agentclinic.shards import JUDGE_CACHE, KEYS, write_rows
from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()
    config, parent = read_config(args.config), read_config(args.parent)
    mine, theirs = identity_parts(config), identity_parts(parent)
    differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    if differing != ["dataset_manifest_sha256"]:
        raise SystemExit(f"the environments differ in more than the dataset version: {differing}")
    source, out = ROOT / parent["output_dir"], ROOT / config["output_dir"]
    frozen = json.loads((source / FROZEN).read_text(encoding="utf-8"))
    if frozen["protocol_identity"] != identity(parent):
        raise SystemExit("the parent environment is not frozen under its config")
    for name in ("histories", "base_trajectories"):
        if sha(source / f"{name}.jsonl") != frozen[f"{name}_sha256"]:
            raise SystemExit(f"the parent's frozen {name} changed")
    cases, _ = select_train_dev(config)
    parent_cases = {c["case_id"]: c["split"] for c in select_train_dev(parent)[0]}
    cases = {c["case_id"]: c for c in cases}
    if any(parent_cases.get(cid) != case["split"] for cid, case in cases.items()):
        raise SystemExit("a case of this dataset version is not in the parent's with the same split")
    result = {"cases": len(cases), "by_split": {s: sum(c["split"] == s for c in cases.values()) for s in ("train", "dev")}}
    if not (out / "base_trajectories.jsonl").exists():
        out.mkdir(parents=True, exist_ok=True)
        histories = [h for h in read_jsonl(source / "histories.jsonl") if h["case_id"] in cases]
        trajectories = [dict(t, protocol_identity=identity(config)) for t in read_jsonl(source / "base_trajectories.jsonl") if t["case_id"] in cases]
        if sorted(h["case_id"] for h in histories) != sorted(cases) or sorted(t["case_id"] for t in trajectories) != sorted(cases):
            raise SystemExit("the parent does not have one history and one base collaboration for every case")
        write_rows(out / "histories.jsonl", histories)
        write_rows(out / "base_trajectories.jsonl", trajectories)
        (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "provenance.json").write_text(json.dumps({
            "source_run": parent["output_dir"], "derived": time.strftime("%Y-%m-%d %H:%M:%S"),
            "procedure": "scripts/derive_subset_environment.py", "parent_config": args.parent, "parent_version": parent["version"],
            "parent_protocol_identity": identity(parent), "protocol_identity": identity(config),
            "identity_parts_that_differ": differing,
            "parent_histories_sha256": frozen["histories_sha256"], "parent_base_trajectories_sha256": frozen["base_trajectories_sha256"],
            "d1": {"reused_from_source": True, "why": "the same cases under the same protocol, models, decoding and prompts"},
            "base_collaborations": {"reused_from_source": True, "rows": len(trajectories),
                                    "changed_field": "protocol_identity (the parent's identity is recorded here)"},
            "generated_here": "nothing", "dataset_manifest": config["dataset"]["manifest"]}, indent=2), encoding="utf-8")
        result["copied"] = {"histories": len(histories), "base_trajectories": len(trajectories)}
    if args.labels:
        labels, parent_labels = ROOT / config["labels_dir"], ROOT / parent["labels_dir"]
        if labels.exists() and any(labels.iterdir()):
            raise SystemExit(f"{labels} is not empty: label rows are taken over only into a new label directory")
        labels.mkdir(parents=True, exist_ok=True)
        wanted = {h["history_id"]: h for h in turn_histories([t for t in read_jsonl(out / "base_trajectories.jsonl")])}
        known = {h["history_id"]: h for h in read_jsonl(parent_labels / "histories_turns.jsonl")}
        if any(known[h]["history"] != row["history"] for h, row in wanted.items()):
            raise SystemExit("a history differs from the parent's label history")
        sources = [parent_labels] + sorted(p for p in (parent_labels / "shards").glob("n*/shard*") if p.is_dir())
        taken = {}
        for name, key in KEYS.items():
            if name == "labels.jsonl":
                continue
            rows = {}
            for directory in sources:
                for row in read_jsonl(directory / name, missing_ok=True):
                    if row["history_id"] not in wanted:
                        continue
                    if name.endswith("rollouts.jsonl") and not (row["status"] == "completed" and row["success"] is not None):
                        continue
                    if key(row) in rows and rows[key(row)] != row:
                        raise SystemExit(f"{name}: two different rows for {key(row)} in the parent's label directories")
                    rows[key(row)] = row
            write_rows(labels / name, [rows[k] for k in sorted(rows)])
            taken[name] = len(rows)
        lines, seen = [], set()
        for directory in sources:
            path = directory / JUDGE_CACHE
            for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
                if line.strip() and json.loads(line)["key"] not in seen:
                    seen.add(json.loads(line)["key"])
                    lines.append(line)
        (labels / JUDGE_CACHE).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        taken[JUDGE_CACHE] = len(lines)
        (labels / "taken_over_from_parent.json").write_text(json.dumps({
            "parent_labels_dir": parent["labels_dir"], "sources": [str(p.relative_to(ROOT)) for p in sources], "rows": taken,
            "histories": len(wanted), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "rows of this environment's histories written by the parent's label run; labels.jsonl is built here"}, indent=2), encoding="utf-8")
        result["label_rows_taken_over"] = taken
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
