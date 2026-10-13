"""A low-resource dataset version: a frozen dataset version with TRAIN reduced to a deterministic stratified subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.agentclinic.selection import load_manifest
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def stratum(trajectory: dict) -> str:
    turns, last = trajectory["num_turns"], trajectory["max_discussion_rounds"]
    return f"{'correct' if trajectory['is_correct'] else 'incorrect'}/{'early' if turns == 1 else 'max' if turns >= last else 'mid'}"


def choose(strata: dict[str, list[str]], size: int, seed: str) -> tuple[list[str], dict]:
    total = sum(len(members) for members in strata.values())
    exact = {name: size * len(members) / total for name, members in strata.items()}
    quota = {name: int(value) for name, value in exact.items()}
    for name in sorted(strata, key=lambda n: (-(exact[n] - quota[n]), n))[: size - sum(quota.values())]:
        quota[name] += 1
    chosen = []
    for name, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest())
        chosen += ordered[: quota[name]]
    return sorted(chosen), {name: {"train_cases": len(strata[name]), "subset": quota[name]} for name in sorted(strata)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--base-trajectories", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    target = ROOT / "data/splits" / f"{args.version}.json"
    if target.exists():
        raise SystemExit(f"{target} exists: a frozen dataset version is not written again")
    parent_path = ROOT / args.parent
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    load_manifest(parent_path, ROOT / parent["dataset_path"])  # the parent is intact and belongs to the dataset file
    train = sorted(cid for cid, row in parent["cases"].items() if row["split"] == "train")
    base = {t["case_id"]: t for t in read_jsonl(ROOT / args.base_trajectories) if t["case_id"] in set(train)}
    if sorted(base) != train or any(t["trajectory_status"] != "valid" for t in base.values()):
        raise SystemExit("every TRAIN case needs one valid base collaboration")
    strata = defaultdict(list)
    for cid in train:
        strata[stratum(base[cid])].append(cid)
    seed = f"0:{args.version}"
    chosen, table = choose(strata, args.train, seed)
    outside = sorted(set(train) - set(chosen))
    manifest = {k: v for k, v in parent.items() if k != "manifest_sha256"}
    manifest.update(dataset_version=args.version, frozen=time.strftime("%Y-%m-%d %H:%M:%S"))
    manifest["cases"] = {cid: row for cid, row in parent["cases"].items() if cid not in set(outside)}
    manifest["not_in_main"] = {**parent["not_in_main"],
                               **{cid: {"status": "train_outside_subset", "split_in_parent": "train",
                                        "parent_row": parent["cases"][cid]} for cid in outside}}
    sizes = dict(Counter(row["split"] for row in manifest["cases"].values()))
    manifest["split"] = dict(parent["split"], sizes=sizes)
    manifest["train_subset"] = {
        "parent_version": parent["dataset_version"], "parent_manifest": args.parent,
        "parent_manifest_file_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "rule": "TRAIN of the parent, stratified by the base collaboration's outcome (correct / incorrect) and termination "
                "(early: stopped after A1; mid; max: ran to the last turn); proportional quotas by largest remainder; inside a "
                "stratum the order of sha256('<seed>:<case_id>'). DEV and TEST are the parent's.",
        "procedure": "scripts/freeze_train_subset.py", "seed": seed, "size": args.train,
        "base_trajectories": args.base_trajectories,
        "base_trajectories_sha256": hashlib.sha256((ROOT / args.base_trajectories).read_bytes()).hexdigest(),
        "strata": table, "train_case_ids": chosen, "train_cases_outside_subset": outside,
        "parent_train": {"cases": len(train), "base_accuracy": sum(base[c]["is_correct"] for c in train) / len(train),
                         "mean_turns": sum(base[c]["num_turns"] for c in train) / len(train)},
        "subset_train": {"cases": len(chosen), "base_accuracy": sum(base[c]["is_correct"] for c in chosen) / len(chosen),
                         "mean_turns": sum(base[c]["num_turns"] for c in chosen) / len(chosen)}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target.chmod(0o444)
    print(json.dumps({"written": str(target.relative_to(ROOT)), "sizes": sizes, "manifest_sha256": manifest["manifest_sha256"],
                      **{k: manifest["train_subset"][k] for k in ("strata", "parent_train", "subset_train", "train_case_ids")}}, indent=2))


if __name__ == "__main__":
    main()
