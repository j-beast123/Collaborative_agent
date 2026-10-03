"""Shards of the full Effectiveness label run (src/agentclinic/shards.py): which history goes where, how far every shard
is, and the merge into the label directory.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from scripts.run_agentclinic_turns import frozen_protocol, scope_histories, select_train_dev
from src.agentclinic.shards import MARKER, assign, assignment_digest, merge, progress, shard_dir
from src.common.config import read_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "status", "merge"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard", type=int, help="status: this shard only")
    parser.add_argument("--require-complete", action="store_true", help="status: exit 1 unless everything shown is complete")
    parser.add_argument("--dry-run", action="store_true", help="merge: run the checks, write nothing")
    args = parser.parse_args()
    config = read_config(args.config)
    frozen = frozen_protocol(config)
    selected, _ = select_train_dev(config)
    cases = {c["case_id"]: c for c in selected}
    histories = scope_histories(config, cases, "full")
    labels_dir, names, k = ROOT / config["labels_dir"], tuple(config["candidates"]["names"]), config["k"]
    require = config["labels"]["require_compliance"]
    shard_of = assign(histories, args.shards)
    result = {"protocol_identity": frozen["protocol_identity"], "shards": args.shards, "histories": len(histories),
              "assignment_sha256": assignment_digest(shard_of, args.shards), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.command == "plan":
        result["every_history_in_exactly_one_shard"] = sorted(shard_of) == sorted(h["history_id"] for h in histories)
        result["by_shard"] = [{"shard": i, "histories": sum(s == i for s in shard_of.values()),
                               "cases": len({h["case_id"] for h in histories if shard_of[h["history_id"]] == i}),
                               "by_split": dict(Counter(cases[h["case_id"]]["split"] for h in histories if shard_of[h["history_id"]] == i))}
                              for i in range(args.shards)]
    elif args.command == "status":
        result["by_shard"], complete = [], True
        for index in range(args.shards) if args.shard is None else [args.shard]:
            directory = shard_dir(labels_dir, index, args.shards)
            mine = [h for h in histories if shard_of[h["history_id"]] == index]
            state = progress(directory, mine, names, k, require) if (directory / MARKER).is_file() else {"started": False, "complete": False}
            result["by_shard"].append({"shard": index, **state})
            complete &= state["complete"]
        if args.shard is None:
            keys = ("candidate_pairs", "candidate_pairs_expected", "compliant_pairs", "before", "before_expected", "after",
                    "after_expected_for_the_compliant_pairs_so_far", "labels", "continuation_rows", "continuations_not_completed")
            result["total"] = {key: sum(s.get(key, 0) for s in result["by_shard"]) for key in keys}
        result["complete"] = complete
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if complete or not args.require_complete else 1)
    else:
        result.update(merge(labels_dir, args.shards, histories, names, k, require_compliance=require, dry_run=args.dry_run))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
