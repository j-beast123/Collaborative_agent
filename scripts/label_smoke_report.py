"""What a smoke run of the Effectiveness labels showed, and what the full run will take."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scripts.run_agentclinic_turns import scope_histories, select_train_dev
from src.common.config import read_config
from src.common.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def report(config: dict, usage_path: Path | None, stderr_path: Path | None) -> dict:
    out, k, names = ROOT / config["labels_dir"], config["k"], list(config["candidates"]["names"])
    selected, _ = select_train_dev(config)
    cases = {c["case_id"]: c for c in selected}
    full, smoke = scope_histories(config, cases, "full"), scope_histories(config, cases, "smoke")
    ids = {h["history_id"] for h in smoke}
    turn_of = {h["history_id"]: int(h["intervention_turn"][1:]) for h in smoke}
    attempts = [a for a in read_jsonl(out / "candidate_attempts.jsonl", missing_ok=True) if a["history_id"] in ids]
    compliance = [c for c in read_jsonl(out / "candidate_compliance.jsonl", missing_ok=True) if c["history_id"] in ids]
    before = [r for r in read_jsonl(out / "before_rollouts.jsonl", missing_ok=True) if r["history_id"] in ids]
    after = [r for r in read_jsonl(out / "after_rollouts.jsonl", missing_ok=True) if r["history_id"] in ids]
    labels = [r for r in read_jsonl(out / "labels.jsonl", missing_ok=True) if r["history_id"] in ids]
    by_pair = defaultdict(list)
    for a in attempts:
        by_pair[(a["history_id"], a["candidate_id"])].append(a)
    compliant = {(c["history_id"], c["candidate_id"]) for c in compliance if c["behavior_compliant"]}
    expected_pairs = len(ids) * len(names)
    first = sum(bool(rows[0]["behavior_compliant"]) for rows in by_pair.values())
    before_per_history = Counter(r["history_id"] for r in before)
    before_keys = Counter((r["history_id"], r["rollout_index"]) for r in before)
    shared = defaultdict(set)
    for row in labels:
        shared[row["history_id"]].add(json.dumps(row["before"], sort_keys=True))
    completed = [r for r in before + after if r["status"] == "completed" and r.get("turns")]
    timing = [t for t in read_jsonl(out / "timing_smoke.jsonl", missing_ok=True)]
    usage = [u for u in (read_jsonl(usage_path, missing_ok=True) if usage_path else []) if u.get("stage") == "effectiveness_labels_smoke"]
    # elapsed time: the minutes in which the smoke made at least one API call.
    active = len({u["timestamp"][:16] for u in usage}) * 60.0
    seconds = max(active, sum(t["seconds"] for t in timing))
    by_model = Counter(u["model"] for u in usage)
    checks = {
        "all_candidates_generated_and_judged": set(by_pair) == {(h, n) for h in ids for n in names} and len(compliance) == expected_pairs,
        "before_is_k_continuations_per_history_made_once": set(before_per_history) == ids and all(v == k for v in before_per_history.values())
        and all(v == 1 for v in before_keys.values()),
        "before_is_shared_by_the_candidates_of_a_history": all(len(v) == 1 for v in shared.values()),
        "after_is_k_continuations_per_compliant_pair": Counter((r["history_id"], r["candidate_id"]) for r in after
                                                               if (r["history_id"], r["candidate_id"]) in compliant
                                                               and r.get("candidate_attempt", 1) == next(
                                                                   c["attempt"] for c in compliance if (c["history_id"], c["candidate_id"]) == (r["history_id"], r["candidate_id"])))
        == Counter({pair: k for pair in compliant}),
        "every_compliant_pair_has_a_label": {(r["history_id"], r["response_id"]) for r in labels} == compliant,
        "every_continuation_completed_with_a_judged_outcome": all(r["status"] == "completed" and r["success"] is not None for r in before + after),
        "leakage_audit_passed": json.loads((out / "turn_leakage_audit_smoke.json").read_text())["passed"]
        if (out / "turn_leakage_audit_smoke.json").is_file() else False,
    }
    result = {
        "protocol_identity": json.loads((ROOT / config["output_dir"] / "protocol.frozen.json").read_text())["protocol_identity"],
        "histories": {"smoke": len(smoke), "smoke_by_turn": dict(sorted(Counter(h["intervention_turn"] for h in smoke).items())),
                      "full_train": sum(cases[h["case_id"]]["split"] == "train" for h in full),
                      "full_dev": sum(cases[h["case_id"]]["split"] == "dev" for h in full), "full_total": len(full),
                      "note": "full_total is TRAIN + DEV; TEST has no history"},
        "candidates": {"expected_pairs": expected_pairs, "generations": len(attempts),
                       "pairs_with_a_retry": sum(len(rows) > 1 for rows in by_pair.values()),
                       "retry_rate": sum(len(rows) > 1 for rows in by_pair.values()) / len(by_pair) if by_pair else None,
                       "generator": ((config.get("label_roles") or {}).get("candidate_generator") or {}).get("model_id") or "agent model (stand-in)"},
        "compliance": {"passed_at_first_attempt": first, "passed_after_retry": len(compliant) - first,
                       "never_passed": expected_pairs - len(compliant),
                       "pass_rate": len(compliant) / expected_pairs if expected_pairs else None,
                       "by_candidate": {n: f"{sum(pair[1] == n for pair in compliant)}/{len(ids)}" for n in names},
                       "judge": sorted({str(c.get("judge_version")) for c in compliance})},
        "rollouts": {"k": k, "before": len(before), "before_expected": len(ids) * k,
                     "after": len(after), "after_expected_for_compliant_pairs": len(compliant) * k,
                     "attempted_more_than_once": sum(r.get("attempt", 1) > 1 for r in before + after),
                     "not_completed": sum(r["status"] != "completed" for r in before + after)},
        "before_sharing": {"histories": len(ids), "before_sets": len(before_per_history),
                           "candidates_sharing_one_before": checks["before_is_shared_by_the_candidates_of_a_history"]},
        "continuation": {"mean_final_turns": mean(r["turns"] for r in completed) if completed else None,
                         "mean_turns_added_after_the_intervention": mean(r["turns"] - turn_of[r["history_id"]] for r in completed) if completed else None,
                         "stop": dict(Counter(str(r.get("stop")) for r in completed))},
        "api": {"calls": len(usage), "by_model": dict(by_model), "retries": sum(u.get("retry_count") or 0 for u in usage),
                "retries_by_model": {m: sum(u.get("retry_count") or 0 for u in usage if u["model"] == m) for m in by_model},
                "http_429_lines_in_stderr": stderr_path.read_text(errors="replace").count("HTTP 429; retrying") if stderr_path and stderr_path.is_file() else None,
                "calls_per_minute": len(usage) / (seconds / 60) if seconds and usage else None,
                "cost_usd": sum(u.get("cost_usd") or 0 for u in usage)},
        "timing": {"completed_stage_rows": [{"stage": t["stage"], "seconds": t["seconds"]} for t in timing],
                   "active_seconds_from_api_calls": active, "elapsed_seconds": seconds,
                   "seconds_per_history": seconds / len(smoke) if smoke and seconds else None,
                   "estimated_full_hours": seconds / len(smoke) * len(full) / 3600 if smoke and seconds else None,
                   "estimate_basis": "working time of the smoke per history x full histories, at the speed the smoke had (it shared the GPU "
                                     "with a GRPO run if one was training); the smoke histories are a seeded sample of TRAIN cases with all their turns"},
        "checks": checks, "correctness_passed": all(checks.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--usage", help="api usage log of the smoke run")
    parser.add_argument("--stderr", help="stderr log of the smoke run (HTTP 429 lines)")
    args = parser.parse_args()
    config = read_config(args.config)
    result = report(config, Path(args.usage) if args.usage else None, Path(args.stderr) if args.stderr else None)
    (ROOT / config["labels_dir"] / "smoke_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["correctness_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
