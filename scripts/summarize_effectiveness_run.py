"""Read-only summary of an effectiveness run directory (progress, data, labels, DEV report)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.common.io import read_jsonl


def rows(path: Path) -> list[dict]:
    return read_jsonl(path, missing_ok=True)


def rate(x: dict) -> float:
    return x["successes"] / x["trials"]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/effectiveness")
    log = out / "run.log"
    print(f"== {out}")
    if log.exists():
        steps = [l.strip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.startswith("[") and ("step " in l or "done:" in l or "FAILED" in l)]
        print("progress:", *steps[-4:], sep="\n  ")

    print("\n== data")
    for split in ("train", "dev"):
        traj = rows(out / f"traj_{split}.jsonl")
        valid = [t for t in traj if t.get("trajectory_status") == "valid"]
        cands = rows(out / f"candidates_{split}.jsonl")
        by_stance = defaultdict(int)
        for c in cands:
            by_stance[c["response_id"]] += 1
        print(f"  {split}: trajectories {len(traj)} (valid {len(valid)}), candidates {dict(by_stance) or 'pending'}")

    print("\n== labels  (signal = a history whose BEFORE/AFTER values are not all equal)")
    for split in ("train", "dev"):
        labels = rows(out / f"labels_{split}.jsonl")
        rollouts = out / f"labels_{split}.jsonl.rollouts.jsonl"
        n_roll = sum(1 for _ in rollouts.open()) if rollouts.exists() else 0
        incomplete = len(rows(out / f"labels_{split}.jsonl.incomplete.jsonl"))
        if not labels:
            print(f"  {split}: pending (rollouts so far {n_roll})")
            continue
        by_history = defaultdict(list)
        for r in labels:
            by_history[r["history_id"]].append(r)
        signal = differ = 0
        mean_after = defaultdict(list)
        for group in by_history.values():
            values = [rate(group[0]["before"])] + [rate(r["after"]) for r in group]
            signal += len(set(values)) > 1
            differ += len({rate(r["after"]) for r in group}) > 1
            for r in group:
                mean_after[r["response_id"]].append(rate(r["after"]) - rate(r["before"]))
        print(f"  {split}: pairs {len(labels)}, histories {len(by_history)}, with signal {signal}, "
              f"candidates differ {differ}, incomplete {incomplete}, rollouts {n_roll}")
        print("    mean delta (after - before) by candidate:",
              {k: round(sum(v) / len(v), 3) for k, v in mean_after.items()})

    print("\n== behaviors  (AFTER success rate; histories where the agent's standalone answer differs from the clinician's)")
    for split in ("train", "dev"):
        after = defaultdict(lambda: defaultdict(list))  # who is right -> behavior -> AFTER rates
        for label in rows(out / f"labels_{split}.jsonl"):
            meta = label.get("meta") or {}
            if not label.get("intended_behavior") or meta.get("agent_initial_correctness") is None:
                continue
            who = "agent right" if meta["agent_initial_correctness"] else \
                "clinician right" if meta.get("doctor_initial_correctness") else "both wrong"
            after[who][label["intended_behavior"]].append(label["after_rate"])
            after[who]["(before)"].append(label["before_rate"])
        if not after:
            print(f"  {split}: pending or no behavior candidates")
        for who, by_behavior in after.items():
            print(f"  {split} / {who}: " + ", ".join(f"{k} {sum(v) / len(v):.2f} (n={len(v)})"
                                                      for k, v in sorted(by_behavior.items())))

    print("\n== DEV report (model vs baselines; interval = 95% bootstrap of model - baseline)")
    reports = sorted(out.glob("model*/dev_report.json"))
    if not reports:
        print("  pending")
        return
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"  [{report_path.parent.name}]")
        for metric, entry in report["comparison"].items():
            better = entry["better_is"]
            parts = [f"model {entry['model']:.3f}"]
            for name in ("train_mean", "no_response_effect", "dev_mean_reference"):
                if name in entry:
                    b = entry[name]
                    lo, hi = b.get("ci_low"), b.get("ci_high")
                    wins = lo is not None and ((better == "lower" and hi < 0) or (better == "higher" and lo > 0))
                    parts.append(f"{name} {b['value']:.3f}{' WIN' if wins else ''}")
            print(f"    {metric} ({better} better): " + ", ".join(parts))
        for name, title in (("natural_candidates", "natural responses only (original, unguided samples)"),
                            ("controlled_candidates", "controlled responses only")):
            if name not in report:
                continue
            print(f"    {title}:")
            for metric, entry in report[name]["comparison"].items():
                b = entry["no_response_effect"]
                wins = b.get("ci_low") is not None and ((entry["better_is"] == "lower" and b["ci_high"] < 0)
                                                        or (entry["better_is"] == "higher" and b["ci_low"] > 0))
                print(f"      {metric}: model {entry['model']:.3f}, no_response_effect {b['value']:.3f}{' WIN' if wins else ''}")
        same = report.get("same_answer_behavior_ranking")
        if same and same["informative_pairs"]:
            print(f"    same answer, different behavior ranking: {same['accuracy']:.3f} over {same['informative_pairs']} "
                  f"pairs (chance 0.5; {same['pairs_with_equal_outcome']} pairs with equal outcome left out)")
        ranking = report["full_metrics"]["model"]["ranking"]
        delta = report["full_metrics"]["model"]["delta"]
        print(f"    informative ranking pairs {ranking['informative_pairs']}, "
              f"nonzero-delta pairs {delta.get('nonzero_count')}, delta pearson {delta.get('pearson'):.3f}")
    print("\nWIN = the model is better than that baseline with a 95% interval excluding 0.")
    print("Key question: does the model beat no_response_effect (i.e. does reading the response help)?")


if __name__ == "__main__":
    main()
