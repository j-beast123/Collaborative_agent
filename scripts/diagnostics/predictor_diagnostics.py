"""Diagnostics of a frozen Effectiveness predictor and of the R_E it gives in running GRPO runs. Nothing is trained or changed."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scripts import train_max7 as workflow
from scripts import train_joint as joint
from src.common.io import read_jsonl
from src.effectiveness import calibration

ROOT = Path(__file__).resolve().parents[2]


def pair_table(rows: list[dict]) -> list[dict]:
    """Informative pairs: two after-examples of one history with different empirical success."""
    by_history = defaultdict(list)
    for r in rows:
        by_history[r["history_id"]].append(r)
    pairs = []
    for members in by_history.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a["target"] != b["target"]:
                    pairs.append({"case_id": a["case_id"], "history_id": a["history_id"], "turn": a["turn"], "a": a, "b": b})
    return pairs


def ordering(pairs: list[dict], score: str) -> float | None:
    if not pairs:
        return None
    total = 0.0
    for p in pairs:
        agreement = (p["a"][score] - p["b"][score]) * (p["a"]["target"] - p["b"]["target"])
        total += 1.0 if agreement > 0 else 0.5 if p["a"][score] == p["b"][score] else 0.0
    return total / len(pairs)


def bootstrap(pairs: list[dict], unit: str, statistic, n: int = 2000, seed: int = 0) -> dict:
    groups = defaultdict(list)
    for p in pairs:
        groups[p[unit]].append(p)
    keys, rng, values = sorted(groups), random.Random(seed), []
    for _ in range(n):
        sample = [p for key in (rng.choice(keys) for _ in keys) for p in groups[key]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    values.sort()
    return {"unit": unit, "units": len(keys), "ci95": [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]}


def quartiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "median": q[1], "q1": q[0], "q3": q[2], "min": min(values), "max": max(values), "mean": statistics.mean(values)}


def group_diagnostics(run: Path) -> list[dict]:
    out = []
    for directory in sorted(run.glob("iter*"), key=lambda p: int(p.name[4:])):
        log = directory / "reward_log.jsonl"
        if not log.is_file():
            continue
        groups = defaultdict(list)
        for r in read_jsonl(log):
            turn = (r.get("effectiveness") or {}).get(r.get("stage")) or {}
            if r.get("status") == "completed" and turn.get("available"):
                groups[(r["global_step"], r["sample_id"])].append({"r_e": turn["delta"], "outcome": r["outcome"], "reward": r["reward"]})
        spread = {"range": [], "sd": [], "iqr": [], "reward_range": []}
        same_outcome = tie_broken = mixed = reversed_pairs = discordant = concordant = tied = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            values = sorted(m["r_e"] for m in members)
            spread["range"].append(values[-1] - values[0])
            spread["sd"].append(statistics.pstdev(values))
            q = statistics.quantiles(values, n=4) if len(values) > 2 else [values[0], values[0], values[-1]]
            spread["iqr"].append(q[2] - q[0])
            spread["reward_range"].append(max(m["reward"] for m in members) - min(m["reward"] for m in members))
            if len({m["outcome"] for m in members}) == 1:
                same_outcome += 1
                tie_broken += len(set(values)) > 1
            else:
                mixed += 1
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if a["outcome"] == b["outcome"]:
                        continue
                    better, worse = (a, b) if a["outcome"] > b["outcome"] else (b, a)
                    concordant += better["r_e"] > worse["r_e"]
                    discordant += better["r_e"] < worse["r_e"]
                    tied += better["r_e"] == worse["r_e"]
                    reversed_pairs += better["reward"] < worse["reward"]
        pairs = concordant + discordant + tied
        out.append({"iteration": int(directory.name[4:]), "groups": len(groups),
                    "r_e_within_group": {name: quartiles(values) for name, values in spread.items() if name != "reward_range"},
                    "composite_reward_range_within_group": quartiles(spread["reward_range"]),
                    "all_same_outcome_groups": same_outcome, "of_which_r_e_gives_an_order": tie_broken, "mixed_outcome_groups": mixed,
                    "different_outcome_pairs": pairs, "pairs_where_the_composite_reverses_the_outcome_order": reversed_pairs,
                    "r_e_higher_for_the_successful_response": concordant, "r_e_lower": discordant, "r_e_equal": tied,
                    "r_e_outcome_concordance": (concordant + 0.5 * tied) / pairs if pairs else None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", nargs="*", default=[], help="name=run directory")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    c = workflow.training_config(out)
    dev, train = workflow.examples(c, "dev"), workflow.examples(c, "train")
    logits = json.loads((out / "dev_logits.json").read_text(encoding="utf-8"))
    probabilities = joint.calibrated(calibration.load(out / "calibration.json"), logits, dev)
    candidate = {(r["history_id"], r["response_id"]): r["candidate_id"] for split in ("train", "dev") for r in read_jsonl(out / f"labels/labels_{split}.jsonl")}
    by_type, by_turn_type = defaultdict(list), defaultdict(list)
    for e in train:
        if e.kind == "after":
            kind = candidate[(e.history_id, e.response_id)]
            by_type[kind].append(e.empirical_value)
            by_turn_type[(e.intervention_turn, kind)].append(e.empirical_value)
    prior = {k: statistics.mean(v) for k, v in by_type.items()}
    turn_prior = {k: statistics.mean(v) for k, v in by_turn_type.items()}
    before = {e.history_id: p for p, e in zip(probabilities, dev) if e.kind == "before"}
    rows = []
    for p, e in zip(probabilities, dev):
        if e.kind != "after":
            continue
        kind = candidate[(e.history_id, e.response_id)]
        rows.append({"history_id": e.history_id, "case_id": e.original_case_id, "turn": e.intervention_turn, "candidate": kind,
                     "target": e.empirical_value, "predictor": p, "history_only": before[e.history_id], "type_prior": prior[kind],
                     "turn_type_prior": turn_prior.get((e.intervention_turn, kind), prior[kind])})
    pairs = pair_table(rows)
    scores = ("predictor", "history_only", "type_prior", "turn_type_prior")
    result = {"dev_same_history_ordering": {
        "informative_pairs": len(pairs), "histories_with_a_pair": len({p["history_id"] for p in pairs}), "cases_with_a_pair": len({p["case_id"] for p in pairs}),
        **{s: ordering(pairs, s) for s in scores},
        "by_turn": {turn: {"pairs": len(members), **{s: ordering(members, s) for s in ("predictor", "type_prior")}}
                    for turn in sorted({p["turn"] for p in pairs}) if (members := [p for p in pairs if p["turn"] == turn])},
        "candidate_type_train_mean_success": prior,
        "predictor_agrees_with_type_prior_on_pairs": sum((p["a"]["predictor"] - p["b"]["predictor"]) * (p["a"]["type_prior"] - p["b"]["type_prior"]) > 0 for p in pairs) / len(pairs),
        "note": "history_only gives every candidate of a history the same value: every pair is a tie, 0.5 by construction"},
        "dev_ordering_confidence": {
            "predictor_bootstrap_over_histories": bootstrap(pairs, "history_id", lambda s: ordering(s, "predictor")),
            "predictor_bootstrap_over_cases": bootstrap(pairs, "case_id", lambda s: ordering(s, "predictor")),
            "type_prior_bootstrap_over_cases": bootstrap(pairs, "case_id", lambda s: ordering(s, "type_prior")),
            "predictor_minus_type_prior_over_cases": bootstrap(pairs, "case_id", lambda s: ordering(s, "predictor") - ordering(s, "type_prior") if s else None)},
        "grpo_groups": {name: group_diagnostics(ROOT / path) for name, path in (item.split("=", 1) for item in args.runs)}}
    target = out / "predictor_diagnostics.json"
    from scripts.main_experiments import write_versioned
    write_versioned(target, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
