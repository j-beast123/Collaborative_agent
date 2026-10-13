"""Diagnostics of a reward log (iter<i>/reward_log.jsonl of a run, or rewards.jsonl of an E2E run)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

from src.common.io import read_jsonl
from src.common.config import read_config
from src.common.run_dir import write_new

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {"outcome": "r_outcome", "appropriateness": "r_appropriateness", "effectiveness": "r_effectiveness"}


def spread(values: list[float]) -> dict:
    return {"n": len(values), "mean": mean(values), "std": pstdev(values)} if values else {"n": 0, "mean": None, "std": None}


def groups_of(rows: list[dict]) -> list[list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("global_step"), row["sample_id"])].append(row)
    return [g for g in groups.values() if len(g) > 1]


def component_report(rows: list[dict], weights: dict[str, float]) -> dict:
    groups, out = groups_of(rows), {}
    for name, field in COMPONENTS.items():
        values = [r[field] for r in rows if r[field] is not None]
        within = [pstdev([r[field] for r in g if r[field] is not None]) for g in groups
                  if sum(r[field] is not None for r in g) > 1]
        out[name] = {"value": spread(values), "weight": weights[name],
                     "weighted": spread([weights[name] * v for v in values]),
                     "within_group_std": mean(within) if within else None,
                     "weighted_within_group_std": weights[name] * mean(within) if within else None,
                     "groups_without_variation": sum(s == 0 for s in within), "groups": len(within)}
    return out


def uncovered(row: dict) -> bool:
    """No rubric applied to the response (its R_A, if any, is the group's mean)."""
    return row["no_applicable"] if row.get("no_applicable") is not None else \
        "appropriateness" in (row.get("unavailable_components") or [])


def coverage_report(rows: list[dict]) -> dict:
    turns, by_state, applied = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        for stage, detail in zip(row.get("scored_stages") or [], row.get("turn_appropriateness_details") or []):
            turns[stage].append(bool(detail["no_applicable_rubric"]))
            by_state[str(row.get("collaboration_state"))].append(bool(detail["no_applicable_rubric"]))
        for score in (row.get("rubric_scores") or {}).values():
            for rubric_id, label in (score.get("labels") or {}).items():
                applied[rubric_id].append(label != "NA")
    without, covered = [r for r in rows if uncovered(r)], [r for r in rows if not uncovered(r)]
    return {"no_applicable_rubric_share": {stage: mean(flags) for stage, flags in sorted(turns.items())},
            "no_applicable_rubric_share_by_collaboration_state": {state: mean(flags) for state, flags in sorted(by_state.items())},
            "rubric_applicability_rate": {rubric_id: mean(flags) for rubric_id, flags in sorted(applied.items())},
            "responses_without_appropriateness": len(without), "share": len(without) / len(rows) if rows else None,
            "responses_with_an_imputed_turn": sum(r.get("imputed_R_A") is not None for r in rows),
            "turns_missing_for_the_whole_group": sum(sum((r.get("group_turn_missing") or {}).values()) for r in rows),
            "mean_reward_without_appropriateness": mean(r["reward"] for r in without) if without else None,
            "mean_reward_with_appropriateness": mean(r["reward"] for r in covered) if covered else None}


def rubric_diagnostics(rows: list[dict]) -> dict:
    by_rubric: dict[str, dict] = defaultdict(lambda: {score: {"outcomes": [], "deltas": []} for score in (0, 1, 2)})
    for row in rows:
        for stage, score in (row.get("rubric_scores") or {}).items():
            effect = (row.get("effectiveness") or {}).get(stage) or {}  # the Effectiveness of the scored turn itself
            for rubric_id, label in (score.get("labels") or {}).items():
                if label in (0, 1, 2):
                    by_rubric[rubric_id][label]["outcomes"].append(row["r_outcome"])
                    if effect.get("available"):
                        by_rubric[rubric_id][label]["deltas"].append(effect["delta"])

    def gap(levels: dict, name: str) -> Optional[float]:
        return mean(levels[2][name]) - mean(levels[0][name]) if levels[2][name] and levels[0][name] else None

    return {rubric_id: {"by_score": {str(score): {"turns": len(v["outcomes"]),
                                                    "success_rate": mean(v["outcomes"]) if v["outcomes"] else None,
                                                    "mean_effectiveness_delta": mean(v["deltas"]) if v["deltas"] else None}
                                     for score, v in levels.items()},
                        "task_success_gap": gap(levels, "outcomes"), "effectiveness_gap": gap(levels, "deltas")}
            for rubric_id, levels in sorted(by_rubric.items())}


def rubric_weight_diagnostics(rows: list[dict], rubric_set: Optional[dict] = None, selection: Optional[dict] = None) -> dict:
    """One row per rubric with every candidate basis of its weight side by side."""
    gaps, applicability = rubric_diagnostics(rows), coverage_report(rows)["rubric_applicability_rate"]
    rubrics = {r["rubric_id"]: r for r in (rubric_set or {}).get("rubrics", [])}
    categories = {c["category_id"]: c for c in (selection or {}).get("categories", [])}
    out = {}
    for rubric_id in sorted(set(applicability) | set(rubrics)):
        rubric = rubrics.get(rubric_id, {})
        category = categories.get(rubric.get("category_id"), {})
        out[rubric_id] = {
            "category_id": rubric.get("category_id"), "name": rubric.get("name"),
            "applicability_rate": applicability.get(rubric_id),
            "task_success_gap": (gaps.get(rubric_id) or {}).get("task_success_gap"),
            "effectiveness_gap": (gaps.get(rubric_id) or {}).get("effectiveness_gap"),
            "current_failure_frequency": rubric.get("failure_count", category.get("current_frequency")),
            "historical_frequency": category.get("historical_frequency"),
            "validation_success_rate": category.get("validation_success_rate"),
            "weight_failure_frequency": rubric.get("weight"),
            "weight_uniform": 1 / len(rubrics) if rubrics else None}
    return out


def group_signal(rows: list[dict]) -> dict:
    """What GRPO can learn from: it compares the rewards of the responses to one sample, so a group with equal rewards
    gives no signal. With a binary outcome as the only reward, that is every group whose responses all end the same way."""
    groups = groups_of(rows)
    if not groups:
        return {"groups": 0}
    outcomes = [[r["r_outcome"] for r in g] for g in groups]
    rewards = [[r["reward"] for r in g] for g in groups]
    return {"groups": len(groups),
            "outcome_all_0_share": mean(all(o == 0 for o in g) for g in outcomes),
            "outcome_all_1_share": mean(all(o == 1 for o in g) for g in outcomes),
            "outcome_varies_share": mean(len(set(g)) > 1 for g in outcomes),
            "mean_outcome_variance_within_group": mean(pstdev(g) ** 2 for g in outcomes),
            "groups_with_nonzero_reward_variance_share": mean(max(g) - min(g) > 1e-12 for g in rewards),
            "groups_without_learning_signal_share": mean(max(g) - min(g) <= 1e-12 for g in rewards),
            "response_diversity": response_diversity(groups)}


def response_diversity(groups: list[list[dict]]) -> dict:
    """Do the responses of a group differ at all? Equal outcomes of different responses are a limit of the reward;
    equal outcomes of near-identical responses are a limit of the sampling.
    """
    def overlap(a: str, b: str) -> float:
        a, b = set((a or "").lower().split()), set((b or "").lower().split())
        return len(a & b) / len(a | b) if a | b else 1.0

    def pairs(texts: list[str]) -> float:
        return mean(overlap(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]) if len(texts) > 1 else 1.0

    texts = [[r.get("response") or "" for r in g] for g in groups]
    no_signal = [g for g in groups if max(r["reward"] for r in g) - min(r["reward"] for r in g) <= 1e-12]

    def distinct(field: str, members: list[list[dict]]) -> dict:
        return dict(sorted(Counter(len({(r.get(field) or "").strip().lower() for r in g}) for g in members).items()))

    return {"groups_with_all_responses_distinct_share": mean(len(set(t)) == len(t) for t in texts),
            "mean_pairwise_word_overlap": mean(pairs(t) for t in texts),
            "distinct_stated_assessments_per_group": distinct("assessment", groups),
            "distinct_final_diagnoses_per_group": distinct("final_diagnosis", groups),
            "groups_without_signal": {
                "groups": len(no_signal),
                "all_responses_distinct_share": mean(len({r.get("response") for r in g}) == len(g) for g in no_signal) if no_signal else None,
                "more_than_one_stated_assessment_share": mean(len({(r.get("assessment") or "").strip().lower() for r in g}) > 1
                                                              for g in no_signal) if no_signal else None,
                "more_than_one_final_diagnosis_share": mean(len({(r.get("final_diagnosis") or "").strip().lower() for r in g}) > 1
                                                            for g in no_signal) if no_signal else None}}


def contributions(row: dict) -> dict[str, float]:
    """w * R of each component as it entered the response's reward (the weights the reward used)."""
    used = row.get("weights_used") or {}
    return {name: used[name] * row[field] for name, field in COMPONENTS.items() if name in used and row[field] is not None}


def order(values: list[float]) -> tuple:
    return tuple(sorted(range(len(values)), key=lambda i: (-round(values[i], 9), i)))


def ranking_report(rows: list[dict]) -> dict:
    groups = groups_of(rows)
    changed = {name: {"order": 0, "best_candidate": 0} for name in COMPONENTS}
    constant = {"appropriateness": [], "effectiveness": []}
    for group in groups:
        parts = [contributions(r) for r in group]
        full = order([sum(p.values()) for p in parts])
        for name in COMPONENTS:
            without = order([sum(v for k, v in p.items() if k != name) for p in parts])
            changed[name]["order"] += without != full
            changed[name]["best_candidate"] += without[0] != full[0]
        if len({r["r_outcome"] for r in group}) == 1:
            for name in constant:
                constant[name].append(pstdev([p.get(name, 0.0) for p in parts]))
    return {"groups": len(groups), "groups_where_the_component_changes": changed,
            "outcome_constant_groups": len(constant["appropriateness"]),
            "weighted_within_group_std_in_outcome_constant_groups": {
                name: mean(values) if values else None for name, values in constant.items()}}


def predictor_report(rows: list[dict], stage: str = "A1", bins: int = 5) -> dict:
    pairs = [(r["effectiveness"][stage]["after"], float(r["r_outcome"])) for r in rows
             if (r.get("effectiveness") or {}).get(stage, {}).get("available") and r["r_outcome"] is not None]
    if not pairs:
        return {"n": 0}
    clip = lambda p: min(max(p, 1e-6), 1 - 1e-6)
    table = defaultdict(list)
    for p, y in pairs:
        table[min(int(p * bins), bins - 1)].append((p, y))
    return {"n": len(pairs), "stage": stage,
            "nll": -mean(y * math.log(clip(p)) + (1 - y) * math.log(1 - clip(p)) for p, y in pairs),
            "brier": mean((p - y) ** 2 for p, y in pairs),
            "mean_predicted": mean(p for p, _ in pairs), "observed_success_rate": mean(y for _, y in pairs),
            "bins": [{"range": [b / bins, (b + 1) / bins], "n": len(members),
                      "mean_predicted": mean(p for p, _ in members), "observed": mean(y for _, y in members)}
                     for b, members in sorted(table.items())]}


def report(rows: list[dict], weights: dict[str, float], rubric_set: Optional[dict] = None, selection: Optional[dict] = None) -> dict:
    computed = [r for r in rows if r["reward_source"] == "computed" and r["status"] == "completed"]
    statuses = defaultdict(int)
    for row in rows:
        statuses[f"{row['status']}:{row['reward_source']}"] += 1
    return {"responses": len(rows), "with_computed_reward": len(computed), "status_counts": dict(statuses),
            "reward": spread([r["reward"] for r in computed]), "components": component_report(computed, weights),
            "ranking": ranking_report(computed), "group_signal": group_signal(computed),
            "by_stage": {stage: {"responses": len(members), "reward": spread([r["reward"] for r in members]),
                                 "components": component_report(members, weights), "ranking": ranking_report(members),
                                 "group_signal": group_signal(members)}
                         for stage, members in sorted(by_stage(computed).items())},
            "rubric_coverage": coverage_report(computed), "rubric_diagnostics": rubric_diagnostics(computed),
            "rubric_weight_diagnostics": rubric_weight_diagnostics(computed, rubric_set, selection),
            "predictor": predictor_report(computed)}


def by_stage(rows: list[dict]) -> dict[str, list[dict]]:
    stages = defaultdict(list)
    for row in rows:
        stages[row.get("stage", "A1")].append(row)
    return stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="written once; an existing file is an error")
    parser.add_argument("--rubric-set", help="the iteration's rubric_set.json (frequencies and weights of the active rubrics)")
    parser.add_argument("--selection", help="the iteration's active_selection.json (historical frequency, validation)")
    args = parser.parse_args()
    weights = read_config(args.config)["reward"]["weights"]
    extra = [json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else None
             for path in (args.rubric_set, args.selection)]
    text = json.dumps(dict(report(read_jsonl(args.log), weights, *extra), log=args.log), indent=2, ensure_ascii=False)
    if args.output:
        write_new(Path(args.output), text + "\n")
    print(text)


if __name__ == "__main__":
    main()
