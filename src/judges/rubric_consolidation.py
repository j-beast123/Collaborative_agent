"""Group failure-analysis rubric candidates into categories and build final rubrics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from src.data.schema import digest

from .llm import CachedChat, ChatRequest, parse_json_object
from .rubric_retention import SELECTION_ORDER, CurrentOnly, RetentionStrategy, selection_key
from .rubric_schema import validate_rubric

IMPORTANCE = ("uniform", "failure_frequency")


def candidate_view(record: dict) -> dict[str, Any]:
    analysis = record["analysis"]
    return {"rubric_id": record["rubric_id"], "interaction_failure": analysis["interaction_failure"],
            "ideal_response_behavior": analysis["ideal_response_behavior"], "rubric": record["candidate_rubric"]}


def next_category_id(categories: list[dict]) -> str:
    numbers = [int(c["category_id"].split("-")[1]) for c in categories]
    return f"cat-{max(numbers, default=0) + 1:03d}"


def assign_categories(
    records: list[dict],
    chat: CachedChat,
    *,
    system_prompt: str,
    prompt_version: str,
    params: dict,
    iteration: int,
    existing: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Return (category registry, assignment log). Only valid analyses are grouped."""
    categories = [dict(c, members=list(c.get("members", []))) for c in (existing or [])]
    log = []
    candidates = sorted((r for r in records if r["analysis_status"] == "valid"), key=lambda r: r["rubric_id"])
    for record in candidates:
        listing = [{"category_id": c["category_id"], "name": c["name"], "definition": c["definition"]}
                   for c in categories if c.get("status") != "merged"]
        user = json.dumps({"candidate": candidate_view(record), "existing_categories": listing}, ensure_ascii=False)
        result = chat.complete(ChatRequest(system_prompt, user, prompt_version, params))
        parsed, parse_status = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
        entry = {"rubric_id": record["rubric_id"], "source_trajectory_id": record["source_trajectory_id"],
                 "raw_response": result.text, "status": parse_status}
        known = {c["category_id"]: c for c in categories if c.get("status") != "merged"}
        if parsed is None:
            log.append(entry)
            continue
        decision = parsed.get("decision")
        if decision == "existing" and parsed.get("category_id") in known:
            known[parsed["category_id"]]["members"].append(record["rubric_id"])
            entry.update(status="assigned", category_id=parsed["category_id"], reason=parsed.get("reason"))
        elif decision == "new" and isinstance(parsed.get("new_category"), dict) and all(
            isinstance(parsed["new_category"].get(k), str) and parsed["new_category"][k].strip()
            for k in ("name", "definition")
        ):
            category_id = next_category_id(categories)
            categories.append({
                "category_id": category_id,
                "name": parsed["new_category"]["name"],
                "definition": parsed["new_category"]["definition"],
                "members": [record["rubric_id"]],
                "created_iteration": iteration,
                "status": "open",
                "frequency_history": {},
            })
            entry.update(status="new_category", category_id=category_id, reason=parsed.get("reason"))
        else:
            entry["status"] = "invalid_decision"
        log.append(entry)
    return categories, log


def reconcile_categories(
    categories: list[dict],
    records: list[dict],
    chat: CachedChat,
    *,
    system_prompt: str,
    prompt_version: str,
    params: dict,
    iteration: int,
) -> tuple[list[dict], dict]:
    """Final pass against order effects of one-by-one assignment."""
    open_categories = [c for c in categories if c.get("status") != "merged"]
    by_rubric = {r["rubric_id"] for r in records}
    listing = [{
        "category_id": c["category_id"], "name": c["name"], "definition": c["definition"],
        "created_iteration": c.get("created_iteration"),
        "current_members": sum(m in by_rubric for m in c["members"]),
    } for c in open_categories]
    report = {"status": "skipped", "applied": [], "rejected": []}
    if len(listing) < 2:
        return categories, report
    result = chat.complete(ChatRequest(system_prompt, json.dumps({"categories": listing}, ensure_ascii=False),
                                       prompt_version, params))
    parsed, parse_status = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
    report.update(status=parse_status, raw_response=result.text)
    merges = (parsed or {}).get("merges")
    if not isinstance(merges, list):
        return categories, report
    known = {c["category_id"]: c for c in open_categories}
    for merge in merges:
        target, sources = (merge or {}).get("keep"), (merge or {}).get("merge")
        ok = (target in known and isinstance(sources, list) and sources and target not in sources
              and all(s in known and known[s].get("created_iteration") == iteration for s in sources))
        if not ok:
            report["rejected"].append({"merge": merge, "reason": "unknown ids, self-merge or merging an older category"})
            continue
        for source in sources:
            known[target]["members"].extend(known[source]["members"])
            known[source].update(status="merged", merged_into=target, members=[])
            known.pop(source)
        report["applied"].append({"keep": target, "merged": sources, "reason": merge.get("reason")})
    return categories, report


def category_statistics(categories: list[dict], records: list[dict],
                        validations: Optional[dict[str, dict]] = None) -> dict[str, dict]:
    """Frequencies computed from assignments: failures, cases and failure turns; the frequency over all iterations (the
    registry's history plus this iteration); and, with validations ({rubric_id: row}), the share of successful ...
    """
    by_rubric = {r["rubric_id"]: r for r in records if r.get("rubric_id")}
    total_failures = len({r["source_trajectory_id"] for r in records if r["analysis_status"] == "valid"})
    stats = {}
    for category in categories:
        members = [by_rubric[m] for m in category["members"] if m in by_rubric]
        trajectories = {m["source_trajectory_id"] for m in members}
        # Which collaboration round the agent failed in (agent-turn ordinal, 1-based).
        failure_rounds = Counter()
        for member in members:
            for turn_id in member["analysis"]["failure_turn_ids"]:
                failure_rounds[member.get("agent_turn_ordinals", {}).get(str(turn_id), turn_id)] += 1
        stats[category["category_id"]] = {
            "failure_count": len(trajectories),
            "case_count": len({m["source_case_id"] for m in members}),
            "proportion_of_failures": len(trajectories) / total_failures if total_failures else 0.0,
            "failure_turn_histogram": dict(sorted(failure_rounds.items())),
            "historical_failure_count": sum(category.get("frequency_history", {}).values()) + len(trajectories),
        }
        rows = [validations[m["rubric_id"]] for m in members if m["rubric_id"] in validations] if validations else []
        reruns = sum(r["num_reruns"] for r in rows)
        stats[category["category_id"]]["validation_success_rate"] = sum(r["successes"] for r in rows) / reruns if reruns else None
    return stats


CANONICAL_INSTRUCTION = (
    "These candidates were grouped into one interaction-failure category. Return exactly one rubric in `rubrics`: "
    "the canonical rubric of this category, the one rubric the scorer will use for it. If the candidates differ "
    "in user-side context or agent-side state, write it for the situation most of them share; every score level "
    "repeats the rubric's context and differs only in behavior.\n\n")


def _construct(category, members, chat, system_prompt, prompt_version, params, canonical=False):
    """The rubric(s) of a category from its current candidates: one per distinct context, or (canonical) exactly
    the one the scorer uses."""
    instruction = CANONICAL_INSTRUCTION if canonical else (
        "These candidates were grouped into one interaction-failure category. Return the merged rubric(s) in "
        "`rubrics`: one rubric per distinct user-side context / agent-side state; every score level repeats "
        "its rubric's context and differs only in behavior.\n\n")
    user = (
        instruction
        + json.dumps({"category": {k: category[k] for k in ("name", "definition")}, "candidates": members},
                     ensure_ascii=False)
    )
    result = chat.complete(ChatRequest(system_prompt, user, prompt_version + ("+canonical-v1" if canonical else ""), params))
    parsed, parse_status = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
    rubrics = (parsed or {}).get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        return None, f"manager output invalid ({parse_status})", result.text
    problems = [f"rubric {i}: {p}" for i, r in enumerate(rubrics) for p in validate_rubric(r if isinstance(r, dict) else {})]
    if problems:
        return None, "; ".join(problems), result.text
    return rubrics, None, result.text


def build_final_rubrics(
    categories: list[dict],
    records: list[dict],
    chat: CachedChat,
    *,
    system_prompt: str,
    prompt_version: str,
    params: dict,
    iteration: int,
    max_active_categories: Optional[int] = None,
    min_failure_count: Optional[int] = None,
    retention: Optional[RetentionStrategy] = None,
    importance: str = "uniform",
    validations: Optional[dict[str, dict]] = None,
    canonical: bool = False,
) -> dict[str, Any]:
    """Active rubric set for this iteration."""
    retention = retention or CurrentOnly()
    by_rubric = {r["rubric_id"]: r for r in records if r.get("rubric_id")}
    merged = [c for c in categories if c.get("status") == "merged"]
    categories = [c for c in categories if c.get("status") != "merged"]
    stats = category_statistics(categories, records, validations)
    ranked = sorted(categories, key=lambda c: selection_key(c, stats))
    rank = {c["category_id"]: i for i, c in enumerate(ranked, 1)}
    plan, planned_out = retention.plan(categories, stats, None, min_failure_count)
    plan = sorted(plan, key=lambda e: rank[e["category"]["category_id"]])

    def base(category):
        return {"category_id": category["category_id"], "category_name": category["name"],
                "rank": rank[category["category_id"]], **stats[category["category_id"]]}

    rubrics, excluded = [], [dict(base(e["category"]), reason=e["reason"]) for e in planned_out]
    variants: dict[str, list] = {}
    for entry in plan:
        category = entry["category"]
        if max_active_categories is not None and len({r["category_id"] for r in rubrics}) >= max_active_categories:
            excluded.append(dict(base(category), reason=f"outside the top {max_active_categories} by current failure frequency"))
            continue
        if entry["action"] == "reuse":
            definitions, origin = category["rubrics"], "reused"
        else:
            members = [candidate_view(by_rubric[m]) for m in category["members"] if m in by_rubric]
            definitions, error, raw = _construct(category, members, chat, system_prompt, prompt_version, params, canonical)
            origin = "constructed"
            if definitions is None:
                excluded.append(dict(base(category), reason=error, raw_response=raw))
                continue
        if canonical:  # one scoring rubric per category; anything else the manager wrote is provenance
            canonical_rubric = category.get("canonical_rubric") if entry["action"] == "reuse" else None
            if entry["action"] != "reuse" and len(definitions) > 1:
                variants[category["category_id"]] = definitions[1:]
            definitions = [canonical_rubric or definitions[0]]
        for definition in definitions:
            rubric = {k: definition[k] for k in ("name", "applicability", "scores")}
            rubrics.append({
                **base(category),
                "rubric_id": "rubric-" + digest({"category_id": category["category_id"], **rubric})[:12],
                **rubric,
                "counterfactual_requirement": definition.get("counterfactual_requirement"),
                "source_rubric_candidates": [m for m in category["members"] if m in by_rubric],
                "source_trajectories": sorted({by_rubric[m]["source_trajectory_id"]
                                               for m in category["members"] if m in by_rubric}),
                "iteration": iteration,
                "origin": origin,
                "retention_reason": entry["reason"],
                "status": "active",
            })
    for rubric, weight in zip(rubrics, rubric_weights(rubrics, categories, importance)):
        rubric["weight"] = weight
    payload = {"iteration": iteration, "rubrics": rubrics, "excluded_categories": excluded,
               "max_active_categories": max_active_categories, "min_failure_count": min_failure_count,
               "rubric_importance": importance, "retention_policy": retention.name,
               "canonical_rubric_per_category": canonical, "variants": variants}
    payload["rubric_set_version"] = digest(payload)[:16]
    payload["selection"] = _selection(ranked, merged, stats, rank, plan, rubrics, excluded, max_active_categories, iteration)
    return payload


def _selection(ranked, merged, stats, rank, plan, rubrics, excluded, max_active, iteration) -> dict[str, Any]:
    """Every registry category with its place in the selection of the active set and, when it is not
    active, the reason."""
    active = Counter(r["category_id"] for r in rubrics)
    how = {r["category_id"]: r["retention_reason"] for r in rubrics}
    reason = {e["category_id"]: e["reason"] for e in excluded}
    eligible = {e["category"]["category_id"] for e in plan}
    rows = [{"category_id": c["category_id"], "name": c["name"], "selection_rank": rank[c["category_id"]],
             "eligible": c["category_id"] in eligible, "selected_active": c["category_id"] in active,
             "active_rubrics": active.get(c["category_id"], 0),
             "current_frequency": stats[c["category_id"]]["failure_count"],
             "current_failure_share": stats[c["category_id"]]["proportion_of_failures"],
             "historical_frequency": stats[c["category_id"]]["historical_failure_count"],
             "validation_success_rate": stats[c["category_id"]]["validation_success_rate"],
             # diagnostics only, never used for selection or weights; measured when responses are scored
             # (rubric_diagnostics of the iteration's reward_report.json)
             "effectiveness_gap": None, "task_success_gap": None,
             "previous_status": c.get("status"),
             "reactivation_status": "reactivated" if how.get(c["category_id"]) == "reactivated"
             else "new" if c.get("created_iteration") == iteration else "kept" if c["category_id"] in active else "not active",
             "reason_for_inactive": None if c["category_id"] in active else reason.get(c["category_id"], "not eligible")}
            for c in ranked]
    rows += [{"category_id": c["category_id"], "name": c["name"], "selection_rank": None, "eligible": False,
              "selected_active": False, "active_rubrics": 0, "current_frequency": 0, "current_failure_share": 0.0,
              "historical_frequency": sum(c.get("frequency_history", {}).values()), "validation_success_rate": None,
              "effectiveness_gap": None, "task_success_gap": None,
              "previous_status": "merged", "reactivation_status": "not active",
              "reason_for_inactive": f"merged into {c.get('merged_into')}"} for c in merged]
    fewer = None
    if max_active is not None and len(active) < max_active:
        failed = [r for r in rows if r["eligible"] and not r["selected_active"]]
        fewer = (f"{len(eligible)} eligible categories (validated candidates observed this iteration), "
                 f"{len(active)} active" + (f"; {len(failed)} eligible categories are not active: "
                                              + "; ".join(f"{r['category_id']}: {r['reason_for_inactive']}" for r in failed)
                                              if failed else "; the set is not filled up"))
    return {"iteration": iteration, "max_active_categories": max_active, "selection_order": list(SELECTION_ORDER),
            "all_registry_categories": len(rows), "eligible_categories": sorted(eligible, key=rank.get),
            "selected_active_categories": [r["category_id"] for r in rows if r["selected_active"]],
            "inactive_categories": [r["category_id"] for r in rows if not r["selected_active"]],
            "active_rubric_count": len(rubrics), "active_category_count": len(active),
            "fewer_than_max_reason": fewer, "categories": rows}


def rubric_weights(rubrics: list[dict], categories: list[dict], importance: str) -> list[float]:
    """Weights of the active rubrics, summing to 1."""
    if importance not in IMPORTANCE:
        raise ValueError(f"rubric importance must be one of {IMPORTANCE}")
    if importance == "uniform" or not rubrics:
        return [1.0 / len(rubrics) for _ in rubrics]
    history = {c["category_id"]: c.get("frequency_history", {}) for c in categories}
    per_category = Counter(r["category_id"] for r in rubrics)

    def mass(rubric) -> float:
        earlier = [count for _, count in sorted(history[rubric["category_id"]].items(), key=lambda kv: int(kv[0])) if count]
        return (rubric["failure_count"] or (earlier[-1] if earlier else 1)) / per_category[rubric["category_id"]]

    total = sum(mass(r) for r in rubrics)
    return [mass(r) / total for r in rubrics]


def update_registry(categories: list[dict], rubric_set: dict, records: list[dict], iteration: int,
                    validations: Optional[dict[str, dict]] = None) -> list[dict]:
    """Record this iteration's frequency, active status and latest rubric definitions."""
    stats = category_statistics([c for c in categories if c.get("status") != "merged"], records)
    selection = {row["category_id"]: row for row in (rubric_set.get("selection") or {}).get("categories", [])}
    active, reactivated = defaultdict(list), set()
    for rubric in rubric_set["rubrics"]:
        active[rubric["category_id"]].append({k: rubric[k] for k in ("name", "applicability", "scores",
                                                                     "counterfactual_requirement")})
        if rubric.get("retention_reason") == "reactivated":
            reactivated.add(rubric["category_id"])
    current = {r["rubric_id"] for r in records if r.get("rubric_id")}
    key = str(iteration)
    for category in categories:
        history = dict(category.get("frequency_history", {}))
        category.setdefault("first_seen_iteration", category.get("created_iteration", iteration))
        category.setdefault("reactivation_count", 0)
        if category.get("status") != "merged":
            history[key] = stats[category["category_id"]]["failure_count"]
            category["status"] = "active" if category["category_id"] in active else "inactive"
            chosen = selection.get(category["category_id"], {})
            category["selection_rank"] = chosen.get("selection_rank")
            category["inactive_reason"] = chosen.get("reason_for_inactive") if category["status"] == "inactive" else None
            if history[key]:
                category["last_observed_iteration"] = category["last_seen_iteration"] = iteration
            if category["category_id"] in active:
                category["rubrics"] = active[category["category_id"]]
                if rubric_set.get("canonical_rubric_per_category"):
                    category["canonical_rubric"] = category["rubrics"][0]
                    if category["category_id"] in rubric_set.get("variants", {}):
                        category["variants"] = rubric_set["variants"][category["category_id"]]
                category.setdefault("active_iterations", [])
                if iteration not in category["active_iterations"]:
                    category["active_iterations"].append(iteration)
                    category["reactivation_count"] += category["category_id"] in reactivated
            if validations is not None:
                rows = [validations[m] for m in category["members"] if m in current and m in validations]
                previous = category.get("validation", {}).get("by_iteration", {})
                by_iteration = {**previous, key: {"candidates": len(rows), "reruns": sum(r["num_reruns"] for r in rows),
                                                  "successful_reruns": sum(r["successes"] for r in rows)}}
                category["validation"] = {"by_iteration": by_iteration,
                                          **{name: sum(v[name] for v in by_iteration.values())
                                             for name in ("candidates", "reruns", "successful_reruns")}}
        category["frequency_history"] = history
        category["current_failure_frequency"] = history.get(key, 0)
        category["historical_failure_frequency"] = sum(history.values())
    return categories


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(w) > 3}


def category_diagnostics(categories: list[dict], iteration: int, similarity_threshold: float = 0.5) -> dict[str, Any]:
    """Flags for manual review after consolidation; nothing is changed."""
    live = [c for c in categories if c.get("status") != "merged"]
    words = {c["category_id"]: _tokens(f"{c['name']} {c['definition']}") for c in live}
    pairs = []
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            union = words[a["category_id"]] | words[b["category_id"]]
            similarity = len(words[a["category_id"]] & words[b["category_id"]]) / len(union) if union else 0.0
            if similarity >= similarity_threshold:
                older, newer = sorted((a, b), key=lambda c: (c.get("created_iteration", 0), c["category_id"]))
                pairs.append({"categories": [older["category_id"], newer["category_id"]],
                              "names": [older["name"], newer["name"]], "similarity": round(similarity, 3),
                              "missed_reactivation": newer.get("created_iteration") == iteration
                              and older.get("created_iteration", iteration) < iteration and older.get("status") == "inactive"})
    observed = [c for c in live if c.get("frequency_history", {}).get(str(iteration))]
    singletons = [c["category_id"] for c in observed if c["frequency_history"][str(iteration)] == 1]
    failures = sum(c["frequency_history"][str(iteration)] for c in observed)
    return {"iteration": iteration, "similarity_threshold": similarity_threshold,
            "near_duplicate_pairs": pairs,
            "possible_missed_reactivation": [p for p in pairs if p["missed_reactivation"]],
            "fragmentation": {"observed_categories": len(observed), "failures": failures,
                              "single_failure_categories": len(singletons),
                              "single_failure_share": len(singletons) / len(observed) if observed else None,
                              "categories_per_failure": len(observed) / failures if failures else None},
            "new_categories": [c["category_id"] for c in live if c.get("created_iteration") == iteration],
            "reactivated_categories": [c["category_id"] for c in live if c.get("status") == "active"
                                       and c.get("active_iterations", [])[-1:] == [iteration]
                                       and len(c.get("active_iterations", [])) > 1
                                       and c["active_iterations"][-2] != iteration - 1]}


def annotate_agent_ordinals(records: list[dict], trajectories: dict[str, dict]) -> None:
    """Map failure turn indices to collaboration rounds (the turn's round field)."""
    for record in records:
        trajectory = trajectories.get(record["source_trajectory_id"])
        if trajectory is None:
            continue
        ordinal, mapping = 0, {}
        for index, turn in enumerate(trajectory.get("turns", [])):
            if turn.get("speaker") == "agent":
                ordinal += 1
                mapping[str(index)] = int(turn.get("round", ordinal))
        record["agent_turn_ordinals"] = mapping


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
