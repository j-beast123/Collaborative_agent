"""The rubric set of one iteration, from that iteration's failed TRAIN collaborations."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.collaboration.rerun import rerun_from_turn
from src.collaboration.runtime import build_runtime, run_with_retries
from src.agentclinic.trajectory import PROTOCOL
from src.common.config import IntegrationUnavailable, read_config
from src.common.main_config import PIPELINE, open_run, validate_main_config, load_main_config
from src.common.io import read_jsonl
from src.data.schema import digest
from src.data.split import read_manifest
from src.judges.agent_state import resolve_include_initial_turn
from src.judges.failure_analysis import analyze_failures, load_latest, select_failures
from src.judges.effectiveness_gap import VALIDATION_FAILED, filter_and_weight
from src.judges.raw_rubrics import SCHEDULE as NO_FILTERING, raw_rubric_set
from src.judges.rubric_retention import CurrentOnly
from src.judges.rubric_retention import build_retention
from src.judges.rubric_validation import GENERATION_FAILURE, load_validations, passed_records, validate_candidates
from src.judges.llm import CachedChat, build_chat_backend
from src.judges.rubric_consolidation import (
    annotate_agent_ordinals,
    assign_categories,
    build_final_rubrics,
    category_diagnostics,
    reconcile_categories,
    update_registry,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def split_filter(config: dict):
    data = config.get("data", {})
    if not data.get("split_manifest"):
        if config.get("mode") == "production":
            raise IntegrationUnavailable("Rubric discovery must use TRAIN cases: set data.split_manifest/source_dataset")
        return None, None
    return read_manifest(data["split_manifest"]), data["source_dataset"]


def run(config: dict, trajectories_path: Path, output_dir: Path, iteration: int,
        previous_categories: Path | None = None, max_failures: int | None = None, chat_factory=None,
        rerun_fn=None, post_hoc: bool = False, failure_sample: dict | None = None,
        previous_rubric_set: Path | None = None, observe_fn=None, intervene_fn=None, behavior_fn=None) -> dict:
    """intervene_fn(record, trajectory, instruction): the frozen intervention of the rerun validation (clinic_rerun),
    or None.
    """
    if (fixed_schedule(config) or not uses_rubrics(config)) and not post_hoc:
        return record_failures(config, trajectories_path, output_dir, iteration, max_failures)
    shared = shared_initial(config, iteration)
    if shared and not unfiltered(config) and not post_hoc:
        return copy_initial(shared, output_dir)
    roles = config["roles"]
    chat_factory = chat_factory or (lambda role, name: CachedChat(
        build_chat_backend(role), ROOT / config["cache_dir"] / f"{name}.jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)

    include_initial_turn = resolve_include_initial_turn(config)  # METHOD_TBD, explicit in production
    retention = build_retention(roles["rubric_manager"].get("retention_policy", "current_only"))
    trajectories = read_jsonl(trajectories_path)
    manifest, source_dataset = split_filter(config)
    failures, counts = select_failures(trajectories, manifest=manifest, source_dataset=source_dataset)
    if max_failures is not None:
        failures = sample_failures(failures, max_failures, failure_sample["seed"]) if failure_sample else failures[:max_failures]
    write_json(output_dir / "failure_selection.json", dict(
        counts, analyzed=len(failures), iteration=iteration,
        rule=(f"seeded sample (seed {failure_sample['seed']}), taken in turn from each collaboration state" if failure_sample
              else "all failures" if max_failures is None else "the first failures in file order"),
        analyzed_trajectory_ids=[t["trajectory_id"] for t in failures],
        analyzed_by_collaboration_state=dict(Counter(str(t.get("collaboration_state")) for t in failures)),
        failed_by_collaboration_state=dict(Counter(str(t.get("collaboration_state")) for t in
                                                   select_failures(trajectories, manifest=manifest, source_dataset=source_dataset)[0]))))

    analyzer = roles["failure_analyzer"]
    analyzer_prompt = (ROOT / analyzer["prompt"]).read_text(encoding="utf-8")
    if analyzer.get("prompt_addendum"):
        analyzer_prompt += (ROOT / analyzer["prompt_addendum"]).read_text(encoding="utf-8")
    if shared and unfiltered(config) and not post_hoc:  # the shared iteration-0 analysis: its candidates, no new call
        output_dir.mkdir(parents=True, exist_ok=True)
        if not (output_dir / "failure_analyses.jsonl").exists():
            shutil.copyfile(shared / "failure_analyses.jsonl", output_dir / "failure_analyses.jsonl")
        write_json(output_dir / "initial_rubrics_source.json", {"source": str(shared), "files": ["failure_analyses.jsonl"]})
    analysis_counts = analyze_failures(
        failures,
        chat_factory(analyzer, "failure_analyzer"),
        system_prompt=analyzer_prompt,
        prompt_version=analyzer["prompt_version"],
        params=analyzer["params"],
        output_path=output_dir / "failure_analyses.jsonl",
        include_initial_turn=include_initial_turn,
    )
    wanted = {str(t["trajectory_id"]) for t in failures}
    by_trajectory = {str(t["trajectory_id"]): t for t in failures}
    analyses = [r for tid, r in load_latest(output_dir / "failure_analyses.jsonl").items() if tid in wanted]
    if unfiltered(config) and not post_hoc:
        return unfiltered_set(config, analyses, counts, analysis_counts, output_dir, iteration, previous_rubric_set)

    # Step 2-1: rerun from the failed turn with the ideal behavior; keep only flips to correct.
    validation = config.get("validation", {})
    validation_counts = None
    if validation.get("enabled"):
        if rerun_fn is None:
            needed = {"roles.agent.model_id": config["roles"]["agent"].get("model_id"),
                      "roles.collaborator_simulator.model_id": config["roles"]["collaborator_simulator"].get("model_id"),
                      "validation.reconstructor_model": validation.get("reconstructor_model")}
            missing = [name for name, value in needed.items() if not value]
            if missing:
                raise IntegrationUnavailable("Rerun validation needs: " + ", ".join(missing))
            runtime = build_runtime(
                agent_model=config["roles"]["agent"]["model_id"],
                doctor_model=config["roles"]["collaborator_simulator"]["model_id"],
                reconstructor_model=validation["reconstructor_model"],
                max_tokens=validation["max_tokens"], root=ROOT,
            )
            rerun_fn = lambda trajectory, turn, instruction, index: run_with_retries(  # noqa: E731
                lambda: rerun_from_turn(trajectory, turn, instruction, runtime, allow_initial_turn=include_initial_turn),
                format_attempts=validation.get("max_rerun_attempts", 1))
        validation_counts = validate_candidates(
            analyses, by_trajectory, rerun_fn,
            num_reruns=validation["num_reruns"], min_successes=validation["min_successes"],
            output_path=output_dir / "validations.jsonl", intervene=intervene_fn, behavior=behavior_fn,
        )
    records = passed_records(
        [r for r in analyses if r["analysis_status"] == "valid"],
        load_validations(output_dir / "validations.jsonl"),
        required=bool(validation.get("enabled")),
    )
    annotate_agent_ordinals(records, by_trajectory)

    manager = roles["rubric_manager"]
    manager_chat = chat_factory(manager, "rubric_manager")
    gap = gap_selection(config)
    # the main method starts every iteration from nothing: categories of earlier iterations are not read
    existing = json.loads(previous_categories.read_text(encoding="utf-8")) if previous_categories and not gap else None
    categories, assignment_log = assign_categories(
        records, manager_chat,
        system_prompt=(ROOT / manager["assign_prompt"]).read_text(encoding="utf-8"),
        prompt_version=manager["assign_prompt_version"], params=manager["params"],
        iteration=iteration, existing=existing,
    )
    write_json(output_dir / "category_assignments.json", assignment_log)
    categories, reconciliation = reconcile_categories(
        categories, records, manager_chat,
        system_prompt=(ROOT / manager["reconcile_prompt"]).read_text(encoding="utf-8"),
        prompt_version=manager["reconcile_prompt_version"], params=manager["params"], iteration=iteration,
    )
    write_json(output_dir / "category_reconciliation.json", reconciliation)
    reruns = load_validations(output_dir / "validations.jsonl") if validation.get("enabled") else None
    registry_settings = config.get("rubric_registry") or {}
    if gap:
        return gap_rubric_set(config, gap, categories, records, analyses, reruns, counts, analysis_counts, validation_counts,
                              manager_chat, output_dir, iteration, observe_fn, post_hoc)
    rubric_set = build_final_rubrics(
        categories, records, manager_chat,
        system_prompt=(ROOT / manager["construct_prompt"]).read_text(encoding="utf-8"),
        prompt_version=manager["construct_prompt_version"], params=manager["params"],
        iteration=iteration,
        max_active_categories=registry_settings.get("max_active_categories", manager.get("max_active_rubrics")),
        canonical="max_active_categories" in registry_settings,
        min_failure_count=manager.get("min_failure_count"), retention=retention,
        importance=(config.get("rubric_importance") or {}).get("strategy", "uniform"), validations=reruns,
    )
    selection = rubric_set["selection"]
    write_json(output_dir / "rubric_set.json", {k: v for k, v in rubric_set.items() if k != "selection"})
    write_json(output_dir / "active_selection.json", selection)
    # Category registry: never deleted; input to the next iteration (--previous-categories).
    registry = update_registry(categories, rubric_set, records, iteration, reruns)
    write_json(output_dir / "categories.json", registry)
    diagnostics = category_diagnostics(registry, iteration, manager.get("duplicate_similarity_threshold", 0.5))
    write_json(output_dir / "category_diagnostics.json", diagnostics)
    summary = {
        "iteration": iteration,
        "selection": counts,
        "analysis": analysis_counts,
        "validation": validation_counts if validation_counts is not None else "disabled (candidates unvalidated)",
        "candidates_consolidated": len(records),
        "categories": len(categories),
        # Candidates whose category decision was unparsable/invalid are not counted in any category.
        "unassigned_candidates": sum(e["status"] not in ("assigned", "new_category") for e in assignment_log),
        "merged_by_reconciliation": sum(len(m["merged"]) for m in reconciliation["applied"]),
        "active_rubrics": len(rubric_set["rubrics"]),
        "include_initial_turn": include_initial_turn,
        "retention_policy": retention.name,
        "rubric_importance": rubric_set["rubric_importance"],
        "retention": dict(Counter(r["retention_reason"] for r in rubric_set["rubrics"])),
        "excluded_categories": len(rubric_set["excluded_categories"]),
        "near_duplicate_category_pairs": len(diagnostics["near_duplicate_pairs"]),
        "possible_missed_reactivations": len(diagnostics["possible_missed_reactivation"]),
        "fragmentation": diagnostics["fragmentation"],
        "rubric_set_version": rubric_set["rubric_set_version"],
        # Frequencies of every category, including those outside the final set.
        "frequencies": sorted({
            r["category_id"]: {"rank": r["rank"], "category": r["category_name"], "failure_count": r["failure_count"],
                               "proportion": round(r["proportion_of_failures"], 3),
                               "failure_rounds": r["failure_turn_histogram"],
                               "active": any(a["category_id"] == r["category_id"] for a in rubric_set["rubrics"])}
            for r in rubric_set["excluded_categories"] + rubric_set["rubrics"]
        }.values(), key=lambda r: r["rank"]),
    }
    if post_hoc:
        summary["post_hoc_diagnostic"] = "not used by any reward or model update"
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "rubric_report.json", rubric_report(summary, selection, diagnostics, failures, analyses, records, registry))
    return summary


def gap_selection(config: dict) -> dict | None:
    """The settings of the effectiveness-gap selection when the config uses it, else None."""
    selection = (config.get("rubric_registry") or {}).get("selection") or {}
    return selection if selection.get("method") == "effectiveness_gap" else None


def gap_rubric_set(config: dict, settings: dict, categories: list[dict], records: list[dict], analyses: list[dict],
                   reruns: dict | None, counts: dict, analysis_counts: dict, validation_counts, manager_chat, output_dir: Path,
                   iteration: int, observe_fn, post_hoc: bool) -> dict:
    """Consolidated categories -> canonical rubrics -> effectiveness-gap filter and weights -> rubric_set.json."""
    if observe_fn is None:
        raise IntegrationUnavailable("effectiveness-gap selection needs the scored responses and their Effectiveness (observe_fn): "
                                     "the rubric scorer and the Effectiveness predictor of this protocol")
    manager = config["roles"]["rubric_manager"]
    consolidated = build_final_rubrics(  # every category observed in this iteration, one canonical rubric each; no cap
        categories, records, manager_chat,
        system_prompt=(ROOT / manager["construct_prompt"]).read_text(encoding="utf-8"),
        prompt_version=manager["construct_prompt_version"], params=manager["params"], iteration=iteration,
        max_active_categories=None, canonical=True, min_failure_count=None, retention=CurrentOnly(), importance="uniform",
        validations=reruns)
    candidates = [{k: v for k, v in r.items() if k not in ("weight", "status", "retention_reason")} for r in consolidated["rubrics"]]
    write_json(output_dir / "consolidated_rubrics.json", {"iteration": iteration, "rubrics": candidates,
                                                           "not_constructed": consolidated["excluded_categories"]})
    observations = observe_fn(candidates) if candidates else []
    (output_dir / "gap_observations.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in observations),
                                                       encoding="utf-8")
    kept, measured = filter_and_weight(candidates, observations, high=settings["high_score"], low=settings["low_score"],
                                       min_group_n=settings["min_group_n"], max_rubrics=settings.get("max_rubrics"))
    for rubric in kept:
        rubric["status"] = "retained"
    by_rubric = {row["rubric_id"]: row for row in measured}
    rubric_set = {"iteration": iteration, "rubric_schedule": "evolving", "selection_method": "effectiveness_gap",
                  "rubrics": kept, "rubric_importance": "effectiveness_gap", "canonical_rubric_per_category": True,
                  "carried_over_from_earlier_iterations": [], "max_active_categories": None,
                  "selection": {**{k: settings[k] for k in ("high_score", "low_score", "min_group_n", "population")},
                                "max_rubrics": settings.get("max_rubrics")},
                  "scored_responses": len(observations),
                  "scored_responses_with_effectiveness": sum(o.get("effectiveness") is not None for o in observations)}
    rubric_set["rubric_set_version"] = digest(rubric_set)[:16]
    write_json(output_dir / "rubric_set.json", rubric_set)

    # the iteration's table: every consolidated category, and every candidate that validation rejected
    passed = {r["rubric_id"] for r in records}
    table = [{"level": "category", "category_id": r["category_id"], "category": r["category_name"], "rubric_id": r["rubric_id"],
              "rubric": r["name"], "source_failures": r["source_trajectories"], "source_rubric_candidates": r["source_rubric_candidates"],
              "generation_count": r["failure_count"], "validation_success_rate": r["validation_success_rate"],
              **{k: v for k, v in by_rubric[r["rubric_id"]].items() if k != "rubric_id"}} for r in candidates]
    table += [{"level": "category", "category_id": e["category_id"], "category": e["category_name"], "rubric_id": None, "rubric": None,
               "generation_count": e["failure_count"], "validation_success_rate": e["validation_success_rate"],
               "retained": False, "final_weight": None, "removal_reason": f"no canonical rubric: {e['reason']}"}
              for e in consolidated["excluded_categories"] if e["failure_count"]]
    table += [{"level": "candidate", "category_id": None, "category": None, "rubric_id": a["rubric_id"],
               "rubric": a["candidate_rubric"]["name"], "source_failures": [a["source_trajectory_id"]],
               "source_rubric_candidates": [a["rubric_id"]], "generation_count": 1,
               "validation_success_rate": (reruns[a["rubric_id"]]["successes"] / reruns[a["rubric_id"]]["num_reruns"]
                                           if reruns and a["rubric_id"] in reruns and reruns[a["rubric_id"]].get("num_reruns")
                                           and not generation_failed(reruns, a) else None),
               # a candidate whose intervention was never realized was not tested: it is not a validation failure
               "retained": False, "final_weight": None,
               "removal_reason": GENERATION_FAILURE if generation_failed(reruns, a) else VALIDATION_FAILED}
              for a in analyses if a["analysis_status"] == "valid" and a["rubric_id"] not in passed]
    write_json(output_dir / "rubric_table.json", {"iteration": iteration, "rows": table})
    # the iteration's categories, as a record (not read by any later iteration)
    retained = {r["category_id"] for r in kept}
    removal = {row["category_id"]: row["removal_reason"] for row in table if row["level"] == "category"}
    canonical = {r["category_id"]: r for r in candidates}
    write_json(output_dir / "categories.json", [
        {"iteration": iteration, "category_id": c["category_id"], "name": c["name"], "definition": c.get("definition"),
         "members": c["members"], "merged_into": c.get("merged_into"),
         "status": "merged" if c.get("status") == "merged" else "retained" if c["category_id"] in retained else "removed",
         "removal_reason": removal.get(c["category_id"]),
         "frequency_history": {str(iteration): canonical[c["category_id"]]["failure_count"]} if c["category_id"] in canonical else {},
         "canonical_rubric": {k: canonical[c["category_id"]][k] for k in ("name", "applicability", "scores")}
         if c["category_id"] in canonical else None}
        for c in categories])
    reasons = Counter(row["removal_reason"] for row in table if not row["retained"])
    summary = {"iteration": iteration, "rubric_schedule": "evolving", "selection_method": "effectiveness_gap",
               "selection": counts, "analysis": analysis_counts,
               "validation": validation_counts if validation_counts is not None else "disabled (candidates unvalidated)",
               "raw_candidates": sum(a["analysis_status"] == "valid" for a in analyses), "validated_candidates": len(records),
               "consolidated_categories": len(candidates), "scored_responses": len(observations),
               "retained_rubrics": len(kept), "removed": dict(reasons),
               "weights": {r["rubric_id"]: r["weight"] for r in kept},
               "carried_over_from_earlier_iterations": 0, "rubric_set_version": rubric_set["rubric_set_version"]}
    if not kept:
        summary["warning"] = "no rubric was retained: every response of this iteration has no Appropriateness value"
    if post_hoc:
        summary["post_hoc_diagnostic"] = "not used by any reward or model update"
    write_json(output_dir / "summary.json", summary)
    return summary


def generation_failed(reruns: dict | None, analysis: dict) -> bool:
    return bool(reruns) and reruns.get(analysis["rubric_id"], {}).get("validation_status") == GENERATION_FAILURE


def gap_observer(config: dict, iteration: int, trajectories_path: Path, output_dir: Path):
    """observe_fn of the real pipeline."""
    from concurrent.futures import ThreadPoolExecutor

    from src.agentclinic.adapter import agent_view, render
    from src.agentclinic.pipeline import QUESTION, Appender
    from src.agentclinic.selection import split_eligible_cases
    from src.effectiveness.service import build_effectiveness
    from src.grpo.trainer import build_scorer, effectiveness_settings
    from src.judges.rubric_scoring import build_scoring_items, final_rubric_view

    section = effectiveness_settings(config, iteration)
    if not section.get("stages"):
        raise IntegrationUnavailable("effectiveness-gap selection needs the Effectiveness predictor of this protocol "
                                     "(reward.effectiveness.stages is empty)")
    cases = split_eligible_cases(read_config(ROOT / config["environment"]["config"]), ROOT)
    trajectories = [t for t in read_jsonl(trajectories_path)
                    if t["trajectory_status"] == "valid" and cases[t["case_id"]]["split"] == config["environment"]["train_split"]]

    def observe(rubrics: list[dict]) -> list[dict]:
        scorer = build_scorer(config, ROOT)
        provider = build_effectiveness(section, 1.0, ROOT)  # also for a reward without R_E: the filter is part of the rubric method
        views = [final_rubric_view(r) for r in rubrics]
        scores = Appender(output_dir / "gap_scores.jsonl", key=lambda r: r["item_id"])
        effects = Appender(output_dir / "gap_effectiveness.jsonl", key=lambda r: r["item_id"])
        by_trajectory = {str(t["trajectory_id"]): t for t in trajectories}
        items = [item for t in trajectories for item in build_scoring_items(t, views, include_initial_turn=True)]

        def score(item):
            if item.item_id in scores.keys:
                return
            result = scorer.score(item)
            # the scorer's input and its whole decision are kept, so that a sample can be audited independently later
            scores.add({"item_id": item.item_id, "trajectory_id": item.trajectory_id, "case_id": item.case_id, "turn_id": item.turn_id,
                        "status": result.status, "labels": result.labels, "probabilities": result.probabilities,
                        "scorer": scorer.identity(), "input": item.payload()})

        with ThreadPoolExecutor(config["rollout"]["workers"]) as pool:
            list(pool.map(score, items))
        for item in items:  # the predictor is called from one thread
            if item.item_id in effects.keys:
                continue
            trajectory = by_trajectory[item.trajectory_id]
            turn = trajectory["turns"][item.turn_id]
            stage = turn["stage"]
            if stage not in section["stages"]:
                effects.add({"item_id": item.item_id, "stage": stage, "available": False, "delta": None})
                continue
            history = {"history_id": item.item_id, "intervention_turn": stage, "question": QUESTION, "options": {},
                       "scenario": render(agent_view(cases[trajectory["case_id"]])),
                       "turns": [{"speaker": t["speaker"], "text": t["text"]} for t in trajectory["turns"][:item.turn_id]]}
            prediction = provider.predict(history, turn["text"], stage, trajectory_id=item.item_id)
            effects.add({"item_id": item.item_id, "stage": stage, "available": True, "delta": prediction.delta,
                         "p_before": prediction.p_before, "p_after": prediction.p_after, "model_version": prediction.model_version})
        effect = {r["item_id"]: r for r in effects.rows}
        wanted = {item.item_id for item in items}
        return [{"item_id": r["item_id"], "trajectory_id": r["trajectory_id"], "case_id": r["case_id"], "turn_id": r["turn_id"],
                 "stage": effect[r["item_id"]]["stage"], "labels": r["labels"] if r["status"] == "scored" else {},
                 "effectiveness": effect[r["item_id"]]["delta"]} for r in scores.rows if r["item_id"] in wanted]

    return observe


def unfiltered(config: dict) -> bool:
    return (config.get("rubric_registry") or {}).get("schedule") == NO_FILTERING


def shared_initial(config: dict, iteration: int) -> Path | None:
    """The iter0/rubrics directory this run takes its iteration-0 rubric step from, if any."""
    source = (config.get("rubric_registry") or {}).get("initial_rubrics")
    return ROOT / source if source and iteration == 0 else None


def copy_initial(source: Path, output_dir: Path) -> dict:
    """Evolving schedule, iteration 0: R_0 and its registry are the shared ones, file for file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.glob("*.json*")):
        if not (output_dir / path.name).exists():
            shutil.copyfile(path, output_dir / path.name)
        copied.append(path.name)
    for name in ("rubric_set.json", "categories.json", "failure_analyses.jsonl"):
        if name not in copied:
            raise IntegrationUnavailable(f"{source} has no {name}: it is not a finished iteration-0 rubric step")
    write_json(output_dir / "initial_rubrics_source.json", {"source": str(source), "files": copied})
    return dict(json.loads((output_dir / "summary.json").read_text(encoding="utf-8")), initial_rubrics_source=str(source))


def unfiltered_set(config: dict, analyses: list[dict], counts: dict, analysis_counts: dict, output_dir: Path, iteration: int,
                   previous_rubric_set: Path | None) -> dict:
    """schedule no_filtering: every valid candidate of this iteration joins the set of the iteration before."""
    if iteration > 0 and previous_rubric_set is None:
        raise IntegrationUnavailable("schedule no_filtering needs --previous-rubric-set from iteration 1 on (the set is cumulative)")
    previous = json.loads(previous_rubric_set.read_text(encoding="utf-8")) if iteration > 0 else None
    rubric_set = raw_rubric_set(analyses, iteration, previous, (config.get("rubric_importance") or {}).get("strategy", "failure_frequency"))
    write_json(output_dir / "rubric_set.json", rubric_set)
    summary = {"iteration": iteration, "rubric_schedule": NO_FILTERING, "selection": counts, "analysis": analysis_counts,
               "validation": "not run (no_filtering)", "consolidation": "not run (no_filtering)",
               "raw_candidates_this_iteration": rubric_set["discovered_by_iteration"][str(iteration)],
               "active_rubrics": len(rubric_set["rubrics"]), "discovered_by_iteration": rubric_set["discovered_by_iteration"],
               "technical_dedup": len(rubric_set["technical_dedup"]), "rubric_set_version": rubric_set["rubric_set_version"]}
    write_json(output_dir / "summary.json", summary)
    return summary


def uses_rubrics(config: dict) -> bool:
    """False for a reward without Appropriateness (w_A = 0): nothing would read the rubrics."""
    return ((config.get("reward") or {}).get("weights") or {}).get("appropriateness", 1) > 0


def sample_failures(failures: list[dict], n: int, seed: int) -> list[dict]:
    """n failures chosen before any of them is analysed: the failures of each collaboration state are shuffled with the
    seed and taken in turn (Accept, Challenge, Preserve, Recover, undefined), so that no state fills the sample.
    """
    import random

    groups: dict[str, list[dict]] = {}
    for trajectory in sorted(failures, key=lambda t: str(t["trajectory_id"])):
        groups.setdefault(str(trajectory.get("collaboration_state")), []).append(trajectory)
    rng = random.Random(seed)
    for members in groups.values():
        rng.shuffle(members)
    chosen, names = [], sorted(groups)
    while len(chosen) < min(n, len(failures)):
        for name in names:
            if groups[name] and len(chosen) < n:
                chosen.append(groups[name].pop(0))
    return sorted(chosen, key=lambda t: str(t["trajectory_id"]))


def fixed_schedule(config: dict) -> bool:
    return (config.get("rubric_registry") or {}).get("schedule") == "fixed"


def record_failures(config: dict, trajectories_path: Path, output_dir: Path, iteration: int,
                    max_failures: int | None = None) -> dict:
    """Fixed schedule: which collaborations of the iteration failed, and nothing else. No model is called."""
    manifest, source_dataset = split_filter(config)
    failures, counts = select_failures(read_jsonl(trajectories_path), manifest=manifest, source_dataset=source_dataset)
    if max_failures is not None:
        failures = failures[:max_failures]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "failure_selection.json", dict(counts, analyzed=0, recorded=len(failures), iteration=iteration))
    (output_dir / "failures.jsonl").write_text("".join(json.dumps(
        {"trajectory_id": t["trajectory_id"], "case_id": t["case_id"], "history_id": t.get("history_id"),
         "collaboration_state": t.get("collaboration_state"), "final_diagnosis": t.get("final_diagnosis")},
        ensure_ascii=False) + "\n" for t in failures), encoding="utf-8")
    fixed = fixed_schedule(config)
    summary = {"iteration": iteration, "rubric_schedule": "fixed" if fixed else "none (w_A = 0)", "selection": counts,
               "failures_recorded": len(failures),
               "rubric_generation": "not run: the reward uses rubric_registry.fixed_rubric_set as it is" if fixed
               else "not run: the reward has no Appropriateness component",
               "fixed_rubric_set": config["rubric_registry"]["fixed_rubric_set"] if fixed else None}
    write_json(output_dir / "summary.json", summary)
    return summary


def rubric_report(summary: dict, selection: dict, diagnostics: dict, failures: list[dict], analyses: list[dict],
                  records: list[dict], registry: list[dict]) -> dict:
    """The iteration's rubric pipeline at a glance. Scorer applicability and no-applicable rates are measured
    when responses are scored and are in the iteration's reward report (scripts/reward_report.py)."""
    category_of = {member: c["category_id"] for c in registry for member in c.get("members", [])}
    active = set(selection["selected_active_categories"])
    validated = {r["source_trajectory_id"]: r["rubric_id"] for r in records}
    coverage: dict[str, dict] = {}
    for trajectory in failures:
        row = coverage.setdefault(str(trajectory.get("collaboration_state")), {"failures": 0, "with_validated_candidate": 0,
                                                                              "covered_by_active_rubric": 0})
        rubric = validated.get(str(trajectory["trajectory_id"]))
        row["failures"] += 1
        row["with_validated_candidate"] += rubric is not None
        row["covered_by_active_rubric"] += category_of.get(rubric) in active
    return {
        "iteration": summary["iteration"], "failure_trajectories": len(failures),
        "raw_rubric_candidates": sum(r["analysis_status"] == "valid" for r in analyses),
        "validation": summary["validation"], "validated_candidates": len(records),
        "consolidated_categories": sum(c.get("status") != "merged" for c in registry),
        "merged_categories": sum(c.get("status") == "merged" for c in registry),
        "max_active_categories": selection["max_active_categories"],
        "active_categories": selection["active_category_count"], "active_rubrics": selection["active_rubric_count"],
        "inactive_categories": len(selection["inactive_categories"]),
        "fewer_than_max_reason": selection["fewer_than_max_reason"],
        "selection_order": selection["selection_order"],
        "categories": [{k: row[k] for k in ("category_id", "name", "selection_rank", "current_frequency", "historical_frequency",
                                             "validation_success_rate", "selected_active", "reactivation_status",
                                             "reason_for_inactive")} for row in selection["categories"]],
        "reactivated_categories": [r["category_id"] for r in selection["categories"] if r["reactivation_status"] == "reactivated"],
        "duplication": {"near_duplicate_pairs": diagnostics["near_duplicate_pairs"],
                        "possible_missed_reactivation": diagnostics["possible_missed_reactivation"],
                        "fragmentation": diagnostics["fragmentation"]},
        "failure_coverage_by_collaboration_state": coverage,
        "measured_at_scoring": "scorer applicability rate and no-applicable rate: reward_report.json of the iteration",
    }


def clinic_rerun(config: dict, iteration: int, output_dir: Path):
    """(rerun, intervene) of the AgentClinic protocol: the iteration's Agent (theta_iteration) answers the failed
        turn again under the candidate's ideal behavior and the collaboration continues to the Doctor's final diagnosis.
    """
    from src.agentclinic.pipeline import Appender
    from src.agentclinic.runtime import build_clinic, load_policy
    from src.agentclinic.trajectory import intervention_message
    from src.agentclinic.trajectory import rerun_from_turn as clinic_rerun_from_turn
    from src.grpo.trainer import build_scorer
    from src.judges.rubric_scoring import build_scoring_items, rubric_view
    from src.grpo.trainer import previous_adapter

    model, tokenizer = load_policy(config["policy"], previous_adapter(config, ROOT, iteration), trainable=False,
                                   device=config["devices"]["policy"])
    clinic = build_clinic(config, ROOT, model, tokenizer)
    attempts = config["validation"].get("max_rerun_attempts", 1)
    from src.grpo.trainer import spec_of

    spec = spec_of(clinic.source)  # the rerun resumes under the run's protocol (early stopping included)
    rerun = lambda trajectory, turn, instruction, index, fixed=None: run_with_retries(  # noqa: E731
        lambda: clinic_rerun_from_turn(trajectory, turn, instruction, clinic.cases[trajectory["case_id"]],
                                       clinic.runtime, clinic.judge, spec, fixed=fixed), format_attempts=attempts)
    # Diagnostic-only from here on: frozen intervention, private re-assessment and the behavioral gate (off in the main config).
    settings = config["validation"].get("intervention") or {}
    if not settings:
        return rerun, None, None
    if settings.get("mode") != "frozen_compliant" or settings.get("required_score") != 2 or settings.get("max_attempts", 0) < 1:
        raise SystemExit("validation.intervention: mode frozen_compliant with required_score 2 and max_attempts >= 1")
    scorer = build_scorer(config, ROOT)
    frozen = Appender(output_dir / "interventions.jsonl", key=lambda r: r["rubric_id"])

    def intervene(record: dict, trajectory: dict, instruction: str) -> dict:
        stored = next((r for r in frozen.rows if r["rubric_id"] == record["rubric_id"]), None)
        if stored:
            return stored
        turn, rubric, tried = record["analysis"]["intervention_turn_id"], rubric_view(record), []
        reassessment = None
        if settings.get("reassessment"):  # diagnostic only: private re-assessment before the attempts
            from scripts.diagnostics.reassessment import private_reassessment, reassessment_guidance

            reassessment = private_reassessment(trajectory, turn, clinic.cases[trajectory["case_id"]], clinic.runtime,
                                                (ROOT / settings["reassessment"]["prompt"]).read_text(encoding="utf-8"))
            if reassessment["parsed"] is None:
                outcome = {"rubric_id": record["rubric_id"], "source_trajectory_id": record["source_trajectory_id"],
                           "intervention_turn_id": turn, "stage": trajectory["turns"][turn]["stage"], "status": GENERATION_FAILURE,
                           "text": None, "assessment": None, "reassessment": reassessment,
                           "compliance": {"passed": False, "reason": "the private re-assessment was not one JSON object"},
                           "instruction": instruction, "max_attempts": settings["max_attempts"], "attempts": []}
                frozen.add(outcome)
                return outcome
        guideline = instruction if reassessment is None else instruction + "\n\n" + reassessment_guidance(reassessment["parsed"])
        for attempt in range(1, settings["max_attempts"] + 1):
            message = intervention_message(trajectory, turn, guideline, clinic.cases[trajectory["case_id"]], clinic.runtime, spec)
            row = {"attempt": attempt, "text": message["text"], "assessment": message["assessment"], "label": None}
            tried.append(row)
            if not message["text"]:
                row["problem"] = "empty message"
                continue
            # what the scorer of the reward would see had the Agent given this answer: the dialogue up to the failed
            # turn, this answer as the target turn, and the state before it (unchanged: nothing before the turn differs)
            answered = dict(trajectory, turns=trajectory["turns"][:turn] + [dict(trajectory["turns"][turn], text=message["text"])])
            item, = [i for i in build_scoring_items(answered, [rubric], include_initial_turn=True) if i.turn_id == turn]
            result = scorer.score(item)
            row.update(label=result.labels.get(record["rubric_id"]), scorer_status=result.status,
                       probabilities=result.probabilities.get(record["rubric_id"]))
            if result.status != "scored":  # not a decision on the answer: the candidate is retried on the next run
                raise IntegrationUnavailable(f"compliance check of {record['rubric_id']}: {result.problems}")
            if row["label"] == 2:
                break
        chosen = tried[-1] if tried[-1]["label"] == 2 else None
        outcome = {"rubric_id": record["rubric_id"], "source_trajectory_id": record["source_trajectory_id"],
                   "intervention_turn_id": turn, "stage": trajectory["turns"][turn]["stage"],
                   "status": "frozen" if chosen else GENERATION_FAILURE,
                   "text": chosen["text"] if chosen else None, "assessment": chosen["assessment"] if chosen else None,
                   "compliance": {"checker": scorer.identity(), "criterion": "score 2 of the candidate rubric on the failed turn",
                                  "passed": bool(chosen), "accepted_attempt": chosen["attempt"] if chosen else None},
                   "generator": "the iteration's Agent under the ideal-behavior instruction"
                                + (" after its private re-assessment" if reassessment else ""), "instruction": guideline,
                   **({"reassessment": reassessment} if reassessment else {}),
                   "max_attempts": settings["max_attempts"], "attempts": tried}
        frozen.add(outcome)
        return outcome

    if config["validation"].get("criterion", "task") != "behavior":
        return rerun, intervene, None
    samples = Appender(output_dir / "behavior_samples.jsonl", key=lambda r: r["rubric_id"])

    def behavior(record: dict, trajectory: dict, instruction: str) -> dict:
        """validation.criterion behavior: the original response's label, then num_reruns independent responses under the
        rubric's instruction from the same prefix, each scored by the run's rubric scorer against the candidate."""
        stored = next((r for r in samples.rows if r["rubric_id"] == record["rubric_id"]), None)
        if stored:
            return {k: stored[k] for k in ("before", "samples")}
        turn, rubric = record["analysis"]["intervention_turn_id"], rubric_view(record)

        def label(text: str) -> dict:
            answered = dict(trajectory, turns=trajectory["turns"][:turn] + [dict(trajectory["turns"][turn], text=text)])
            item, = [i for i in build_scoring_items(answered, [rubric], include_initial_turn=True) if i.turn_id == turn]
            result = scorer.score(item)
            if result.status != "scored":
                raise IntegrationUnavailable(f"behavior check of {record['rubric_id']}: {result.problems}")
            return {"label": result.labels.get(record["rubric_id"]), "probabilities": result.probabilities.get(record["rubric_id"])}

        before = label(trajectory["turns"][turn]["text"])
        drawn = []
        for _ in range(config["validation"]["num_reruns"]):
            message = intervention_message(trajectory, turn, instruction, clinic.cases[trajectory["case_id"]], clinic.runtime, spec)
            drawn.append({"text": message["text"], **(label(message["text"]) if message["text"] else {"label": None})})
        row = {"rubric_id": record["rubric_id"], "intervention_turn_id": turn, "before": before["label"],
               "before_probabilities": before["probabilities"], "samples": drawn, "instruction": instruction}
        samples.add(row)
        return {"before": row["before"], "samples": drawn}

    return rerun, intervene, behavior


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trajectories", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--previous-categories")
    p.add_argument("--previous-rubric-set", help="schedule no_filtering: rubric_set.json of the iteration before")
    p.add_argument("--max-failures", type=int)
    p.add_argument("--run-id", help="main pipeline: the run this iteration belongs to")
    p.add_argument("--post-hoc-diagnostic", action="store_true",
                   help="fixed schedule only: run the rubric pipeline on the run's trajectories for analysis; "
                        "--output-dir must not be a rubrics directory that training reads")
    p.add_argument("--set", action="append", default=[], metavar="a.b=JSON")
    p.add_argument("--allow-stand-ins", action="store_true",
                   help="run although a decided rubric model (Sol, Jev) is not connected and its stand-in would be used")
    a = p.parse_args()
    config = read_config(a.config)
    if config.get("pipeline") == PIPELINE:
        if not a.run_id:
            raise SystemExit("the main pipeline config needs --run-id (see scripts/run_grpo.py init)")
        config = open_run(load_main_config(a.config, ROOT, a.set), ROOT, run_id=a.run_id, resume=True)
    if a.post_hoc_diagnostic and (not fixed_schedule(config) or Path(a.output_dir).name == "rubrics"):
        raise SystemExit("--post-hoc-diagnostic is for a fixed-schedule run and writes to its own directory "
                         "(e.g. iter<i>/posthoc_rubrics), not to iter<i>/rubrics")
    # which rubric models this step calls, checked before any model is loaded: none for a fixed set, a reward without
    # Appropriateness or a shared iteration-0 step; the failure analyzer for the unfiltered set; all three for the main method
    calls_rubric_models = (not fixed_schedule(config) and uses_rubrics(config)
                           and not (shared_initial(config, a.iteration) is not None)) or a.post_hoc_diagnostic
    if calls_rubric_models:
        from src.common.main_config import refuse_stand_ins

        roles = ["failure_analyzer"] if unfiltered(config) and not a.post_hoc_diagnostic else \
            ["failure_analyzer", "rubric_manager"] + (["rubric_scorer"] if gap_selection(config) else [])
        refuse_stand_ins(config, roles, "rubric generation", a.allow_stand_ins)
    rerun_fn = intervene_fn = behavior_fn = None
    needs_models = (not fixed_schedule(config) and uses_rubrics(config) and not unfiltered(config)
                    and not shared_initial(config, a.iteration)) or a.post_hoc_diagnostic  # models of the validation rerun
    if needs_models and str(config["collaboration"].get("protocol", "")).startswith("agentclinic") and config.get("validation", {}).get("enabled"):
        rerun_fn, intervene_fn, behavior_fn = clinic_rerun(config, a.iteration, Path(a.output_dir))
    observe_fn = gap_observer(config, a.iteration, Path(a.trajectories), Path(a.output_dir)) \
        if needs_models and gap_selection(config) else None
    summary = run(config, Path(a.trajectories), Path(a.output_dir), a.iteration,
                  Path(a.previous_categories) if a.previous_categories else None, a.max_failures, rerun_fn=rerun_fn,
                  observe_fn=observe_fn, intervene_fn=intervene_fn, behavior_fn=behavior_fn,
                  post_hoc=a.post_hoc_diagnostic,
                  previous_rubric_set=Path(a.previous_rubric_set) if a.previous_rubric_set else None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
