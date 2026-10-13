"""RQ1 accuracy tables and RQ2-A1 collaboration-state appropriateness (EVAL)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional

from src.collaboration.generator import option

from .stats import cluster_bootstrap_mean

STATES = {(True, True): "Preserve", (False, True): "Accept", (True, False): "Challenge", (False, False): "Recover"}
SCHEMA_VERSION = "1"


def collaboration_state(trajectory: dict) -> tuple[Optional[str], Optional[bool], Optional[bool], Optional[str]]:
    """(state, agent_initial_correct, doctor_initial_correct, exclusion_reason)."""
    if "agent_initial_correct" in trajectory:
        agent_ok, doctor_ok = trajectory["agent_initial_correct"], trajectory.get("doctor_initial_correct")
        if agent_ok is None or doctor_ok is None:
            return None, None, None, "initial agent or doctor diagnosis not stated or not judged"
        return STATES[(agent_ok, doctor_ok)], agent_ok, doctor_ok, None
    truth = (trajectory.get("ground_truth") or {}).get("correct_option")
    agent, doctor = trajectory.get("agent_initial_option"), trajectory.get("doctor_initial_option")
    if not truth:
        return None, None, None, "missing ground truth"
    if not agent or not doctor:
        return None, None, None, "initial agent or doctor option not parsed"
    agent_ok, doctor_ok = agent == truth, doctor == truth
    return STATES[(agent_ok, doctor_ok)], agent_ok, doctor_ok, None


def accuracy_table(runs: Iterable[dict], *, n_boot: int = 2000) -> list[dict[str, Any]]:
    """RQ1 main table. Each run: {dataset, system, checkpoint, iteration, trajectories, standalone?}."""
    rows = []
    for run in runs:
        trajectories = run["trajectories"]
        valid = [t for t in trajectories if t.get("trajectory_status") == "valid"]
        cases = [str(t["case_id"]) for t in trajectories]
        itt = cluster_bootstrap_mean([float(bool(t.get("is_correct"))) for t in trajectories], cases, n_boot=n_boot)
        row = {
            "dataset": run["dataset"], "system": run["system"], "checkpoint": run["checkpoint"],
            "iteration": run["iteration"], "n_cases": len(set(cases)), "n_invalid": len(trajectories) - len(valid),
            "collaborative_accuracy": itt["mean"], "ci_low": itt["ci_low"], "ci_high": itt["ci_high"],
            "collaborative_accuracy_valid_only": (sum(bool(t["is_correct"]) for t in valid) / len(valid)) if valid else None,
        }
        if run.get("standalone") is not None:
            standalone = run["standalone"]
            sa = cluster_bootstrap_mean([float(bool(r.get("is_correct"))) for r in standalone],
                                        [str(r["case_id"]) for r in standalone], n_boot=n_boot)
            row.update(standalone_accuracy=sa["mean"], standalone_ci_low=sa["ci_low"], standalone_ci_high=sa["ci_high"])
        by_state = defaultdict(list)
        for t in valid:
            state = collaboration_state(t)[0]
            if state:
                by_state[state].append(bool(t["is_correct"]))
        for state in STATES.values():
            values = by_state.get(state, [])
            row[f"final_accuracy_{state}"] = sum(values) / len(values) if values else None
            row[f"n_{state}"] = len(values)
        rows.append(row)
    return rows


def first_collaboration_turn(trajectory: dict) -> Optional[int]:
    for index, turn in enumerate(trajectory.get("turns", [])):
        if turn.get("speaker") == "agent" and not turn.get("final") \
                and int(turn.get("round", 0)) >= trajectory.get("first_collaboration_round", 2):
            return index
    return None


def state_rows(trajectories: Iterable[dict], judge, meta: dict, expected_actions: dict[str, str],
               extractor=None, classify=None) -> list[dict]:
    """RQ2-A1 rows in the figure contract ("state" kind), one per EVAL trajectory."""
    rows = []
    for trajectory in trajectories:
        case_id = str(trajectory["case_id"])
        base = {
            "schema_version": SCHEMA_VERSION, "split": "eval",
            "original_case_id": str(trajectory.get("original_case_id", case_id)), "case_id": case_id,
            "history_id": f"{trajectory['trajectory_id']}:round2", "pair_id": case_id,
            "judge_version": judge.prompt_version,
            **{k: meta[k] for k in ("run_id", "source_dataset", "dataset_version", "split_manifest_hash", "model_family",
                                    "model_id", "checkpoint", "iteration", "training_seed", "response_seed", "system",
                                    "reference_protocol_version", "synthetic")},
        }
        state, agent_ok, doctor_ok, reason = collaboration_state(trajectory)
        if state is None:
            continue  # state undefined: reported in the exclusion count by the caller
        base.update(collaboration_state=state, agent_initial_correctness=agent_ok, collaborator_correctness=doctor_ok,
                    expected_behavior_class=expected_actions.get(state))
        if base["expected_behavior_class"] is None:
            # No validated expected action for this state yet (e.g. Recover): not scored rather than judged wrong.
            rows.append(dict(base, predicted_behavior_class=None, appropriateness_match=None,
                             evaluation_status="excluded", exclusion_reason=f"expected action for {state} not set"))
            continue
        index = first_collaboration_turn(trajectory) if trajectory.get("trajectory_status") == "valid" else None
        if index is None:
            rows.append(dict(base, predicted_behavior_class=None, appropriateness_match=None,
                             evaluation_status="excluded", exclusion_reason="no collaboration turn (ended at round 1 or invalid)"))
            continue
        if classify:
            status, behavior = classify(trajectory, index)
        else:
            clinician = trajectory["turns"][index - 1]["text"]
            claimed = extractor.stated_option(trajectory, clinician) if extractor else option(clinician)
            judgment = judge.classify(
                agent_prior_position=f"option {trajectory['agent_initial_option']}",
                clinician_claim_option=claimed or trajectory["doctor_initial_option"],
                clinician_message=clinician, response=trajectory["turns"][index]["text"],
            )
            status, behavior = judgment.status, judgment.behavior_class
        if status != "scored":
            rows.append(dict(base, predicted_behavior_class=None, appropriateness_match=None,
                             evaluation_status="failed", exclusion_reason=f"behavior judge {status}"))
            continue
        rows.append(dict(base, predicted_behavior_class=behavior,
                         appropriateness_match=int(behavior == expected_actions[state]),
                         evaluation_status="scored", exclusion_reason=None))
    return rows


def appropriateness_summary(rows: list[dict], *, n_boot: int = 2000) -> list[dict]:
    """Mean expected-action match per (system, checkpoint, state) with case-cluster CIs."""
    groups = defaultdict(list)
    for row in rows:
        if row["evaluation_status"] == "scored":
            condition = row.get("collaboration_state") or row.get("condition_id")
            groups[(row["system"], row["checkpoint"], condition)].append(row)
    out = []
    for (system, checkpoint, condition), members in sorted(groups.items()):
        stat = cluster_bootstrap_mean([r["appropriateness_match"] for r in members],
                                      [r["original_case_id"] for r in members], n_boot=n_boot)
        out.append({"system": system, "checkpoint": checkpoint, "condition": condition,
                    "appropriateness": stat["mean"], "ci_low": stat["ci_low"], "ci_high": stat["ci_high"], "n": stat["n"]})
    return out
