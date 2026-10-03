"""Effectiveness labels from continuation rollouts, per (history, candidate response) pair."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from src.collaboration.rerun import continue_collaboration
from src.collaboration.runtime import run_with_retries
from src.common.config import IntegrationUnavailable
from src.common.io import read_jsonl
from src.data.schema import digest
from src.judges.agent_state import is_collaboration_turn

from .continuation import ContinuationSpec
from .history import continuation_trajectory, history_at

LABEL_VERSION = "2"
# last_collaboration_turn: the agent turn of the final round, so the final decision follows the
# response directly and no later agent turn can undo it (diagnostic).
TURN_SCOPES = ("first_collaboration_turn", "all_collaboration_turns", "last_collaboration_turn")


CANDIDATE_FIELDS = ("candidate_type", "intended_behavior", "intended_behavior_class", "observed_behavior",
                    "gives_evidence", "compliance", "meta")


@dataclass(frozen=True)
class Candidate:
    history_id: str
    response_id: str
    response: str
    source: str
    info: dict = field(default_factory=dict)  # CANDIDATE_FIELDS from the candidates file, copied to the label row


def resolve_labels_section(config: dict) -> dict:
    """The config's labels section; include_initial_turn must be explicit in production."""
    section = dict(config.get("labels") or {})
    if section.get("include_initial_turn") is None:
        if config["mode"] == "production":
            raise IntegrationUnavailable("METHOD_TBD: set labels.include_initial_turn (true/false) for production")
        section["include_initial_turn"] = False
    return section


def collect_histories(trajectories: Iterable[dict], *, turn_scope: str, include_initial_turn: bool,
                      max_histories: Optional[int] = None) -> list[tuple[dict, dict]]:
    """[(history, ground_truth)] from valid trajectories, in the configured turn scope."""
    if turn_scope not in TURN_SCOPES:
        raise ValueError(f"labels.turn_scope must be one of {TURN_SCOPES}")
    selected = []
    for trajectory in trajectories:
        if trajectory.get("trajectory_status") != "valid" or not (trajectory.get("ground_truth") or {}).get("correct_option"):
            continue
        indices = [i for i in range(len(trajectory["turns"]))
                   if is_collaboration_turn(trajectory, i, include_initial_turn=include_initial_turn)]
        if turn_scope == "first_collaboration_turn":
            indices = indices[:1]
        elif turn_scope == "last_collaboration_turn":
            final_round = int(trajectory["max_discussion_rounds"])
            indices = [i for i in indices[-1:] if int(trajectory["turns"][i]["round"]) == final_round]
        for index in indices:
            selected.append((history_at(trajectory, index), trajectory["ground_truth"]))
        if max_histories is not None and len(selected) >= max_histories:
            return selected[:max_histories]
    return selected


def original_candidates(trajectories: Iterable[dict], histories: list[tuple[dict, dict]]) -> list[Candidate]:
    """The response actually given in each trajectory at the history's turn."""
    by_id = {str(t["trajectory_id"]): t for t in trajectories}
    candidates = []
    for history, _ in histories:
        index = int(history["history_id"].rsplit(":turn", 1)[1])
        text = by_id[history["source_trajectory_id"]]["turns"][index]["text"]
        candidates.append(Candidate(history["history_id"], "original", text, "trajectory",
                                    {"candidate_type": "natural"}))
    return candidates


def load_candidates(path: str | Path, *, require_compliance: bool = False) -> list[Candidate]:
    """Extra candidate responses: JSONL {history_id, response_id, response, source, CANDIDATE_FIELDS...}."""
    rows = [r for r in read_jsonl(path) if not (require_compliance and r.get("compliance") is False)]
    return [Candidate(r["history_id"], r["response_id"], r["response"], r.get("source", "candidates_file"),
                      {k: r[k] for k in CANDIDATE_FIELDS if k in r}) for r in rows]


def label_metadata(spec: ContinuationSpec, config_hash: str, seed_control: dict) -> dict[str, Any]:
    return {
        "agent_model": spec.agent_model.__dict__, "doctor_model": spec.doctor_model.__dict__,
        "reconstructor_model": spec.reconstructor_model.__dict__,
        "continuation_policy": "src/collaboration/rerun.py:continue_collaboration",
        "max_rounds": spec.max_rounds, "turn_order": list(spec.turn_order) if spec.turn_order else None,
        "stopping_rule": spec.stopping_rule, "final_decision_role": spec.final_decision_role,
        "decoding_config": spec.decoding, "seed_policy": spec.seed_policy.__dict__, "seed_control": seed_control,
        "success_criterion": spec.success_criterion, "continuation_protocol_hash": spec.protocol_hash(),
        "config_hash": config_hash, "label_version": LABEL_VERSION,
    }


class _Jsonl:
    def __init__(self, path: Path, key: Optional[Callable[[dict], Any]] = None):
        self.path = path
        self.rows = read_jsonl(path, missing_ok=True)
        self.keys = {key(r) for r in self.rows} if key else set()
        self.lock = threading.Lock()  # rollouts may run in threads

    def append(self, row: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows.append(row)


def _rollout(trajectory: dict, index: int, response: Optional[str], history_id: str, state_id: str,
             rollout_index: int, seed: int, runtime, spec: ContinuationSpec, set_seed, log: _Jsonl,
             max_attempts: int) -> dict:
    """One continuation."""
    rollout_id = f"{state_id}#r{rollout_index}"
    summary = {}

    def call():
        if set_seed:
            set_seed(seed)
        return continue_collaboration(trajectory, index, runtime, fixed_response=response, allow_initial_turn=True)

    def record(attempt, result):
        summary.update({"rollout_id": rollout_id, "seed": seed, "seed_applied": bool(set_seed), "attempt": attempt,
                        "status": result["status"], "final_option": result.get("final_option"),
                        "correct": result.get("correct"), "termination_reason": result.get("termination_reason")})
        log.append(dict(summary, state_id=state_id, history_id=history_id,
                        protocol_hash=spec.protocol_hash(), dialogue=result.get("dialogue")))

    run_with_retries(call, format_attempts=max_attempts, on_attempt=record)
    return summary


def _rollouts(history: dict, ground_truth: dict, response: Optional[str], state_id: str, runtime,
              spec: ContinuationSpec, set_seed, log: _Jsonl, max_attempts: int = 1,
              pool: Optional[ThreadPoolExecutor] = None) -> tuple[Optional[dict], list[dict]]:
    """K continuations; returns ({successes, trials, rollout_ids} or None if incomplete, summaries)."""
    if history["round"] > spec.max_rounds:
        raise ValueError(f"history {history['history_id']} is at round {history['round']} > max_rounds {spec.max_rounds}")
    trajectory, index = continuation_trajectory(history, ground_truth, max_rounds=spec.max_rounds,
                                                final_decision_role=spec.final_decision_role, response=response)
    jobs = list(enumerate(spec.rollout_seeds(history["history_id"], state_id)))

    def run(job):
        return _rollout(trajectory, index, response, history["history_id"], state_id, job[0], job[1], runtime, spec,
                        set_seed, log, max_attempts)

    summaries = list(pool.map(run, jobs)) if pool else [run(job) for job in jobs]
    if any(s["status"] != "completed" for s in summaries):
        return None, summaries
    return {"successes": sum(bool(s["correct"]) for s in summaries), "trials": len(summaries),
            "rollout_ids": [s["rollout_id"] for s in summaries]}, summaries


def build_pair_labels(
    histories: list[tuple[dict, dict]],
    candidates: list[Candidate],
    runtime,
    spec: ContinuationSpec,
    *,
    split: str,
    output_path: str | Path,
    metadata: dict[str, Any],
    set_seed: Optional[Callable[[int], None]] = None,
    max_rollout_attempts: int = 1,
    parallel_rollouts: int = 1,
) -> dict[str, int]:
    """Label every (history, candidate) pair; resumable; BEFORE shared per history."""
    if split not in ("train", "dev"):
        raise ValueError("Effectiveness labels are built for TRAIN or DEV only, never EVAL")
    if spec.k is None or spec.max_rounds is None or spec.final_decision_role is None \
            or spec.seed_policy.mode is None or spec.seed_policy.base_seed is None:
        raise ValueError("ContinuationSpec needs k, max_rounds, final_decision_role and a seed policy")
    if parallel_rollouts > 1 and set_seed is not None:
        raise ValueError("per-rollout seeding is not possible with parallel rollouts; pass set_seed=None")
    output = Path(output_path)
    labels = _Jsonl(output, key=lambda r: (r["history_id"], r["response_id"]))
    before_cache = _Jsonl(output.with_name(output.name + ".before_cache.jsonl"), key=lambda r: r["history_id"])
    rollouts = _Jsonl(output.with_name(output.name + ".rollouts.jsonl"))
    incomplete = _Jsonl(output.with_name(output.name + ".incomplete.jsonl"))
    protocol = spec.protocol_hash()
    cached = {r["history_id"]: r for r in before_cache.rows if r["protocol_hash"] == protocol}
    by_history = {h["history_id"]: (h, gt) for h, gt in histories}
    counts = {"labels_written": 0, "skipped_done": 0, "before_computed": 0, "before_cached": 0,
              "incomplete": 0, "unknown_history": 0}

    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.history_id not in by_history:
            counts["unknown_history"] += 1
        elif (candidate.history_id, candidate.response_id) in labels.keys or any(
                c.response_id == candidate.response_id for c in groups.get(candidate.history_id, [])):
            counts["skipped_done"] += 1  # labelled already, or a duplicate row in the candidates
        else:
            groups.setdefault(candidate.history_id, []).append(candidate)

    rollout_pool = ThreadPoolExecutor(parallel_rollouts) if parallel_rollouts > 1 else None
    try:
        for history_id, group in groups.items():
            history, ground_truth = by_history[history_id]
            states = [] if history_id in cached else [(None, None, f"{history_id}:before")]
            states += [(c, c.response, f"{history_id}:after:{c.response_id}") for c in group]

            def run(state):
                return _rollouts(history, ground_truth, state[1], state[2], runtime, spec, set_seed, rollouts,
                                 max_rollout_attempts, rollout_pool)

            if rollout_pool:
                with ThreadPoolExecutor(len(states)) as state_pool:
                    results = list(state_pool.map(run, states))
            else:
                results = []
                for state in states:  # sequential: stop early if BEFORE is incomplete
                    results.append(run(state))
                    if state[0] is None and results[-1][0] is None:
                        break
            by_state = {state[2]: result for state, result in zip(states, results)}

            if history_id not in cached:
                before, summaries = by_state[f"{history_id}:before"]
                if before is None:
                    counts["incomplete"] += len(group)
                    incomplete.append({"history_id": history_id, "state": "before", "rollouts": summaries})
                    continue
                cached[history_id] = dict(before, history_id=history_id, protocol_hash=protocol)
                before_cache.append(cached[history_id])
                counts["before_computed"] += 1
                counts["before_cached"] += len(group) - 1
            else:
                counts["before_cached"] += len(group)
            before = cached[history_id]
            for candidate in group:
                after, summaries = by_state[f"{history_id}:after:{candidate.response_id}"]
                if after is None:
                    counts["incomplete"] += 1
                    incomplete.append({"history_id": history_id, "response_id": candidate.response_id,
                                       "state": "after", "rollouts": summaries})
                    continue
                labels.append({
                    "case_id": history["case_id"], "original_case_id": history["original_case_id"],
                    "history_id": history_id, "response_id": candidate.response_id,
                    "response_source": candidate.source, "split": split,
                    "history": history, "response": candidate.response,
                    "before": {k: before[k] for k in ("successes", "trials", "rollout_ids")},
                    "after": after,
                    "before_rate": before["successes"] / before["trials"],
                    "after_rate": after["successes"] / after["trials"],
                    "delta_effectiveness": after["successes"] / after["trials"] - before["successes"] / before["trials"],
                    **candidate.info,
                    "metadata": metadata,
                })
                labels.keys.add((history_id, candidate.response_id))
                counts["labels_written"] += 1
    finally:
        if rollout_pool:
            rollout_pool.shutdown()
    return counts


def config_hash(config: dict) -> str:
    return digest({k: v for k, v in config.items() if k != "continuation_spec"})
