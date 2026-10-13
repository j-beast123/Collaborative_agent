"""Send failed collaboration trajectories to the failure analyzer (rubric generator)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional

from src.common import usage
from src.data.schema import digest, identity

from .agent_state import is_collaboration_turn, pre_response_state
from .llm import CachedChat, ChatRequest, parse_json_object
from .rubric_schema import build_levels, validate_rubric

REQUIRED_TEXT = (
    "user_side_behavior", "user_contribution", "agent_side_behavior", "interaction_failure",
    "ideal_response_behavior", "rubric_name", "applicability", "score_0", "score_1", "score_2",
    # Added by prompts/rubric_agent_state_addendum.txt
    "user_side_context", "agent_side_state",
)
NULLABLE_TEXT = ("surface_request", "decision_shaping_contribution", "surface_handling", "reasoning_handling")
RETRYABLE_ANALYSIS_PROBLEMS = frozenset({"invalid_json", "no_response"})


def select_failures(
    trajectories: Iterable[dict],
    *,
    manifest: Optional[dict] = None,
    source_dataset: Optional[str] = None,
) -> tuple[list[dict], dict[str, int]]:
    """Valid, wrong-answer trajectories; with a manifest, TRAIN cases only."""
    selected, counts = [], {"total": 0, "invalid_trajectory": 0, "correct": 0, "not_train": 0, "failed": 0}
    for trajectory in trajectories:
        counts["total"] += 1
        if trajectory.get("trajectory_status") != "valid" or "is_correct" not in trajectory:
            counts["invalid_trajectory"] += 1
            continue
        if trajectory["is_correct"]:
            counts["correct"] += 1
            continue
        if manifest is not None:
            member = manifest["members"].get(identity(source_dataset, str(trajectory["case_id"])))
            if member is None or member["split"] != "train":
                counts["not_train"] += 1
                continue
        counts["failed"] += 1
        selected.append(trajectory)
    return selected, counts


def analyzer_input(trajectory: dict) -> dict[str, Any]:
    """Trajectory view for the analyzer, with turn_id = index in turns."""
    turns = []
    for index, turn in enumerate(trajectory.get("turns", [])):
        view = {"turn_id": index, "speaker": turn.get("speaker"), "text": turn.get("text", "")}
        if turn.get("speaker") == "agent":
            view["pre_response_state"] = pre_response_state(trajectory, index)
        turns.append(view)
    return {
        "case_id": trajectory["case_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "scenario": trajectory.get("scenario", ""),
        "question": trajectory.get("question", ""),
        "options": trajectory.get("options", {}),
        # The correct answer is shown only for the multiple-choice protocol. AgentClinic trajectories are
        # analysed without the correct diagnosis: it is evaluator-only and must not shape a rubric.
        **({} if trajectory.get("protocol") == "agentclinic_two_turn"
           else {"ground_truth": trajectory.get("ground_truth"), "final_option": trajectory.get("final_option")}),
        **{k: trajectory[k] for k in ("final_diagnosis", "doctor_information") if k in trajectory},
        **({} if trajectory.get("protocol") == "agentclinic_two_turn" else {"is_correct": trajectory["is_correct"]}),
        "turns": turns,
    }


def candidate_rubric(analysis: dict) -> dict[str, Any]:
    """Self-contained rubric from an analysis: one shared context, three behaviors."""
    return {
        "name": analysis["rubric_name"],
        "applicability": {"user_side_context": analysis["user_side_context"],
                          "agent_side_state": analysis["agent_side_state"]},
        "scores": build_levels(analysis["user_side_context"], analysis["agent_side_state"],
                               {level: analysis[f"score_{level}"] for level in ("0", "1", "2")}),
    }


def allowed_failure_turns(trajectory: dict, include_initial_turn: bool) -> list[int]:
    """Agent turns that may be named as failure/intervention turns (the scored turn scope)."""
    return [i for i in range(len(trajectory.get("turns", [])))
            if is_collaboration_turn(trajectory, i, include_initial_turn=include_initial_turn)]


def validate_analysis(analysis: dict, trajectory: dict, include_initial_turn: bool = False) -> list[str]:
    """Check schema and that 'where' claims point to real turns and exact quotes."""
    problems = []
    turns = trajectory.get("turns", [])
    for name in REQUIRED_TEXT:
        if not isinstance(analysis.get(name), str) or not analysis[name].strip():
            problems.append(f"missing {name}")
    for name in NULLABLE_TEXT:
        if name not in analysis or not (analysis[name] is None or isinstance(analysis[name], str)):
            problems.append(f"invalid {name}")

    allowed = set(allowed_failure_turns(trajectory, include_initial_turn))

    def is_allowed(value):
        return type(value) is int and value in allowed

    if not is_allowed(analysis.get("intervention_turn_id")):
        problems.append("intervention_turn_id is not an allowed collaboration turn")
    failure_turns = analysis.get("failure_turn_ids")
    if not isinstance(failure_turns, list) or not failure_turns or not all(is_allowed(t) for t in failure_turns):
        problems.append("failure_turn_ids must be nonempty allowed collaboration turn ids")
    spans = analysis.get("evidence_spans")
    if not isinstance(spans, list) or not spans:
        problems.append("evidence_spans missing")
        spans = []
    speakers = set()
    for span in spans:
        turn_id = span.get("turn_id") if isinstance(span, dict) else None
        if type(turn_id) is not int or not 0 <= turn_id < len(turns):
            problems.append(f"evidence turn_id {turn_id} out of range")
            continue
        turn = turns[turn_id]
        if span.get("speaker") != turn.get("speaker"):
            problems.append(f"evidence speaker mismatch at turn {turn_id}")
        quote = span.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in turn.get("text", ""):
            problems.append(f"evidence quote not found verbatim at turn {turn_id}")
        speakers.add(turn.get("speaker"))
    if spans and "agent" not in speakers:
        problems.append("no agent evidence quote")
    for name in ("case_id", "trajectory_id"):
        if str(analysis.get(name)) != str(trajectory[name]):
            problems.append(f"{name} does not match the trajectory")
    return problems


def load_latest(path: str | Path) -> dict[str, dict]:
    """Latest analysis record per source trajectory."""
    path = Path(path)
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["source_trajectory_id"]] = record
    return latest


def analyze_failures(
    trajectories: list[dict],
    chat: CachedChat,
    *,
    system_prompt: str,
    prompt_version: str,
    params: dict,
    output_path: str | Path,
    include_initial_turn: bool = False,
    workers: int = 1,
) -> dict[str, int]:
    """Analyze each failed trajectory once; resumable by trajectory_id.

    API calls may run concurrently, while JSONL writes remain in the caller
    thread so a resumed run never interleaves partial records.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output = Path(output_path)
    done = set()
    for trajectory_id, record in load_latest(output).items():
        if record["analysis_status"] == "valid":
            done.add(trajectory_id)
            continue
        problems = set(record.get("validation_problems") or [])
        if record["analysis_status"] == "invalid" and not problems.intersection(RETRYABLE_ANALYSIS_PROBLEMS):
            done.add(trajectory_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = [trajectory for trajectory in trajectories if str(trajectory["trajectory_id"]) not in done]
    counts = {"skipped_done": len(trajectories) - len(pending), "valid": 0, "invalid": 0, "call_failed": 0}

    def analyze_one(trajectory: dict) -> tuple[str, dict]:
        allowed = allowed_failure_turns(trajectory, include_initial_turn)
        user = (
            "Analyze this failed collaboration trajectory and extract exactly one reusable interaction failure. "
            "Do not define failure as final incorrectness alone. turn_id is the index in turns. "
            f"intervention_turn_id and failure_turn_ids must be chosen from these agent turns: {allowed}.\n\n"
            + json.dumps(analyzer_input(trajectory), ensure_ascii=False)
        )
        with usage.case(trajectory["case_id"]):
            result = chat.complete(ChatRequest(system_prompt, user, prompt_version, params))
        parsed, parse_status = parse_json_object(result.text) if result.status == "ok" else (None, "call_failed")
        problems = validate_analysis(parsed, trajectory, include_initial_turn) if parsed is not None else [parse_status]
        rubric = None
        if parsed is not None and not problems:
            rubric = candidate_rubric(parsed)
            problems = validate_rubric(rubric)
        status = "call_failed" if result.status != "ok" else ("valid" if not problems else "invalid")
        record = {
            "source_case_id": str(trajectory["case_id"]),
            "source_trajectory_id": str(trajectory["trajectory_id"]),
            "analysis_status": status,
            "validation_problems": problems,
            "analysis": parsed,
            "candidate_rubric": rubric,
            "raw_response": result.text,
            "call": {k: v for k, v in result.to_dict().items() if k != "text"},
            "prompt_version": prompt_version,
            "prompt_sha256": digest(system_prompt),
        }
        if parsed is not None and not problems:
            # One candidate per failed trajectory: the id includes the source, so
            # identical wording from two failures still counts as two failures.
            record["rubric_id"] = "cand-" + digest({
                "source_trajectory_id": record["source_trajectory_id"], "rubric": rubric})[:12]
        return status, record

    if workers == 1:
        results = map(analyze_one, pending)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(analyze_one, pending)
    try:
        with output.open("a", encoding="utf-8") as handle:
            for status, record in results:
                counts[status] += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        if workers != 1:
            pool.shutdown(wait=True)
    return counts
