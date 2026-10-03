"""RQ2-B effectiveness by matched continuation (EVAL, actual outcomes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from src.collaboration.rerun import agent_response_at, continue_collaboration
from src.collaboration.runtime import run_with_retries
from src.common.io import read_jsonl

from .stats import paired_difference


def generate_responses(histories: list[tuple[dict, int]], runtime, system_name: str, *,
                       format_attempts: int = 1) -> dict[str, dict]:
    """Step 1, once per system: the system's response to each shared history."""
    responses = {}
    for trajectory, turn_index in histories:
        history_id = f"{trajectory['trajectory_id']}:turn{turn_index}"
        generated = run_with_retries(lambda: agent_response_at(trajectory, turn_index, runtime),
                                     format_attempts=format_attempts)
        responses[history_id] = {"system": system_name, "status": generated["status"], "response": generated.get("response")}
    return responses


def run_continuations(
    histories: list[tuple[dict, int]],
    responses_by_system: dict[str, dict[str, dict]],
    continuation_runtime,
    *,
    k: int,
    seeds_for: Callable[[str], list[int]],
    set_seed: Optional[Callable[[int], None]] = None,
    format_attempts: int = 1,
    progress_path: Optional[str | Path] = None,
) -> list[dict]:
    """Step 2, one continuation runtime: K continuations of every fixed response."""
    done = {(r["history_id"], r["system"]): r for r in read_jsonl(progress_path, missing_ok=True)} \
        if progress_path else {}

    def continuation(trajectory, turn_index, response, seed):
        if set_seed:
            set_seed(seed)
        return continue_collaboration(trajectory, turn_index, continuation_runtime, fixed_response=response)

    rows = []
    for trajectory, turn_index in histories:
        history_id = f"{trajectory['trajectory_id']}:turn{turn_index}"
        seeds = seeds_for(history_id)
        if len(seeds) != k:
            raise ValueError("seeds_for must return exactly k seeds")
        for system_name, responses in responses_by_system.items():
            if (history_id, system_name) in done:
                rows.append(done[(history_id, system_name)])
                continue
            generated = responses[history_id]
            row = {"history_id": history_id, "case_id": str(trajectory["case_id"]),
                   "original_case_id": str(trajectory.get("original_case_id", trajectory["case_id"])),
                   "system": system_name, "k": k, "response": generated["response"], "outcomes": []}
            if generated["status"] != "completed":
                row.update(status=generated["status"], success_rate=None)
            else:
                for seed in seeds:
                    result = run_with_retries(
                        lambda: continuation(trajectory, turn_index, generated["response"], seed),
                        format_attempts=format_attempts)
                    row["outcomes"].append({"seed": seed, "status": result["status"], "correct": result.get("correct"),
                                            "final_option": result.get("final_option")})
                completed = [o for o in row["outcomes"] if o["status"] == "completed"]
                row.update(status="completed", num_completed=len(completed),
                           num_success=sum(bool(o["correct"]) for o in completed),
                           success_rate=(sum(bool(o["correct"]) for o in completed) / len(completed))
                           if completed else None)
            rows.append(row)
            if progress_path:
                with Path(progress_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize(rows: list[dict], a: str = "base", b: str = "serl") -> dict:
    """Paired success difference (b - a) over histories where both branches completed."""
    by_history = {}
    for row in rows:
        by_history.setdefault(row["history_id"], {})[row["system"]] = row
    pairs, incomplete = [], 0
    for systems in by_history.values():
        if a in systems and b in systems and systems[a].get("success_rate") is not None \
                and systems[b].get("success_rate") is not None:
            pairs.append((systems[a]["original_case_id"], systems[a]["success_rate"], systems[b]["success_rate"]))
        else:
            incomplete += 1
    result = paired_difference(pairs)
    result.update(histories=len(by_history), incomplete_pairs=incomplete,
                  mean_success_a=(sum(p[1] for p in pairs) / len(pairs)) if pairs else None,
                  mean_success_b=(sum(p[2] for p in pairs) / len(pairs)) if pairs else None, a=a, b=b)
    return result
