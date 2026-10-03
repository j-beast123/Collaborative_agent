"""Validate rubric candidates with independent rerun workers.

Each worker loads its own local agent runtime on the visible GPU and validates
every ``num_shards``-th candidate. The candidate generation stage is run once;
this script is intended to be launched once per GPU with distinct outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaboration.rerun import rerun_from_turn
from src.collaboration.runtime import build_runtime, run_with_retries
from src.judges.failure_analysis import load_latest
from src.judges.rubric_validation import validate_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Valid trajectory JSONL used to resolve candidate source trajectories.",
    )
    parser.add_argument(
        "--failures",
        required=True,
        help="Invalid trajectory JSONL retained in the source collection.",
    )
    parser.add_argument("--candidate-input", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--agent-model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--reconstructor-model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--doctor-prompt", default="prompts/user_simulator.txt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reconstruction-max-tokens", type=int, default=2048)
    parser.add_argument("--num-reruns", type=int, default=3)
    parser.add_argument("--min-successes", type=int, default=2)
    parser.add_argument(
        "--rerun-agent-turns",
        type=int,
        default=None,
        help="Limit each rerun to this many Agent responses before the Doctor final decision.",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=7)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")

    trajectories = read_jsonl(ROOT / args.trajectories) + read_jsonl(ROOT / args.failures)
    by_trajectory = {str(trajectory["trajectory_id"]): trajectory for trajectory in trajectories}
    latest = load_latest(ROOT / args.candidate_input)
    records = sorted(
        (record for record in latest.values() if record.get("analysis_status") == "valid"),
        key=lambda record: str(record["rubric_id"]),
    )
    shard_records = records[args.shard_index :: args.num_shards]
    missing = [record["source_trajectory_id"] for record in shard_records
               if record["source_trajectory_id"] not in by_trajectory]
    if missing:
        raise SystemExit(f"Candidate source trajectories are missing: {missing}")

    print(json.dumps({
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_candidates": len(records),
        "shard_candidates": len(shard_records),
        "validation_output": args.validation_output,
    }, ensure_ascii=False), flush=True)
    if not shard_records:
        return

    runtime = build_runtime(
        agent_model=args.agent_model,
        doctor_model=args.doctor_model,
        reconstructor_model=args.reconstructor_model,
        max_tokens=args.max_tokens,
        reconstruction_max_tokens=args.reconstruction_max_tokens,
        root=ROOT,
        doctor_prompt=args.doctor_prompt,
    )

    def rerun(trajectory: dict, turn_index: int, instruction: str, rerun_index: int) -> dict:
        del rerun_index
        return run_with_retries(
            lambda: rerun_from_turn(
                trajectory,
                turn_index,
                instruction,
                runtime,
                allow_initial_turn=True,
                max_agent_turns=args.rerun_agent_turns,
            ),
            format_attempts=3,
        )

    try:
        counts = validate_candidates(
            shard_records,
            by_trajectory,
            rerun,
            num_reruns=args.num_reruns,
            min_successes=args.min_successes,
            output_path=ROOT / args.validation_output,
        )
    finally:
        if runtime.close:
            runtime.close()
    print(json.dumps({"shard_index": args.shard_index, "counts": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
