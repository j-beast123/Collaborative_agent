"""Generate rubric candidates from failed pilot trajectories and validate them by rerun.

This adapter is for the Gemma12B--Qwen3-8B MedMCQA pilot output.  It keeps the
existing evaluator prompt, failure-analysis schema, collaboration runtime, and
rerun protocol; it only supplies the pilot JSONL as the trajectory source.
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
from src.judges.failure_analysis import analyze_failures, load_latest, select_failures
from src.judges.llm import CachedChat, build_chat_backend
from src.judges.rubric_validation import validate_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="outputs/collaboration/gemma12b_qwen3_8b/trajectories_10.jsonl",
        help="Valid pilot trajectories JSONL.",
    )
    parser.add_argument(
        "--failures",
        default="outputs/collaboration/gemma12b_qwen3_8b/trajectories_10_failures.jsonl",
        help="Invalid pilot trajectories JSONL; these are retained in the input but not rubric-selected.",
    )
    parser.add_argument(
        "--candidate-output",
        default="outputs/rubric_candidates/rubric_candidates_gpt56_sol.jsonl",
    )
    parser.add_argument(
        "--validation-output",
        default="outputs/rubric_validation/gemma12b_qwen3_8b/gpt56_sol_validations.jsonl",
    )
    parser.add_argument(
        "--cache",
        default="outputs/rubric_candidates/gpt56_sol_cache.jsonl",
        help="Successful evaluator-call cache.",
    )
    parser.add_argument("--rubric-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--agent-model", default="/home/kai/models/Qwen3-8B")
    parser.add_argument("--doctor-model", default="google/gemma-3-12b-it")
    parser.add_argument("--reconstructor-model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--doctor-prompt", default="prompts/user_simulator.txt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--rubric-max-tokens",
        type=int,
        default=5000,
        help="Maximum completion tokens for the rubric evaluator.",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=1,
        help="Concurrent rubric evaluator calls; keep within the provider rate limit.",
    )
    parser.add_argument("--reconstruction-max-tokens", type=int, default=2048)
    parser.add_argument("--num-reruns", type=int, default=3)
    parser.add_argument("--min-successes", type=int, default=2)
    parser.add_argument(
        "--rerun-agent-turns",
        type=int,
        default=None,
        help="Limit each rerun to this many Agent responses before the Doctor final decision.",
    )
    parser.add_argument(
        "--reset-candidates",
        action="store_true",
        help="Delete the existing candidate JSONL before generating candidates.",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Generate rubric candidates without loading the local agent or running validation reruns.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_text() -> str:
    prompt = (ROOT / "prompts/rubric_evaluator.txt").read_text(encoding="utf-8")
    addendum = ROOT / "prompts/rubric_agent_state_addendum.txt"
    if addendum.exists():
        prompt += addendum.read_text(encoding="utf-8")
    return prompt


def main() -> None:
    args = parse_args()
    candidate_output = ROOT / args.candidate_output
    validation_output = ROOT / args.validation_output

    if args.reset_candidates and candidate_output.exists():
        candidate_output.unlink()

    trajectories = read_jsonl(ROOT / args.trajectories) + read_jsonl(ROOT / args.failures)
    failures, selection_counts = select_failures(trajectories)
    if not failures:
        raise SystemExit(f"No valid incorrect trajectories found. Selection counts: {selection_counts}")

    analyzer_settings = {
        "backend": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "model_id": args.rubric_model,
        "api_key_env": "OPENROUTER_API_KEY",
        "revision": "openrouter-unpinned",
        "max_retries": 5,
    }
    chat = CachedChat(build_chat_backend(analyzer_settings), ROOT / args.cache)
    analysis_counts = analyze_failures(
        failures,
        chat,
        system_prompt=prompt_text(),
        prompt_version="rubric-evaluator-v1+agent-state-v1",
        params={"temperature": 0, "max_tokens": args.rubric_max_tokens},
        output_path=candidate_output,
        include_initial_turn=True,
        workers=args.analysis_workers,
    )

    latest = load_latest(candidate_output)
    records = [
        record
        for trajectory in failures
        if (record := latest.get(str(trajectory["trajectory_id"]))) is not None
        and record.get("analysis_status") == "valid"
    ]
    print(json.dumps({"selection": selection_counts, "analysis": analysis_counts, "valid_candidates": len(records)}))
    if not records:
        print(f"No valid rubric candidates. Inspect {candidate_output}")
        return
    if args.candidates_only:
        print(f"Candidate generation complete; validation skipped. Candidates: {candidate_output}")
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
    by_trajectory = {str(trajectory["trajectory_id"]): trajectory for trajectory in failures}

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
        validation_counts = validate_candidates(
            records,
            by_trajectory,
            rerun,
            num_reruns=args.num_reruns,
            min_successes=args.min_successes,
            output_path=validation_output,
        )
    finally:
        if runtime.close:
            runtime.close()

    summary = {
        "trajectories": args.trajectories,
        "failures": args.failures,
        "candidate_output": str(candidate_output),
        "validation_output": str(validation_output),
        "rubric_model": args.rubric_model,
        "selection": selection_counts,
        "analysis": analysis_counts,
        "validation": validation_counts,
        "num_reruns": args.num_reruns,
        "min_successes": args.min_successes,
    }
    summary_path = validation_output.with_name(validation_output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
