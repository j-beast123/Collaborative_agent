"""Diagnostic rubric audit: Reason -> Jev applicability -> consistency -> score."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judges.jev import JevClient
from src.judges.llm import CachedChat, ChatRequest, build_chat_backend, parse_json_object


REASON_PROMPT = ROOT / "prompts/rubric_reason_assessor.txt"


def read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_trajectories(pattern: str, limit: int | None) -> list[dict]:
    seen, result = set(), []
    for name in sorted(glob.glob(str(ROOT / pattern))):
        for row in read_jsonl(Path(name)):
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id or trajectory_id in seen or row.get("trajectory_status") != "valid":
                continue
            seen.add(trajectory_id)
            result.append(row)
            if limit and len(result) >= limit:
                return result
    return result


def load_rubrics(path: str) -> list[dict]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    result = []
    for rubric in payload.get("rubrics", []):
        if not rubric.get("rubric_id") or not rubric.get("applicability_condition"):
            raise ValueError(f"invalid rubric in {path}: {rubric}")
        if set((rubric.get("scores") or {})) != {"0", "1", "2"}:
            raise ValueError(f"rubric lacks complete 0/1/2 scores: {rubric.get('rubric_id')}")
        result.append(rubric)
    if not result:
        raise ValueError(f"no rubrics in {path}")
    return result


def visible_case(trajectory: dict) -> dict:
    return {
        "case_id": trajectory.get("case_id"),
        "question": trajectory.get("question", ""),
        "scenario": trajectory.get("scenario", ""),
        "options": trajectory.get("options", {}),
    }


def visible_turn(turn: dict) -> dict:
    return {key: turn.get(key) for key in ("speaker", "text", "round")}


def reason_for(chat: CachedChat, system: str, trajectory: dict, history: list[dict], rubric: dict) -> tuple[str | None, dict]:
    user = json.dumps(
        {
            "case": visible_case(trajectory),
            "history_before_target_agent_response": history,
            "rubric": {
                "rubric_id": rubric["rubric_id"],
                "name": rubric["name"],
                "applicability_condition": rubric["applicability_condition"],
            },
        },
        ensure_ascii=False,
    )
    result = chat.complete(
        ChatRequest(
            system,
            "Assess this rubric condition from the visible history only.\n\n" + user,
            "rubric-reason-v1",
            {"temperature": 0, "max_tokens": 256},
        )
    )
    call = {key: value for key, value in result.to_dict().items() if key != "text"}
    if result.status != "ok":
        return None, {"status": "call_failed", "call": call, "error": result.error}
    parsed, parse_status = parse_json_object(result.text)
    if not parsed or not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        return None, {"status": "invalid", "call": call, "parse_status": parse_status, "raw_response": result.text}
    return parsed["reason"].strip()[:2000], {"status": "ok", "call": call}


def decide(client: JevClient, state: dict, questions: dict[str, dict], purpose: str) -> tuple[dict, dict | None, str | None]:
    try:
        answers, call = client.decide(json.dumps(state, ensure_ascii=False), questions, purpose=purpose)
        return answers, call, None
    except Exception as error:  # diagnostic record; one bad item must not lose the run
        return {}, None, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-glob", required=True)
    parser.add_argument("--rubric-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reason-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--reason-cache", default="outputs/judge_cache/rubric_reason_jev.jsonl")
    parser.add_argument("--jev-endpoint", required=True)
    parser.add_argument("--jev-model", required=True)
    parser.add_argument("--jev-api-key-env", default="JEV_API_KEY")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--include-initial-turn", action="store_true")
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectory_glob, args.max_trajectories)
    rubrics = load_rubrics(args.rubric_set)
    reason_chat = CachedChat(
        build_chat_backend({
            "backend": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "model_id": args.reason_model,
            "api_key_env": "OPENROUTER_API_KEY",
            "revision": "openrouter-unpinned",
        }),
        ROOT / args.reason_cache,
    )
    jev = JevClient(endpoint=args.jev_endpoint, model=args.jev_model, api_key_env=args.jev_api_key_env)
    reason_system = REASON_PROMPT.read_text(encoding="utf-8")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            turns = trajectory.get("turns", [])
            for turn_id, target in enumerate(turns):
                if target.get("speaker") != "agent" or target.get("final"):
                    continue
                if not args.include_initial_turn and int(target.get("round", 1)) < 2:
                    continue
                history = [visible_turn(turn) for turn in turns[:turn_id]]
                prefix_state = {
                    "case": visible_case(trajectory),
                    "history_before_target_agent_response": history,
                }

                assessments = {}
                reason_errors = {}
                for rubric in rubrics:
                    reason, metadata = reason_for(reason_chat, reason_system, trajectory, history, rubric)
                    assessments[rubric["rubric_id"]] = {
                        "rubric_id": rubric["rubric_id"],
                        "name": rubric["name"],
                        "reason": reason,
                        "reason_status": metadata["status"],
                        "reason_call": metadata.get("call"),
                    }
                    if metadata["status"] != "ok":
                        reason_errors[rubric["rubric_id"]] = metadata

                applicability_questions = {}
                for rubric in rubrics:
                    rid = rubric["rubric_id"]
                    if rid in reason_errors:
                        continue
                    applicability_questions[rid] = {
                        "type": "choice",
                        "instructions": (
                            "Judge applicability only from the history before the target Agent response. "
                            "The rubric condition must be supported by observable dialogue; topical similarity is not enough. "
                            f"Condition: {rubric['applicability_condition']}\n"
                            f"Reason: {assessments[rid]['reason']}"
                        ),
                        "criteria": {
                            "Y": "The condition is supported by the visible pre-response history.",
                            "N": "The condition is absent, unsupported, or cannot be identified from the visible pre-response history.",
                        },
                    }
                applicability_state = dict(prefix_state, assessments={rid: {
                    "reason": assessments[rid]["reason"]
                } for rid in applicability_questions})
                applicability, applicability_call, applicability_error = decide(
                    jev, applicability_state, applicability_questions, "rubric_applicability"
                ) if applicability_questions else ({}, None, None)

                for rid, answer in applicability.items():
                    assessments[rid]["applicability"] = answer["value"]
                    assessments[rid]["applicability_probabilities"] = answer.get("probabilities")
                if applicability_error:
                    for rid in applicability_questions:
                        assessments[rid]["applicability_status"] = "call_failed"
                else:
                    for rid in applicability_questions:
                        assessments[rid]["applicability_status"] = "scored"

                consistency_questions = {}
                for rid in applicability_questions:
                    if rid not in assessments or "applicability" not in assessments[rid]:
                        continue
                    consistency_questions[rid] = {
                        "type": "choice",
                        "instructions": (
                            "Check only whether the stated Reason logically supports the chosen applicability decision. "
                            "Do not use the target Agent response, future turns, final outcome, or ground truth.\n"
                            f"Reason: {assessments[rid]['reason']}\n"
                            f"Decision: {assessments[rid]['applicability']}"
                        ),
                        "criteria": {
                            "consistent": "The Reason and Y/N decision are logically compatible.",
                            "inconsistent": "The Reason and Y/N decision conflict or the Reason does not support the decision.",
                        },
                    }
                consistency_state = dict(prefix_state, assessments={rid: {
                    "reason": assessments[rid]["reason"],
                    "applicability": assessments[rid].get("applicability"),
                } for rid in consistency_questions})
                consistency, consistency_call, consistency_error = decide(
                    jev, consistency_state, consistency_questions, "rubric_reason_consistency"
                ) if consistency_questions else ({}, None, None)
                for rid, answer in consistency.items():
                    assessments[rid]["consistency"] = answer["value"]
                    assessments[rid]["consistency_probabilities"] = answer.get("probabilities")
                if consistency_error:
                    for rid in consistency_questions:
                        assessments[rid]["consistency_status"] = "call_failed"
                else:
                    for rid in consistency_questions:
                        assessments[rid]["consistency_status"] = "scored"

                score_questions = {}
                by_id = {rubric["rubric_id"]: rubric for rubric in rubrics}
                for rid, assessment in assessments.items():
                    if assessment.get("applicability") != "Y" or assessment.get("consistency") != "consistent":
                        continue
                    rubric = by_id[rid]
                    score_questions[rid] = {
                        "type": "choice",
                        "instructions": (
                            "Score only the target Agent response against this rubric. "
                            "Do not use final correctness, ground truth, or later turns. "
                            f"Applicability condition: {rubric['applicability_condition']}\n"
                            f"Reason: {assessment['reason']}"
                        ),
                        "criteria": rubric["scores"],
                    }
                score_state = dict(prefix_state, target_turn=visible_turn(target), assessments={rid: {
                    "reason": assessments[rid]["reason"],
                    "applicability": assessments[rid].get("applicability"),
                    "consistency": assessments[rid].get("consistency"),
                } for rid in score_questions})
                scores, score_call, score_error = decide(
                    jev, score_state, score_questions, "rubric_adherence_score"
                ) if score_questions else ({}, None, None)
                for rid, answer in scores.items():
                    assessments[rid]["score"] = int(answer["value"])
                    assessments[rid]["score_probabilities"] = answer.get("probabilities")
                applicable_scores = [a["score"] for a in assessments.values() if isinstance(a.get("score"), int)]
                row = {
                    "trajectory_id": trajectory["trajectory_id"],
                    "case_id": trajectory["case_id"],
                    "turn_id": turn_id,
                    "round": target.get("round"),
                    "assessments": assessments,
                    "summary": {
                        "rubric_count": len(rubrics),
                        "applicable_count": sum(a.get("applicability") == "Y" for a in assessments.values()),
                        "consistent_count": sum(a.get("consistency") == "consistent" for a in assessments.values()),
                        "scored_count": len(applicable_scores),
                        "score_0": applicable_scores.count(0),
                        "score_1": applicable_scores.count(1),
                        "score_2": applicable_scores.count(2),
                        "mean_score_0_to_2": sum(applicable_scores) / len(applicable_scores) if applicable_scores else None,
                    },
                    "calls": {
                        "applicability": applicability_call,
                        "applicability_error": applicability_error,
                        "consistency": consistency_call,
                        "consistency_error": consistency_error,
                        "score": score_call,
                        "score_error": score_error,
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    print(json.dumps({"output": str(output), "trajectories": len(trajectories), "rubrics": len(rubrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
