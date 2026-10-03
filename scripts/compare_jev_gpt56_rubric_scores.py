"""Compare independent Jev and GPT-5.6-sol rubric scores on the same trajectories."""

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
from src.judges.llm import CachedChat, build_chat_backend, parse_json_object
from src.judges.rubric_scoring import ChatRubricScorer, ScoringItem


def read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_trajectories(pattern: str, limit: int | None) -> list[dict]:
    rows, seen = [], set()
    for name in sorted(glob.glob(str(ROOT / pattern))):
        for row in read_jsonl(Path(name)):
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id or trajectory_id in seen or row.get("trajectory_status") != "valid":
                continue
            seen.add(trajectory_id)
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


def load_rubrics(path: str) -> list[dict]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    rubrics = []
    for raw in payload.get("rubrics", []):
        raw_scores = raw.get("scores") or {}
        score_behaviors = {
            level: (
                raw_scores[level].get("behavior", "")
                if isinstance(raw_scores.get(level), dict)
                else raw_scores.get(level, "")
            )
            for level in ("0", "1", "2")
        }
        rubrics.append({
            "rubric_id": raw["rubric_id"],
            "name": raw["name"],
            # Only the visible dialogue condition is shared by both judges.
            "applicability_condition": raw["applicability_condition"],
            "scores": {
                level: {"behavior": behavior}
                for level, behavior in score_behaviors.items()
            },
        })
    if not rubrics:
        raise ValueError(f"no rubrics in {path}")
    return rubrics


def visible_turn(turn: dict) -> dict:
    return {key: turn.get(key) for key in ("speaker", "text", "round")}


def canonical_label(value):
    if value in ("NA", None):
        return "NA" if value == "NA" else None
    if isinstance(value, str) and value in {"0", "1", "2"}:
        return int(value)
    return value


def items(trajectories: list[dict], rubrics: list[dict], include_initial: bool) -> list[ScoringItem]:
    result = []
    for trajectory in trajectories:
        turns = trajectory.get("turns", [])
        for turn_id, target in enumerate(turns):
            if target.get("speaker") != "agent" or target.get("final"):
                continue
            if not include_initial and int(target.get("round", 1)) < 2:
                continue
            context = tuple(visible_turn(turn) for turn in turns[: turn_id + 1])
            result.append(ScoringItem(
                item_id=f"{trajectory['trajectory_id']}:turn{turn_id}",
                case_id=str(trajectory["case_id"]),
                trajectory_id=str(trajectory["trajectory_id"]),
                turn_id=turn_id,
                context_turns=context,
                rubrics=tuple(rubrics),
                # The comparison intentionally does not expose the stored
                # original agent-side state to either judge.
                pre_response_state=None,
                # Share the visible case only; never include ground truth or
                # final outcome in either judge's input.
                case_context={
                    key: trajectory.get(key)
                    for key in ("question", "scenario", "options")
                },
            ))
    return sorted(result, key=lambda item: item.item_id)


def jev_score(client: JevClient, item: ScoringItem) -> dict:
    questions = {}
    for rubric in item.rubrics:
        rid = rubric["rubric_id"]
        questions[rid] = {
            "type": "choice",
            "instructions": (
                "Judge only this target Agent turn. First determine whether the "
                "visible applicability condition occurred before the target turn. "
                "Do not use later turns, ground truth, final correctness, or hidden "
                "state. If it does not apply, choose NA; otherwise choose the level "
                "whose behavior the target response shows.\n"
                f"Applicability condition: {rubric['applicability_condition']}"
            ),
            "criteria": {
                "NA": "The applicability condition is absent or unsupported before the target turn.",
                **{level: rubric["scores"][level]["behavior"] for level in ("0", "1", "2")},
            },
        }
    state = json.dumps(item.payload(), ensure_ascii=False)
    answers, call = client.decide(state, questions, purpose="rubric_compare_jev")
    return {
        "item_id": item.item_id,
        "scorer": "jev",
        "status": "scored",
        "labels": {rid: answer["value"] for rid, answer in answers.items()},
        "probabilities": {rid: answer.get("probabilities") for rid, answer in answers.items()},
        "call": call,
    }


def compare(jev_rows: list[dict], gpt_rows: list[dict], items_by_id: dict[str, ScoringItem], rubric_ids: list[str]) -> dict:
    gpt = {row["item_id"]: row for row in gpt_rows}
    jev = {row["item_id"]: row for row in jev_rows}
    counts = {"units": 0, "label_agreement": 0, "applicability_agreement": 0, "numeric_score_agreement_when_both_applicable": 0,
              "both_applicable": 0, "jev_only_applicable": 0, "gpt_only_applicable": 0}
    disagreements = []
    for item_id in sorted(set(jev) & set(gpt)):
        for rid in rubric_ids:
            jl = canonical_label(jev[item_id]["labels"].get(rid))
            gl = canonical_label(next(
                (s["score"] if s.get("applicable") else "NA"
                 for s in gpt[item_id].get("raw_scores", [])
                 if s.get("rubric_id") == rid),
                gpt[item_id].get("labels", {}).get(rid),
            ))
            counts["units"] += 1
            if jl == gl:
                counts["label_agreement"] += 1
            ja, ga = jl != "NA", gl != "NA"
            if ja == ga:
                counts["applicability_agreement"] += 1
            elif ja:
                counts["jev_only_applicable"] += 1
            else:
                counts["gpt_only_applicable"] += 1
            if ja and ga:
                counts["both_applicable"] += 1
                if int(jl) == int(gl):
                    counts["numeric_score_agreement_when_both_applicable"] += 1
            if jl != gl:
                disagreements.append({"item_id": item_id, "rubric_id": rid, "jev": jl, "gpt56_sol": gl})
    return {
        "counts": counts,
        "label_agreement_rate": counts["label_agreement"] / counts["units"] if counts["units"] else None,
        "applicability_agreement_rate": counts["applicability_agreement"] / counts["units"] if counts["units"] else None,
        "numeric_score_agreement_rate_when_both_applicable": (
            counts["numeric_score_agreement_when_both_applicable"] / counts["both_applicable"]
            if counts["both_applicable"] else None
        ),
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-glob", required=True)
    parser.add_argument("--rubric-set", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--include-initial-turn", action="store_true")
    parser.add_argument("--gpt-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--jev-model", default="typesafe/jev-1.13")
    parser.add_argument("--jev-api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectory_glob, args.max_trajectories)
    rubrics = load_rubrics(args.rubric_set)
    scoring_items = items(trajectories, rubrics, args.include_initial_turn)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    gpt_system = (ROOT / "prompts/rubric_scorer_visible_only.txt").read_text(encoding="utf-8")
    gpt_chat = CachedChat(
        build_chat_backend({
            "backend": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "model_id": args.gpt_model,
            "api_key_env": "OPENROUTER_API_KEY",
            "revision": "openrouter-unpinned",
        }),
        out / "gpt_cache.jsonl",
    )
    gpt_scorer = ChatRubricScorer("gpt56_sol", gpt_chat, system_prompt=gpt_system,
                                  prompt_version="rubric-scorer-visible-only-v1", params={"temperature": 0, "max_tokens": 2000})
    jev_client = JevClient(endpoint=args.jev_endpoint, model=args.jev_model, api_key_env=args.jev_api_key_env)
    jev_rows, gpt_rows = [], []
    for item in scoring_items:
        jev_rows.append(jev_score(jev_client, item))
        gpt_result = gpt_scorer.score(item)
        gpt_row = gpt_result.to_dict()
        # Keep the raw structured score entries for applicability/label comparison.
        if gpt_result.raw_response:
            parsed, _ = parse_json_object(gpt_result.raw_response)
            gpt_row["raw_scores"] = parsed.get("scores", []) if parsed else []
        gpt_rows.append(gpt_row)

    for name, rows in (("scores_jev.jsonl", jev_rows), ("scores_gpt56_sol.jsonl", gpt_rows)):
        (out / name).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    report = compare(jev_rows, gpt_rows, {item.item_id: item for item in scoring_items},
                     [rubric["rubric_id"] for rubric in rubrics])
    report.update({"trajectories": len(trajectories), "items": len(scoring_items), "rubrics": len(rubrics),
                   "gpt_model": args.gpt_model, "jev_model": args.jev_model})
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("trajectories", "items", "rubrics", "label_agreement_rate",
                                                    "applicability_agreement_rate", "numeric_score_agreement_rate_when_both_applicable")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
