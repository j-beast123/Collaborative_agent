"""Compare independent Jev and GPT-5.6-sol rubric scores on the same trajectories."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judges.jev import JevClient
from src.judges.llm import CachedChat, build_chat_backend, parse_json_object
from src.judges.rubric_scoring import (
    TWO_STAGE_JEV_PROTOCOL,
    ChatRubricScorer,
    ScoringItem,
    two_stage_jev_score,
)


def read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_trajectories(pattern: str, limit: int | None) -> tuple[list[dict], list[str]]:
    rows, seen = [], set()
    names = sorted(glob.glob(str(ROOT / pattern)))
    for name in names:
        for row in read_jsonl(Path(name)):
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id or trajectory_id in seen or row.get("trajectory_status") != "valid":
                continue
            seen.add(trajectory_id)
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows, names
    return rows, names


def load_rubrics(path: str) -> tuple[list[dict], dict]:
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
                level: behavior
                for level, behavior in score_behaviors.items()
            },
        })
    if not rubrics:
        raise ValueError(f"no rubrics in {path}")
    if len({rubric["rubric_id"] for rubric in rubrics}) != len(rubrics):
        raise ValueError(f"duplicate rubric_id values in {path}")
    return rubrics, payload


def display_path(path: str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


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
    scored = two_stage_jev_score(
        client,
        history_state=item.pre_target_payload(),
        target_state=item.target_payload(),
        rubrics=list(item.rubrics),
        purpose="rubric_compare_jev",
    )
    calls = scored.pop("calls")
    return {
        "item_id": item.item_id,
        "scorer": "jev",
        "status": "scored",
        **scored,
        "call": calls,
    }


def compare(jev_rows: list[dict], gpt_rows: list[dict], rubric_ids: list[str]) -> dict:
    gpt = {row["item_id"]: row for row in gpt_rows}
    jev = {row["item_id"]: row for row in jev_rows}
    counts = {"units": 0, "label_agreement": 0, "applicability_agreement": 0, "numeric_score_agreement_when_both_applicable": 0,
              "both_applicable": 0, "jev_only_applicable": 0, "gpt_only_applicable": 0,
              "skipped_invalid_units": 0}
    disagreements = []
    for item_id in sorted(set(jev) & set(gpt)):
        if jev[item_id].get("status") != "scored" or gpt[item_id].get("status") != "scored":
            counts["skipped_invalid_units"] += len(rubric_ids)
            continue
        for rid in rubric_ids:
            jl = canonical_label(jev[item_id]["labels"].get(rid))
            gl = canonical_label(gpt[item_id].get("labels", {}).get(rid))
            if jl not in {"NA", 0, 1, 2} or gl not in {"NA", 0, 1, 2}:
                counts["skipped_invalid_units"] += 1
                continue
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trajectories, trajectory_files = load_trajectories(args.trajectory_glob, args.max_trajectories)
    rubrics, rubric_payload = load_rubrics(args.rubric_set)
    scoring_items = items(trajectories, rubrics, args.include_initial_turn)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out}; use --overwrite or a new directory")
    if args.overwrite:
        for name in ("scores_jev.jsonl", "scores_gpt56_sol.jsonl", "comparison.json", "run_metadata.json"):
            (out / name).unlink(missing_ok=True)

    rubric_path = ROOT / args.rubric_set
    rubric_sha256 = hashlib.sha256(rubric_path.read_bytes()).hexdigest()
    rubric_content_sha256 = hashlib.sha256(json.dumps(
        rubric_payload.get("rubrics") or [], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    rubric_set_version = rubric_payload.get("rubric_set_version")
    if rubric_set_version and rubric_set_version != rubric_content_sha256[:len(str(rubric_set_version))]:
        raise ValueError(
            f"rubric_set_version does not match rubric content: declared={rubric_set_version} "
            f"computed={rubric_content_sha256[:len(str(rubric_set_version))]}"
        )
    trajectory_versions = {
        str(row["trajectory_id"]): (
            row.get("rubric_set_version") or (row.get("generation") or {}).get("rubric_set_version")
        )
        for row in trajectories
    }

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
        metadata = {
            "rubric_set": args.rubric_set,
            "rubric_set_version": rubric_set_version,
            "rubric_set_sha256": rubric_sha256,
            "rubric_set_content_sha256": rubric_content_sha256,
            "trajectory_rubric_set_version": trajectory_versions.get(item.trajectory_id),
        }
        jev_rows.append({
            **jev_score(jev_client, item), **metadata,
            "scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        })
        gpt_result = gpt_scorer.score(item)
        gpt_row = {**gpt_result.to_dict(), **metadata, "scoring_protocol": "single_call_gpt_visible_only_v1"}
        # Keep the raw structured score entries for applicability/label comparison.
        if gpt_result.raw_response:
            parsed, _ = parse_json_object(gpt_result.raw_response)
            gpt_row["raw_scores"] = parsed.get("scores", []) if parsed else []
        gpt_rows.append(gpt_row)

    for name, rows in (("scores_jev.jsonl", jev_rows), ("scores_gpt56_sol.jsonl", gpt_rows)):
        (out / name).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    report = compare(jev_rows, gpt_rows, [rubric["rubric_id"] for rubric in rubrics])
    report.update({"trajectories": len(trajectories), "items": len(scoring_items), "rubrics": len(rubrics),
                   "gpt_model": args.gpt_model, "jev_model": args.jev_model,
                   "rubric_set": args.rubric_set, "rubric_set_version": rubric_set_version,
                   "rubric_set_sha256": rubric_sha256,
                   "rubric_set_content_sha256": rubric_content_sha256,
                   "trajectory_rubric_set_versions": sorted({v for v in trajectory_versions.values() if v})})
    metadata = {
        "trajectory_glob": args.trajectory_glob,
        "trajectory_files": [display_path(name) for name in trajectory_files],
        "max_trajectories": args.max_trajectories,
        "include_initial_turn": args.include_initial_turn,
        "rubric_set": args.rubric_set,
        "rubric_set_version": rubric_set_version,
        "rubric_set_sha256": rubric_sha256,
        "rubric_set_content_sha256": rubric_content_sha256,
        "trajectory_rubric_set_versions": report["trajectory_rubric_set_versions"],
        "jev_scoring_protocol": TWO_STAGE_JEV_PROTOCOL,
        "gpt_scoring_protocol": "single_call_gpt_visible_only_v1",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("trajectories", "items", "rubrics", "label_agreement_rate",
                                                    "applicability_agreement_rate", "numeric_score_agreement_rate_when_both_applicable")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
