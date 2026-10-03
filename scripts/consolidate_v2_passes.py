"""Consolidate the validated V2 pass rubrics using the behavior only."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from src.judges.llm import ChatRequest, openrouter_chat, parse_json_object


ROOT = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """You consolidate validated interaction-behavior rubric candidates for a clinical collaborative agent.
Use only the behavioral failure and ideal behavior. Ignore disease names, answer options, final correctness, user-side context, and agent belief/state.
Group candidates only when one reusable behavioral principle can score all of them without changing its trigger or required action.
Do not merge merely because they use words like criterion, evidence, comparison, or alternative.
Return exactly JSON with this schema:
{
  "rubrics": [
    {
      "name": "short reusable behavior name",
      "source_rubric_ids": ["..."],
      "scope": "the shared behavioral situation and required action",
      "score_0": "clear violation behavior",
      "score_1": "partial behavior",
      "score_2": "ideal behavior"
    }
  ],
  "unmerged_reason": "brief explanation of important distinctions"
}
Score descriptions must be observable response behavior and must not require changing the final answer."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/rubric_candidates/v2_pass_consolidation_gpt56_sol.json",
    )
    args = parser.parse_args()

    validation_paths = glob.glob(
        str(ROOT / "outputs/rubric_validation/gemma12b_qwen3_8b_300_gpt56_sol/gpu_*_v2/validations.jsonl")
    )
    pass_rows = {}
    for path in validation_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("validation_status") == "pass":
                    pass_rows[row["rubric_id"]] = row

    candidate_by_id = {}
    source = ROOT / "outputs/rubric_candidates/rubric_candidates_300_gpt56_sol.jsonl"
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("rubric_id") in pass_rows:
                    candidate_by_id[row["rubric_id"]] = row

    items = []
    for rubric_id in sorted(pass_rows):
        analysis = candidate_by_id[rubric_id].get("analysis") or {}
        items.append({
            "rubric_id": rubric_id,
            "rubric_name": analysis.get("rubric_name"),
            "interaction_failure": analysis.get("interaction_failure"),
            "ideal_response_behavior": analysis.get("ideal_response_behavior"),
        })

    user_prompt = json.dumps({"validated_pass_candidates": items}, ensure_ascii=False, indent=2)
    output = ROOT / args.output
    cache = ROOT / "outputs/rubric_candidates/.cache_v2_pass_consolidation_gpt56_sol.jsonl"
    chat = openrouter_chat("openai/gpt-5.6-sol", cache)
    result = chat.complete(ChatRequest(
        SYSTEM_PROMPT,
        user_prompt,
        "v2-pass-consolidation-v1",
        {"temperature": 0, "max_tokens": 4000},
    ))
    parsed, parse_status = parse_json_object(result.text if result.status == "ok" else None)
    payload = {
        "source": "V2 validation pass records",
        "source_validation_paths": validation_paths,
        "input_count": len(items),
        "input_rubric_ids": [item["rubric_id"] for item in items],
        "model": "openai/gpt-5.6-sol",
        "prompt_version": "v2-pass-consolidation-v1",
        "call": result.to_dict(),
        "parse_status": parse_status,
        "consolidation": parsed,
        "raw_response": result.text,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "input_count": len(items),
        "status": result.status,
        "parse_status": parse_status,
        "rubric_count": len((parsed or {}).get("rubrics", [])),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
