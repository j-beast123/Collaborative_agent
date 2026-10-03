"""Consolidate V2 validation passes with the repository's original prompts."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.judges.llm import CachedChat, build_chat_backend
from src.judges.rubric_consolidation import (
    assign_categories,
    build_final_rubrics,
    reconcile_categories,
)
from src.judges.rubric_retention import CurrentOnly


MODEL = "openai/gpt-5.6-sol"
PARAMS = {"temperature": 0, "max_tokens": 3000}
MANAGER = {
    "backend": "openai_compatible",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "model_id": MODEL,
    "revision": "openrouter-unpinned",
}


def load_passes() -> tuple[list[dict], dict[str, dict], list[str]]:
    validation_paths = glob.glob(
        str(ROOT / "outputs/rubric_validation/gemma12b_qwen3_8b_300_gpt56_sol_original_rubric_retry7/gpu_*/validations.jsonl")
    )
    validations = {}
    for path in validation_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("validation_status") == "pass":
                        validations[row["rubric_id"]] = row

    candidates = {}
    source = ROOT / "outputs/rubric_candidates/rubric_candidates_300_gpt56_sol.jsonl"
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("rubric_id") in validations:
                    candidates[row["rubric_id"]] = row
    records = [candidates[rubric_id] for rubric_id in sorted(validations)]
    return records, validations, validation_paths


def main() -> None:
    records, validations, validation_paths = load_passes()
    out = ROOT / "outputs/rubric_candidates/v2_pass_consolidation_original_gpt56_sol_11.json"
    cache = ROOT / "outputs/rubric_candidates/.cache_v2_pass_consolidation_original_gpt56_sol_11.jsonl"
    chat = CachedChat(build_chat_backend(MANAGER), cache)

    assign_prompt = (ROOT / "prompts/rubric_category_assign.txt").read_text(encoding="utf-8")
    reconcile_prompt = (ROOT / "prompts/rubric_category_reconcile.txt").read_text(encoding="utf-8")
    construct_prompt = (ROOT / "prompts/rubric_final_construct.txt").read_text(encoding="utf-8")

    categories, assignment_log = assign_categories(
        records,
        chat,
        system_prompt=assign_prompt,
        prompt_version="rubric-category-assign-v1",
        params=PARAMS,
        iteration=0,
    )
    categories, reconciliation = reconcile_categories(
        categories,
        records,
        chat,
        system_prompt=reconcile_prompt,
        prompt_version="rubric-category-reconcile-v1",
        params=PARAMS,
        iteration=0,
    )
    rubric_set = build_final_rubrics(
        categories,
        records,
        chat,
        system_prompt=construct_prompt,
        prompt_version="rubric-final-construct-v1",
        params=PARAMS,
        iteration=0,
        max_active_categories=None,
        min_failure_count=None,
        retention=CurrentOnly(),
        importance="uniform",
        validations=validations,
        canonical=True,
    )

    payload = {
        "source": "V2 validation pass records",
        "model": MODEL,
        "prompt_files": [
            "prompts/rubric_category_assign.txt",
            "prompts/rubric_category_reconcile.txt",
            "prompts/rubric_final_construct.txt",
        ],
        "source_validation_paths": validation_paths,
        "input_pass_count": len(records),
        "input_rubric_ids": [r["rubric_id"] for r in records],
        "assignment_log": assignment_log,
        "reconciliation": reconciliation,
        "category_count": len([c for c in categories if c.get("status") != "merged"]),
        "rubric_set": rubric_set,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "input_pass_count": len(records),
        "category_count": payload["category_count"],
        "rubric_count": len(rubric_set["rubrics"]),
        "merged_by_reconciliation": len(reconciliation.get("applied", [])),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
