#!/usr/bin/env python3
"""Independently score the three C3 rerun replies with the frozen public rubric."""
from __future__ import annotations

import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "gate_v3_2_remaining_20261011/frozen_all29/frozen_manifest.json"
RERUN = HERE / "results/candidates/C3_0.0/rerun_luna_v1/summary.json"
OUT = HERE / "results/candidates/C3_0.0/rerun_luna_v1/scoring_v1"
SECRET = Path.home() / ".config/sherpa/openai.env"
BASE = Path("/home/kai/Collaborative_agent_reward_prompt_matrix/prompts/sherpa/luna_scorer_v2.txt")
SEMANTICS = Path("/home/kai/audits/scorer_app_bank_20261010/frozen/active_contract/prompts/sherpa/compact_behavior_semantics_v2.txt")
MODEL = "gpt-6-luna"

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["level", "quotes", "checks", "gap", "defect"],
    "properties": {
        "level": {"enum": ["0", "1", "2", "UNSCORABLE"]},
        "quotes": {"type": "array", "items": {"type": "string"}},
        "checks": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["definition_ref", "met", "quote_indices", "explanation"],
            "properties": {
                "definition_ref": {"enum": ["minimum_performance", "completion_condition", "score_anchors.0", "score_anchors.1", "score_anchors.2"]},
                "met": {"type": ["boolean", "null"]},
                "quote_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                "explanation": {"type": "string"},
            },
        }},
        "gap": {"type": "string"}, "defect": {"type": "string"},
    },
}


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def key() -> str:
    for line in SECRET.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENAI_API_KEY unavailable")


def main() -> None:
    if (OUT / "summary.json").exists():
        raise RuntimeError("Scoring already complete; refusing repeated semantic calls")
    frozen = next(x for x in json.loads(FROZEN.read_text())["sample"] if x["tag"] == "C3/0.0")
    state = json.loads(frozen["requests"]["compact_v2"]["messages"][1]["content"])["state"]
    rubric = state["candidate_rubric"]
    runs = json.loads(RERUN.read_text())["runs"]
    system = BASE.read_text() + "\n\n" + SEMANTICS.read_text()
    client = OpenAI(api_key=key(), max_retries=0, timeout=300.0)
    scored = []
    for run in runs:
        payload = {
            "stage": "level", "applicability_already_fixed": True,
            "state": {
                "case_context": state["case_context"],
                "dialogue_before_target_reply": state["dialogue_before_target_turn"],
                "target_agent_reply": run["agent_reply"],
            },
            "rubric": rubric,
            "information_excluded": ["ground_truth", "source_outcome", "requested_level", "intervention_instruction", "doctor_final"],
        }
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            max_completion_tokens=2400,
            response_format={"type": "json_schema", "json_schema": {"name": "compact_behavior_score_v1", "strict": True, "schema": SCHEMA}},
        )
        raw_path = OUT / f"run_{run['run']}_raw.json"
        atomic(raw_path, {"run": run["run"], "model": MODEL, "response": response.model_dump(mode="json"), "api_key_stored": False})
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise RuntimeError(f"Run {run['run']} incomplete")
        parsed = json.loads(choice.message.content)
        Draft202012Validator(SCHEMA).validate(parsed)
        for quote in parsed["quotes"]:
            if quote not in run["agent_reply"]:
                raise RuntimeError(f"Run {run['run']} has nonliteral evidence")
        for check in parsed["checks"]:
            if any(index >= len(parsed["quotes"]) for index in check["quote_indices"]):
                raise RuntimeError(f"Run {run['run']} has invalid quote index")
        record = {"run": run["run"], "reply": run["agent_reply"], "score": parsed, "raw_response": str(raw_path)}
        atomic(OUT / f"run_{run['run']}.json", record)
        scored.append(record)
    summary = {
        "schema": "c3_rerun_independent_scoring_v1", "model": MODEL,
        "shared_applicability": True, "source_score_from_gate": 0,
        "rerun_levels": [x["score"]["level"] for x in scored],
        "runs": scored,
        "gt_outcome_requested_level_instruction_and_doctor_final_excluded": True,
    }
    atomic(OUT / "summary.json", summary)
    print(json.dumps({"source": 0, "reruns": summary["rerun_levels"], "summary": str(OUT / 'summary.json')}, indent=2))


if __name__ == "__main__":
    main()
