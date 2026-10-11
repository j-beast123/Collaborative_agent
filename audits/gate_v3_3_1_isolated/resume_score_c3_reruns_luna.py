#!/usr/bin/env python3
"""Resume only incomplete C3 scoring rows; preserve failed raw attempts."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

import score_c3_reruns_luna as base


def score_one(client, system, state, rubric, run, attempt):
    payload = {
        "stage": "level", "applicability_already_fixed": True,
        "state": {"case_context": state["case_context"], "dialogue_before_target_reply": state["dialogue_before_target_turn"], "target_agent_reply": run["agent_reply"]},
        "rubric": rubric,
        "information_excluded": ["ground_truth", "source_outcome", "requested_level", "intervention_instruction", "doctor_final"],
    }
    response = client.chat.completions.create(
        model=base.MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        max_completion_tokens=2400,
        response_format={"type": "json_schema", "json_schema": {"name": "compact_behavior_score_v1", "strict": True, "schema": base.SCHEMA}},
    )
    suffix = f"_attempt_{attempt}" if attempt > 1 else ""
    raw_path = base.OUT / f"run_{run['run']}_raw{suffix}.json"
    base.atomic(raw_path, {"run": run["run"], "attempt": attempt, "model": base.MODEL, "response": response.model_dump(mode="json"), "api_key_stored": False})
    choice = response.choices[0]
    if choice.finish_reason != "stop" or not choice.message.content:
        raise RuntimeError(f"Run {run['run']} incomplete")
    parsed = json.loads(choice.message.content)
    Draft202012Validator(base.SCHEMA).validate(parsed)
    for quote in parsed["quotes"]:
        if quote not in run["agent_reply"]:
            raise RuntimeError(f"Run {run['run']} has nonliteral evidence")
    for check in parsed["checks"]:
        if any(index >= len(parsed["quotes"]) for index in check["quote_indices"]):
            raise RuntimeError(f"Run {run['run']} has invalid quote index")
    record = {"run": run["run"], "reply": run["agent_reply"], "score": parsed, "raw_response": str(raw_path), "attempt": attempt}
    base.atomic(base.OUT / f"run_{run['run']}.json", record)
    return record


def main():
    frozen = next(x for x in json.loads(base.FROZEN.read_text())["sample"] if x["tag"] == "C3/0.0")
    state = json.loads(frozen["requests"]["compact_v2"]["messages"][1]["content"])["state"]
    rubric = state["candidate_rubric"]
    runs = json.loads(base.RERUN.read_text())["runs"]
    system = base.BASE.read_text() + "\n\n" + base.SEMANTICS.read_text()
    client = OpenAI(api_key=base.key(), max_retries=0, timeout=300.0)
    scored = []
    for run in runs:
        complete = base.OUT / f"run_{run['run']}.json"
        if complete.exists():
            scored.append(json.loads(complete.read_text()))
            continue
        failed_raw = base.OUT / f"run_{run['run']}_raw.json"
        attempt = 2 if failed_raw.exists() else 1
        scored.append(score_one(client, system, state, rubric, run, attempt))
    scored.sort(key=lambda row: row["run"])
    summary = {
        "schema": "c3_rerun_independent_scoring_v1", "model": base.MODEL,
        "shared_applicability": True, "source_score_from_gate": 0,
        "rerun_levels": [x["score"]["level"] for x in scored], "runs": scored,
        "gt_outcome_requested_level_instruction_and_doctor_final_excluded": True,
        "technical_retry": {"run": 2, "reason": "nonliteral punctuation in evidence quote", "semantic_revote": False},
    }
    base.atomic(base.OUT / "summary.json", summary)
    print(json.dumps({"source": 0, "reruns": summary["rerun_levels"], "summary": str(base.OUT / 'summary.json')}, indent=2))


if __name__ == "__main__":
    main()
