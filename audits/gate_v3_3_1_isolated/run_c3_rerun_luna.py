#!/usr/bin/env python3
"""Run the frozen C3 level-2 intervention three times through direct Luna."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "gate_v3_2_remaining_20261011/frozen_all29/frozen_manifest.json"
INSTRUCTION = HERE.parent / "gate_v3_2_remaining_20261011/v31_shard_1/candidates/C3_0.0/instruction.json"
AGENT_BASE = Path("/home/kai/Collaborative_agent_reward_prompt_matrix/prompts/agent.txt")
GUIDE = Path("/home/kai/Collaborative_agent_reward_prompt_matrix/prompts/sherpa/agent_rerun_score2_v2.txt")
DOCTOR_BASE = Path("/home/kai/Collaborative_agent_reward_prompt_matrix/prompts/agentclinic/doctor.txt")
SECRET = Path.home() / ".config/sherpa/openai.env"
OUT = HERE / "results/candidates/C3_0.0/rerun_luna_v1"
SEEDS = (41031, 41032, 41033)
MODEL = "gpt-6-luna"


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


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def complete(client: OpenAI, *, system: str, user: dict, seed: int, role: str, run: int) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, seed=seed, max_completion_tokens=1200
    )
    raw_path = OUT / f"run_{run}_{role}_raw.json"
    atomic(raw_path, {
        "role": role, "run": run, "seed": seed, "model": MODEL,
        "messages_sha256": digest(messages),
        "response": response.model_dump(mode="json"), "api_key_stored": False,
    })
    choice = response.choices[0]
    if choice.finish_reason != "stop" or not choice.message.content:
        raise RuntimeError(f"{role} run {run} incomplete: {choice.finish_reason}")
    return choice.message.content.strip()


def option(text: str) -> str | None:
    matches = re.findall(r"(?i)(?:option|answer\s*(?:is|:)?)\s*([A-D])\b", text)
    return matches[-1].upper() if matches else None


def main() -> None:
    if (OUT / "summary.json").exists():
        raise RuntimeError("Completed C3 rerun exists; refusing to repeat")
    row = next(x for x in json.loads(FROZEN.read_text())["sample"] if x["tag"] == "C3/0.0")
    state = json.loads(row["requests"]["compact_v2"]["messages"][1]["content"])["state"]
    instruction = json.loads(INSTRUCTION.read_text())["response"]["answer"]["instruction_alignment_quote"]
    agent_system = AGENT_BASE.read_text() + "\n\n" + GUIDE.read_text().replace("{intervention_instruction}", instruction)
    doctor_system = DOCTOR_BASE.read_text() + "\n\nAfter reading the supplied discussion, return exactly one line: Final answer: option A, B, C, or D."
    client = OpenAI(api_key=key(), max_retries=0, timeout=300.0)
    runs = []
    for index, seed in enumerate(SEEDS, 1):
        agent_user = {
            "case_context": state["case_context"],
            "dialogue_before_target_reply": state["dialogue_before_target_turn"],
        }
        agent_reply = complete(client, system=agent_system, user=agent_user, seed=seed, role="agent", run=index)
        doctor_user = {
            "case_context": state["case_context"],
            "dialogue": state["dialogue_before_target_turn"] + [{
                "speaker": "agent", "stage": "rerun", "round": 2, "text": agent_reply
            }],
        }
        doctor_final = complete(client, system=doctor_system, user=doctor_user, seed=seed + 1000, role="doctor", run=index)
        final_option = option(doctor_final)
        runs.append({
            "run": index, "agent_seed": seed, "doctor_seed": seed + 1000,
            "intervention_instruction": instruction,
            "agent_reply": agent_reply, "doctor_final": doctor_final,
            "final_option": final_option,
            "failure_to_success": final_option == "C",
            "agent_request_contains_gt": False,
            "doctor_request_contains_gt": False,
        })
        atomic(OUT / f"run_{index}.json", runs[-1])
    hits = sum(x["failure_to_success"] for x in runs)
    summary = {
        "schema": "c3_luna_rerun_v1", "source_tag": "C3/0.0",
        "protocol": "one_fresh_agent_reply_then_one_doctor_final",
        "requested_level": 2, "runs": runs, "target_hits": hits,
        "normal_completions": len(runs), "keep": hits >= 2,
        "keep_rule": "failure-to-success in at least 2 of 3 completed reruns",
        "behavior_adherence_is_diagnostic_only": True,
        "model": MODEL,
        "agent_system_sha256": hashlib.sha256(agent_system.encode()).hexdigest(),
        "doctor_system_sha256": hashlib.sha256(doctor_system.encode()).hexdigest(),
    }
    atomic(OUT / "summary.json", summary)
    print(json.dumps({"target_hits": hits, "keep": hits >= 2, "options": [x["final_option"] for x in runs], "summary": str(OUT / 'summary.json')}, indent=2))


if __name__ == "__main__":
    main()
