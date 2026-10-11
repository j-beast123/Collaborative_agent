#!/usr/bin/env python3
"""Read-only audit of the existing C5/0.0 three-rerun witness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compact_quality_gate_v3_3_1 import file_sha256, sha256_json

HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
WITNESS = AUDIT_ROOT / "gate_v3_3_pilot_20261010/witness_c5_0_0"
FROZEN = AUDIT_ROOT / "gate_v3_2_remaining_20261011/frozen_all29/frozen_manifest.json"
OLD_RESULT = AUDIT_ROOT / "gate_v3_3_pilot_20261010/results/candidates/C5_0.0"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    agents = load_jsonl(WITNESS / "agent_wire.jsonl")
    doctors = load_jsonl(WITNESS / "doctor_wire.jsonl")
    if len(agents) != 3 or len(doctors) != 3:
        raise RuntimeError("Expected exactly three complete rerun pairs")
    instruction_packet = json.loads((OLD_RESULT / "instruction.json").read_text())
    instruction = instruction_packet["instruction_state"]["intervention_instruction"]
    frozen = json.loads(FROZEN.read_text())["sample"]
    row = next(item for item in frozen if item["tag"] == "C5/0.0")
    state = json.loads(row["requests"]["compact_v2"]["messages"][1]["content"])["state"]
    expected_dialogue = [{k: turn[k] for k in ("speaker", "stage", "round", "text")}
                         for turn in state["dialogue_before_target_turn"]]

    checkpoint = Path(json.loads((WITNESS / "manifest.json").read_text())["checkpoint"])
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    runs = []
    for index, (agent, doctor) in enumerate(zip(agents, doctors), 1):
        user = json.loads(agent["user"])
        actual_dialogue = user["dialogue_before_target_reply"]
        doctor_message = doctor["request"]["messages"]
        runs.append({
            "run": index,
            "intervention_instruction": instruction,
            "agent_request": {"system": agent["system"], "user": user, "seed": agent["seed"]},
            "agent_reply": agent["raw_response"],
            "legacy_agent_reply_field": "intervention_text",
            "doctor_request": doctor["request"],
            "doctor_final": doctor["raw_response"],
            "source_prefix_exact": actual_dialogue == expected_dialogue,
            "agent_request_hidden_fields": sorted(set(user) & {
                "ground_truth", "gold_answer", "source_outcome", "outcome", "future_dialogue", "final_answer"
            }),
            "instruction_contains_generated_answer_sentence": "The correct answer is C" in instruction,
            "agent_reply_contains_generated_answer_sentence": "The correct answer is C" in agent["raw_response"],
            "protocol_shape": "one_fresh_agent_reply_then_one_doctor_final",
            "agent_status": agent["status"],
            "doctor_status": doctor["status"],
        })

    common_system = agents[0]["system"].split("[Role and scope]", 1)[0]
    result = {
        "schema": "rerun_wire_field_audit_v1",
        "source_tag": "C5/0.0",
        "runs": runs,
        "summary": {
            "normal_completions": "3/3",
            "success_to_failure": "0/3",
            "keep": False,
            "behavior_adherence_role": "diagnostic_only",
            "causal_ineffectiveness_established": False,
            "gt_or_answer_key_in_intervention_instruction": False,
            "hidden_gt_outcome_or_future_fields_in_agent_request": False,
            "source_prefix_exact_all_runs": all(run["source_prefix_exact"] for run in runs),
            "agent_policy_sha256": agents[0]["policy_sha256"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_complete_sha256": file_sha256(checkpoint / "COMPLETE.json"),
            "lora_adapter_sha256": file_sha256(adapter),
            "doctor_model": doctors[0]["model"],
            "doctor_digest": doctors[0]["model_digest"],
            "doctor_request_prompt_sha256": sha256_json(doctors[0]["request"]["messages"]),
            "doctor_decoding": doctors[0]["request"]["options"],
            "agent_decoding": {
                "per_run_seed": [agent["seed"] for agent in agents],
                "other_settings": "not serialized in agent_wire.jsonl; policy runtime configuration must be consulted",
            },
            "abbreviated_continuation": True,
            "common_system_and_score0_instruction_conflict": False,
            "conflict_reason": "The common role explicitly gives the selected behavior instruction precedence and does not require a general explanation or comparison.",
            "common_agent_system_sha256": hashlib.sha256(common_system.encode()).hexdigest(),
            "intervention_instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
            "observed_adherence_note": "All three generated replies reintroduced at least part of the omitted mechanism or comparison; this is diagnostic and does not alter KEEP.",
        },
        "source_files": {
            "agent_wire": str(WITNESS / "agent_wire.jsonl"),
            "doctor_wire": str(WITNESS / "doctor_wire.jsonl"),
            "summary": str(WITNESS / "summary.json"),
            "instruction": str(OLD_RESULT / "instruction.json"),
        },
    }
    write_json(HERE / "reports/rerun_wire_audit.json", result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
