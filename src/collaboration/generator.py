"""Generate doctor-agent clinical collaboration trajectories."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.belief import (
    build_agent_response_prompt,
    build_initial_agent_response_prompt,
    build_plain_agent_response_prompt,
    build_local_judgment_prompt,
    map_judgment_to_roles,
    parse_local_judgment,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/interim/medmcqa/medicine_sample_500_label_corrected.jsonl",
    )
    p.add_argument(
        "--output", default="outputs/collaboration/collaboration_pilot_10.jsonl"
    )
    p.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    p.add_argument("--doctor-model", default="google/gemma-3-4b-it")
    p.add_argument("--reconstructor-model", default="google/gemini-2.5-flash-lite")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Maximum agent responses/collaboration rounds",
    )
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--claim-order", choices=("doctor_first", "agent_first"), default="doctor_first"
    )
    p.add_argument("--prompt-version", default="grounded-collaboration-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip trajectory IDs already present in the output.",
    )
    p.add_argument(
        "--disable-belief",
        action="store_true",
        help="Run the raw-history agent baseline without reconstruction or Q1/Q2.",
    )
    return p.parse_args()


def load(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def case_text(case: dict) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in case["options"].items())
    scenario = case.get("scenario", "") or "(No separate patient vignette is provided.)"
    return f"Case/vignette:\n{scenario}\n\nQuestion:\n{case['question']}\n\nAnswer choices:\n{options}"


def dialogue_text(dialogue: list[dict]) -> str:
    return (
        "\n\n".join(f"{x['speaker'].title()}: {x['text']}" for x in dialogue)
        or "(No previous dialogue.)"
    )


def doctor_call(key: str, model: str, system: str, user: str, max_tokens: int) -> str:
    key = key.strip()
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 6:
                raise
            body = error.read().decode("utf-8", errors="replace")
            retry_after = error.headers.get("Retry-After")
            print(f"OpenRouter HTTP {error.code}", flush=True)
            print(f"Body: {body[:1000]}", flush=True)
            print(f"Retry-After: {retry_after or '(not provided)'}", flush=True)
            delay = float(retry_after) if retry_after else min(60, 2**attempt)
            print(f"OpenRouter HTTP {error.code}; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)


RECONSTRUCTION_FIELDS = {
    "focal_claim",
    "clinician_current_position",
    "agent_current_position",
    "clinician_claims_so_far",
    "clinician_evidence_so_far",
    "agent_claims_so_far",
    "agent_evidence_so_far",
    "shared_evidence",
    "resolved_points",
    "unresolved_points",
    "unsupported_case_details",
}


def parse_reconstruction(raw: str) -> dict:
    try:
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1]).strip()
        value = json.loads(stripped)
    except (json.JSONDecodeError, IndexError):
        return {"raw_response": raw, "parse_status": "invalid"}
    if not isinstance(value, dict) or set(value) != RECONSTRUCTION_FIELDS:
        return {"raw_response": raw, "parse_status": "invalid"}
    for field in RECONSTRUCTION_FIELDS - {
        "focal_claim",
        "clinician_current_position",
        "agent_current_position",
    }:
        if not isinstance(value[field], list) or not all(
            isinstance(item, str) for item in value[field]
        ):
            return {
                "raw_response": raw,
                "parse_status": "invalid",
                "invalid_field": field,
            }
        value[field] = value[field][:5]
    for field in (
        "focal_claim",
        "clinician_current_position",
        "agent_current_position",
    ):
        if not isinstance(value[field], str):
            return {
                "raw_response": raw,
                "parse_status": "invalid",
                "invalid_field": field,
            }
        value[field] = value[field][:1000]
    value.update({"raw_response": raw, "parse_status": "valid"})
    return value


def reconstruct_call(
    key: str,
    model: str,
    case: dict,
    dialogue: list[dict],
    max_tokens: int,
    retries: int = 2,
) -> dict:
    system = (
        "Reconstruct the complete raw dialogue into the current interaction state. "
        "Preserve relevant earlier claims and evidence, resolve references, distinguish "
        "claims from evidence, and do not decide who is correct. Mark details absent "
        "from the source problem as unsupported_case_details. A participant's assertion "
        "is not evidence merely because it was stated. Case-specific evidence must be "
        "traceable to the source scenario, question, or options. Do not use ground truth. "
        "Return strict JSON with exactly the requested cumulative schema. Deduplicate "
        "semantically equivalent claims and evidence. Include at most five concise "
        "items per list and preserve only information relevant to the current focal "
        "disagreement. The focal_claim must identify the active disputed proposition "
        "or decision boundary, not merely repeat the source question unless the whole "
        "question is the active disagreement. Evaluate whether each fact directly "
        "supports the exact claim; do not treat a general association, comorbidity, "
        "or risk factor as evidence for a specific longitudinal, causal, or incidence "
        "claim. Eliminating another option does not automatically support the remaining one."
    )
    user = (
        f"Clinical question:\n{case['question']}\n\n"
        f"Answer choices:\n{json.dumps(case['options'], ensure_ascii=False)}\n\n"
        f"Scenario:\n{case.get('scenario', '') or 'No separate patient vignette is provided.'}\n\n"
        f"Complete raw dialogue through the latest clinician message:\n{dialogue_text(dialogue)}\n\n"
        "Required JSON schema:\n"
        + json.dumps(
            {
                field: []
                if field.endswith(
                    (
                        "claims_so_far",
                        "evidence_so_far",
                        "evidence",
                        "points",
                        "details",
                    )
                )
                else ""
                for field in RECONSTRUCTION_FIELDS
            },
            indent=2,
        )
    )
    for attempt in range(retries + 1):
        raw = doctor_call(key, model, system, user, max_tokens)
        parsed = parse_reconstruction(raw)
        if parsed["parse_status"] == "valid":
            return parsed
        print(
            f"reconstruction parse failure attempt={attempt + 1}/{retries + 1}",
            flush=True,
        )
    return parsed


def agent_call(model, tokenizer, system: str, user: str, max_tokens: int) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = {
        k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt").items()
    }
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()


def option(text: str, *, first: bool = False) -> str:
    final_pattern = (
        r"final\s+answer\s*[:\-]?\s*(?:is\s*)?(?:option\s+)?(?:\*\*)?([A-D])\b"
    )
    final_hits = list(re.finditer(final_pattern, text, re.I))
    if final_hits:
        return final_hits[0].group(1).upper()
    patterns = (
        r"(?:answer|option|choice)\s*[:\-]?\s*(?:is\s*)?(?:\*\*)?([A-D])\b",
        r"(?:^|\n)\s*(?:\*\*)?([A-D])(?:\*\*)?\s*[.):-]",
        r"(?:leaning|prefer|favor|support|choose|select)\s+(?:toward\s+|for\s+|option\s+)?([A-D])\b",
    )
    matches = [
        match for pattern in patterns for match in re.finditer(pattern, text, re.I)
    ]
    if not matches:
        return ""
    matches.sort(key=lambda match: match.start())
    return matches[0 if first else -1].group(1).upper()


def initial_option(text: str) -> str:
    """Extract the stated initial choice, avoiding option-list enumeration."""
    cue_patterns = (
        r"(?:initial thought|initial impression|initial choice|leaning|favor(?:s|ed)?|most likely)"
        r"[^.\n]{0,100}?(?:option\s+)?([A-D])\b",
        r"(?:option\s+)?([A-D])\b[^.\n]{0,40}(?:most likely|strongest contender|best choice)",
    )
    hits = [hit for pattern in cue_patterns for hit in re.findall(pattern, text, re.I)]
    if hits:
        return hits[-1].upper()
    # If the response has an explicit question boundary, use the last option
    # mentioned before the question rather than the first enumerated option.
    prefix = re.split(r"\?", text, maxsplit=1)[0]
    return option(prefix)


def signals_conclusion(text: str) -> bool:
    """Conservative natural-language fallback for early stopping."""
    patterns = (
        r"ready to conclude",
        r"comfortable concluding",
        r"ready to finalize",
        r"can conclude",
        r"sufficiently established",
        r"resolved the main uncertainty",
        r"satisfied with (?:this|the) (?:answer|conclusion)",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def main() -> None:
    a = args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    root = Path(__file__).resolve().parents[1]
    all_cases = load(root / a.input)[: a.max_cases]
    if not 0 <= a.shard_index < a.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    cases = all_cases[a.shard_index :: a.num_shards]
    doctor_system = (root / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    agent_system = (root / "prompts/agent.txt").read_text(encoding="utf-8")
    judgment_system = (root / "prompts/local_epistemic_judgment.txt").read_text(
        encoding="utf-8"
    )
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    out = root / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    failure_out = out.with_name(out.stem + "_failures.jsonl")
    existing_ids = set()
    if a.resume and out.exists():
        existing_ids = {
            json.loads(line).get("trajectory_id")
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    mode = "a" if a.resume else "w"
    with out.open(mode, encoding="utf-8") as handle, failure_out.open(
        "a" if a.resume else "w", encoding="utf-8"
    ) as failure_handle:
        for n, case in enumerate(cases, 1):
            condition_name = "baseline" if a.disable_belief else "belief"
            trajectory_id = f"pilot_{case['case_id']}_seed_{a.seed}_{condition_name}"
            if trajectory_id in existing_ids:
                print(f"skip existing {trajectory_id}", flush=True)
                continue
            dialogue = []
            doctor_initial_option = ""
            agent_initial_option = ""
            termination_reason = "max_agent_turns"
            for turn in range(1, a.rounds + 1):
                final_turn_instruction = (
                    "\nThis is the final collaboration turn. Give your best-supported final answer explicitly."
                    if turn == a.rounds
                    else ""
                )
                prompt = f"{case_text(case)}\n\nGrounding constraint: Use only the question, answer choices, dialogue, and established domain knowledge. If no patient vignette or event description is provided, do not invent one or add symptoms, injuries, tests, treatments, or events.\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nRespond naturally as the physician. This is discussion turn {turn} of {a.rounds}.{final_turn_instruction}"
                doctor = doctor_call(
                    key, a.doctor_model, doctor_system, prompt, a.max_tokens
                )
                if turn == 1:
                    doctor_initial_option = initial_option(doctor)
                dialogue.append({"speaker": "doctor", "text": doctor, "round": turn})
                if turn == 1:
                    agent_prompt = build_initial_agent_response_prompt(
                        scenario=case.get("scenario", ""),
                        question=case["question"],
                        options=case["options"],
                        clinician_message=doctor,
                    )
                    reconstruction = None
                    judgment = None
                    judgment_status = "not_applicable_turn_1"
                elif a.disable_belief:
                    reconstruction = None
                    judgment = None
                    judgment_status = "not_applicable_belief_disabled"
                    agent_prompt = build_plain_agent_response_prompt(
                        scenario=case.get("scenario", ""),
                        question=case["question"],
                        options=case["options"],
                        dialogue_history=dialogue_text(dialogue),
                        current_round=turn,
                        max_rounds=a.rounds,
                        is_final_round=turn == a.rounds,
                    )
                else:
                    reconstruction = reconstruct_call(
                        key, a.reconstructor_model, case, dialogue, 1024
                    )
                    if reconstruction.get("parse_status") != "valid":
                        termination_reason = "reconstruction_failure"
                        print(f"{case['case_id']} reconstruction_failure", flush=True)
                        break
                    clean_reconstruction = {
                        k: reconstruction[k] for k in RECONSTRUCTION_FIELDS
                    }
                    reconstructed_context = json.dumps(
                        clean_reconstruction, ensure_ascii=False, indent=2
                    )
                    judgment_prompt = build_local_judgment_prompt(
                        question=case["question"],
                        options=case["options"],
                        reconstruction=clean_reconstruction,
                        scenario=case.get("scenario", ""),
                        claim_order=a.claim_order,
                    )
                    judgment_raw = agent_call(
                        model, tokenizer, judgment_system, judgment_prompt, 384
                    )
                    judgment = parse_local_judgment(judgment_raw)
                    if judgment.get("parse_status") != "valid":
                        termination_reason = "judgment_failure"
                        print(f"{case['case_id']} judgment_failure", flush=True)
                        break
                    judgment_status = "valid"
                    clean_judgment = map_judgment_to_roles(judgment, a.claim_order)
                    clean_judgment = {
                        name: {
                            key: value for key, value in clean_judgment[name].items()
                        }
                        for name in ("q1", "q2")
                    }
                    agent_prompt = build_agent_response_prompt(
                        question=case["question"],
                        options=case["options"],
                        reconstructed_context=reconstructed_context,
                        judgment=clean_judgment,
                        dialogue_history=dialogue_text(dialogue),
                        scenario=case.get("scenario", ""),
                        current_round=turn,
                        max_rounds=a.rounds,
                        is_final_round=turn == a.rounds,
                    )
                response = agent_call(
                    model, tokenizer, agent_system, agent_prompt, a.max_tokens
                )
                if turn == 1:
                    agent_initial_option = initial_option(response)
                dialogue.append(
                    {
                        "speaker": "agent",
                        "text": response,
                        "round": turn,
                        "reconstruction": reconstruction,
                        "local_judgment": judgment,
                        "local_judgment_status": judgment_status,
                    }
                )
                if re.search(r"final answer\s*:\s*(?:option\s*)?[A-D]", response, re.I):
                    termination_reason = "agent_final_answer"
                    break
                if turn >= 2:
                    doctor_choice = option(doctor)
                    agent_choice = option(response)
                    if (
                        doctor_choice
                        and agent_choice
                        and doctor_choice == agent_choice
                        and signals_conclusion(doctor)
                        and signals_conclusion(response)
                    ):
                        termination_reason = "mutual_agreement"
                        break
                    doctor_ready = signals_conclusion(doctor)
                    agent_ready = signals_conclusion(response)
                    if doctor_ready and agent_ready:
                        termination_reason = "mutual_conclusion"
                        break
            if termination_reason in {"reconstruction_failure", "judgment_failure"}:
                failure_record = {
                    "case_id": case["case_id"],
                    "trajectory_id": trajectory_id,
                    "trajectory_status": "invalid",
                    "eligible_for_training": False,
                    "termination_reason": termination_reason,
                    "failure": {
                        "stage": "reconstruction"
                        if termination_reason == "reconstruction_failure"
                        else "judgment",
                        "round": len(
                            [item for item in dialogue if item["speaker"] == "doctor"]
                        ),
                        "parse_status": (
                            reconstruction
                            if termination_reason == "reconstruction_failure"
                            else judgment or {}
                        ).get("parse_status", "not_run"),
                        "invalid_field": (
                            reconstruction
                            if termination_reason == "reconstruction_failure"
                            else judgment or {}
                        ).get("invalid_field", ""),
                        "attempts": 3
                        if termination_reason == "reconstruction_failure"
                        else 1,
                        "raw_response": (
                            reconstruction
                            if termination_reason == "reconstruction_failure"
                            else judgment or {}
                        ).get("raw_response", ""),
                    },
                    "turns": dialogue,
                }
                failure_handle.write(
                    json.dumps(failure_record, ensure_ascii=False) + "\n"
                )
                failure_handle.flush()
                continue
            final_prompt = f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nThis is the final decision. State the option best supported by the complete discussion. End with exactly: Final answer: option <A-D> - <answer text>"
            final = doctor_call(
                key, a.doctor_model, doctor_system, final_prompt, a.max_tokens
            )
            actual_rounds = sum(1 for item in dialogue if item["speaker"] == "agent")
            dialogue.append(
                {
                    "speaker": "doctor",
                    "text": final,
                    "round": actual_rounds + 1,
                    "final": True,
                }
            )
            final_option = option(final)
            if not final_option:
                failure_record = {
                    "case_id": case["case_id"],
                    "trajectory_id": trajectory_id,
                    "trajectory_status": "invalid",
                    "eligible_for_training": False,
                    "termination_reason": "final_option_parse_failure",
                    "failure": {
                        "stage": "final_option",
                        "parse_status": "invalid",
                        "raw_response": final,
                    },
                    "turns": dialogue,
                }
                failure_handle.write(
                    json.dumps(failure_record, ensure_ascii=False) + "\n"
                )
                failure_handle.flush()
                continue
            record = {
                "case_id": case["case_id"],
                "trajectory_id": trajectory_id,
                "trajectory_status": "valid",
                "eligible_for_training": True,
                "belief_enabled": not a.disable_belief,
                "question": case["question"],
                "scenario": case.get("scenario", ""),
                "options": case["options"],
                "ground_truth": case["ground_truth"],
                "final_option": final_option,
                "is_correct": final_option == case["ground_truth"]["correct_option"],
                "num_discussion_rounds": actual_rounds,
                "max_discussion_rounds": a.rounds,
                "max_agent_turns": a.rounds,
                "termination_reason": termination_reason,
                "claim_order": a.claim_order,
                "doctor_initial_option": doctor_initial_option,
                "agent_initial_option": agent_initial_option,
                "agent_first_response_option": agent_initial_option,
                "initial_disagreement": bool(
                    doctor_initial_option
                    and agent_initial_option
                    and doctor_initial_option != agent_initial_option
                ),
                "doctor_final_option": final_option,
                "condition": "collaboration_without_belief"
                if a.disable_belief
                else "collaboration_with_local_judgment",
                "turns": dialogue,
                "models": {
                    "doctor": a.doctor_model,
                    "reconstructor": a.reconstructor_model,
                    "agent": a.model,
                },
                "generation": {
                    "doctor_temperature": 0.4,
                    "agent_do_sample": False,
                    "max_tokens": a.max_tokens,
                    "prompt_version": a.prompt_version,
                    "seed": a.seed,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{n}/{len(cases)} {case['case_id']} final={final_option or '?'} correct={record['is_correct']}",
                flush=True,
            )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
