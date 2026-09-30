"""Generate doctor-agent clinical collaboration trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.belief import build_agent_response_prompt, build_local_judgment_prompt, parse_local_judgment


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/interim/medmcqa/medicine_sample_500_label_corrected.jsonl")
    p.add_argument("--output", default="outputs/collaboration/collaboration_pilot_10.jsonl")
    p.add_argument("--model", default="/home/kai/models/Qwen3-8B")
    p.add_argument("--doctor-model", default="google/gemma-3-4b-it")
    p.add_argument("--reconstructor-model", default="google/gemini-2.5-flash-lite")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--claim-order", choices=("doctor_first", "agent_first"), default="doctor_first")
    return p.parse_args()


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def case_text(case: dict) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in case["options"].items())
    return f"Question:\n{case['question']}\n\nAnswer choices:\n{options}"


def dialogue_text(dialogue: list[dict]) -> str:
    return "\n\n".join(f"{x['speaker'].title()}: {x['text']}" for x in dialogue) or "(No previous dialogue.)"


def doctor_call(key: str, model: str, system: str, user: str, max_tokens: int) -> str:
    key = key.strip()
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
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
            delay = float(retry_after) if retry_after else min(60, 2 ** attempt)
            print(f"OpenRouter HTTP {error.code}; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)


def reconstruct_call(key: str, model: str, case: dict, dialogue: list[dict], max_tokens: int) -> dict:
    system = (
        "Extract only explicit claims and evidence from the latest utterances. "
        "Do not infer unstated beliefs, fill gaps, judge correctness, or answer. "
        "Resolve pronouns such as it, that, and this to the relevant numbered "
        "statement using the supplied recent context. Return compact JSON only "
        "with focal_claim, doctor_position, agent_position, doctor_claims, "
        "doctor_evidence, agent_claims, and agent_evidence. The focal_claim "
        "must be a standalone proposition. Do not invent a position when a "
        "speaker is silent; use an empty string. Each array item must be short."
    )
    user = (
        f"Clinical question:\n{case['question']}\n\n"
        f"Answer choices:\n{json.dumps(case['options'], ensure_ascii=False)}\n\n"
        f"Prior turn for resolving references only:\n{dialogue_text(dialogue[-2:-1])}\n\n"
        f"Current doctor utterance:\n{dialogue_text(dialogue[-1:])}"
    )
    raw = doctor_call(key, model, system, user, max_tokens)
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return {"raw_response": raw, "parse_status": "invalid"}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_status": "invalid"}
    required = {"focal_claim", "doctor_position", "agent_position", "doctor_claims", "doctor_evidence", "agent_claims", "agent_evidence"}
    if not isinstance(value, dict) or not required.issubset(value):
        return {"raw_response": raw, "parse_status": "invalid"}
    for field in ("doctor_claims", "doctor_evidence", "agent_claims", "agent_evidence"):
        if not isinstance(value[field], list) or not all(isinstance(item, str) for item in value[field]):
            return {"raw_response": raw, "parse_status": "invalid", "invalid_field": field}
        value[field] = value[field][:5]
    for field in ("focal_claim", "doctor_position", "agent_position"):
        if not isinstance(value[field], str):
            return {"raw_response": raw, "parse_status": "invalid", "invalid_field": field}
        value[field] = value[field][:500]
    value["raw_response"] = raw
    value["parse_status"] = "valid"
    return value


def agent_call(model, tokenizer, system: str, user: str, max_tokens: int) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = {k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def option(text: str) -> str:
    hits = re.findall(r"(?:final answer|answer|option|choice)\s*[:\-]?\s*(?:is\s*)?(?:\*\*)?([A-D])\b", text, re.I)
    if not hits:
        hits = re.findall(r"(?:^|\n)\s*(?:\*\*)?([A-D])(?:\*\*)?\s*[.):-]", text, re.I)
    if not hits:
        hits = re.findall(r"(?:leaning|prefer|favor|support|choose|select)[^A-D\n]{0,30}\b([A-D])\b", text, re.I)
    return hits[-1].upper() if hits else ""


def main() -> None:
    a = args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    root = Path(__file__).resolve().parents[1]
    all_cases = load(root / a.input)[:a.max_cases]
    if not 0 <= a.shard_index < a.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    cases = all_cases[a.shard_index::a.num_shards]
    doctor_system = (root / "prompts/user_simulator.txt").read_text(encoding="utf-8")
    agent_system = (root / "prompts/agent.txt").read_text(encoding="utf-8")
    judgment_system = (root / "prompts/local_epistemic_judgment.txt").read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    out = root / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for n, case in enumerate(cases, 1):
            dialogue = []
            doctor_initial_option = ""
            agent_initial_option = ""
            for turn in range(1, a.rounds + 1):
                prompt = f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nRespond naturally as the physician. This is discussion turn {turn} of {a.rounds}."
                doctor = doctor_call(key, a.doctor_model, doctor_system, prompt, a.max_tokens)
                if turn == 1:
                    doctor_initial_option = option(doctor)
                dialogue.append({"speaker": "doctor", "text": doctor, "round": turn})
                reconstruction = reconstruct_call(key, a.reconstructor_model, case, dialogue, 384)
                if reconstruction.get("parse_status") != "valid":
                    # Do not turn a parser failure into fabricated empty evidence.
                    reconstruction = {
                        "doctor_claims": [], "doctor_evidence": [],
                        "agent_claims": [], "agent_evidence": [],
                        "parse_status": "invalid",
                        "raw_response": reconstruction.get("raw_response", ""),
                    }
                reconstructed_context = json.dumps(
                    {key: reconstruction.get(key, []) for key in ("doctor_claims", "doctor_evidence", "agent_claims", "agent_evidence")},
                    ensure_ascii=False,
                    indent=2,
                )
                judgment_prompt = build_local_judgment_prompt(
                    question=case["question"],
                    options=case["options"],
                    doctor_claims=reconstruction.get("doctor_claims", []),
                    agent_claims=reconstruction.get("agent_claims", []),
                    focal_claim=reconstruction.get("focal_claim", ""),
                    doctor_position=reconstruction.get("doctor_position", ""),
                    agent_position=reconstruction.get("agent_position", ""),
                    claim_order=a.claim_order,
                )
                judgment_raw = agent_call(model, tokenizer, judgment_system, judgment_prompt, 256)
                judgment = parse_local_judgment(judgment_raw)
                agent_prompt = build_agent_response_prompt(
                    question=case["question"],
                    options=case["options"],
                    reconstructed_context=reconstructed_context,
                    judgment=judgment,
                )
                response = agent_call(model, tokenizer, agent_system, agent_prompt, a.max_tokens)
                if turn == 1:
                    agent_initial_option = option(response)
                dialogue.append({"speaker": "agent", "text": response, "round": turn, "reconstruction": reconstruction, "judgment": judgment})
                if turn >= 2:
                    recent = [option(x["text"]) for x in dialogue[-4:] if option(x["text"])]
                    if len(recent) >= 4 and len(set(recent)) == 1:
                        break
            final_prompt = f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nThis is the final decision. State the option best supported by the complete discussion. End with exactly: Final answer: option <A-D> - <answer text>"
            final = doctor_call(key, a.doctor_model, doctor_system, final_prompt, a.max_tokens)
            dialogue.append({"speaker": "doctor", "text": final, "round": a.rounds + 1, "final": True})
            final_option = option(final)
            actual_rounds = sum(1 for item in dialogue if item["speaker"] == "agent")
            record = {"case_id": case["case_id"], "trajectory_id": f"pilot_{case['case_id']}", "question": case["question"], "options": case["options"], "ground_truth": case["ground_truth"], "final_option": final_option, "is_correct": final_option == case["ground_truth"]["correct_option"], "num_discussion_rounds": actual_rounds, "max_discussion_rounds": a.rounds, "claim_order": a.claim_order, "doctor_initial_option": doctor_initial_option, "agent_initial_option": agent_initial_option, "doctor_final_option": final_option, "condition": "collaboration_with_local_judgment", "turns": dialogue, "models": {"doctor": a.doctor_model, "reconstructor": a.reconstructor_model, "agent": a.model}, "max_tokens": a.max_tokens}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{n}/{len(cases)} {case['case_id']} final={final_option or '?'} correct={record['is_correct']}", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
