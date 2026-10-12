"""Generate doctor-agent clinical collaboration trajectories."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


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
from src.common import retry as _retry
from src.common import usage as api_usage


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/interim/medmcqa/medicine_sample_500_label_corrected.jsonl",
    )
    p.add_argument(
        "--output", default="outputs/collaboration/collaboration_pilot_10.jsonl"
    )
    p.add_argument("--model", required=True, help="agent model path or OpenRouter model ID")
    p.add_argument(
        "--adapter",
        default="",
        help="optional PEFT/LoRA adapter directory applied to a local Agent model",
    )
    p.add_argument(
        "--agent-backend",
        choices=("local", "openrouter"),
        default="local",
        help="run the Agent from a local Transformers checkpoint or through OpenRouter",
    )
    p.add_argument(
        "--agent-temperature",
        type=float,
        default=0.0,
        help="Agent sampling temperature when --agent-backend=openrouter",
    )
    p.add_argument("--doctor-model", required=True, help="doctor simulator (GPT family); set per run")
    p.add_argument("--reconstructor-model", required=True, help="state reconstructor; set per run")
    p.add_argument("--doctor-prompt", default="prompts/user_simulator.txt", help="doctor system prompt file")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based index of the first input case; default preserves the original behavior.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Maximum agent responses/collaboration rounds; must match the training pipeline",
    )
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--reconstruction-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for the structured reconstruction call.",
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--claim-order", choices=("doctor_first", "agent_first"), default="doctor_first"
    )
    p.add_argument(
        "--final-decision-role",
        choices=("doctor", "agent"),
        required=True,
        help="Who states the final answer after the discussion; must match the training pipeline.",
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
    p.add_argument(
        "--rubric-set",
        default="",
        help="Optional conditional rubric guidance JSON passed to every agent turn.",
    )
    return p.parse_args()


def load(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def load_rubric_guidance(path: Path | None) -> tuple[str, str]:
    """Render retained rubrics as conditional agent guidance without hidden state."""
    if path is None:
        return "", ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rubrics = payload.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError(f"rubric set must contain a non-empty 'rubrics' list: {path}")

    entries = []
    for index, rubric in enumerate(rubrics, 1):
        name = str(rubric.get("name") or f"rubric_{index}").strip()
        condition = str(rubric.get("applicability_condition") or "").strip()
        behavior = str(rubric.get("required_behavior") or "").strip()
        if not condition or not behavior:
            raise ValueError(
                f"rubric {name!r} must have applicability_condition and required_behavior"
            )
        entries.append(
            f"Rubric {index}: {name}\n"
            f"Applicability condition:\n{condition}\n"
            f"Required behavior:\n{behavior}"
        )

    guidance = (
        "CONDITIONAL COLLABORATION GUIDANCE\n\n"
        "The following are behavior guidelines discovered from earlier failure "
        "trajectories. They are not answer keys. At every turn, reassess whether "
        "each applicability condition is supported by the current dialogue. Apply "
        "only the guidelines whose conditions match the current interaction; "
        "ignore the others. If a guideline applies, perform its required behavior "
        "directly in this response and address the physician's specific reasoning. "
        "Do not mention this guidance, rubrics, scoring, hidden state, or future "
        "outcomes. Do not change or preserve an answer merely to satisfy a guideline. "
        "Use only available evidence and do not invent case details.\n\n"
        + "\n\n".join(entries)
    )
    version = str(payload.get("rubric_set_version") or payload.get("version") or path.name)
    return guidance, version


def case_text(case: dict) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in case["options"].items())
    scenario = case.get("scenario", "") or "(No separate patient vignette is provided.)"
    return f"Case/vignette:\n{scenario}\n\nQuestion:\n{case['question']}\n\nAnswer choices:\n{options}"


def dialogue_text(dialogue: list[dict]) -> str:
    return (
        "\n\n".join(f"{x['speaker'].title()}: {x['text']}" for x in dialogue)
        or "(No previous dialogue.)"
    )


def doctor_call(key: str, model: str, system: str, user: str, max_tokens: int, temperature: float = 0.4,
                extra: dict | None = None, meta: dict | None = None) -> str:
    """One OpenRouter chat completion (retry policy: src/common/retry.py). `extra` adds request fields (e.g. reasoning
    settings); `meta`, when given, receives finish_reason / native_finish_reason / usage of the returned completion."""
    key = key.strip()
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
            **(extra or {}),
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    retry = _retry.Retry(f"OpenRouter {model}")
    while True:
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                body = json.loads(response.read().decode())
            error = body.get("error") if isinstance(body, dict) else None
            if error:  # a 200 response carrying a provider error instead of a completion
                raise _ProviderError(error)
            text = body["choices"][0]["message"]["content"].strip()
            if meta is not None:
                choice = body["choices"][0]
                meta.update(finish_reason=choice.get("finish_reason"), native_finish_reason=choice.get("native_finish_reason"),
                            usage=body.get("usage"), model=body.get("model"), provider=body.get("provider"))
            api_usage.record("collaboration", body.get("model", model), body.get("usage"), retry.attempts)
            retry.succeeded()
            return text
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            reason = f"HTTP {error.code} {body[:300]}"
            if error.code not in _retry.RETRYABLE_STATUS:
                print(f"OpenRouter {model}: not retried: {reason}", flush=True)
                raise
            kind = "rate_limit" if _retry.rate_limited(error.code, body) else "transient"
            if not retry.wait(kind, reason, error.headers.get("Retry-After")):
                raise
        except _ProviderError as error:
            if error.code is not None and 400 <= error.code < 500 and error.code not in _retry.RETRYABLE_STATUS:
                print(f"OpenRouter {model}: not retried: {error}", flush=True)
                raise
            kind = "rate_limit" if _retry.rate_limited(error.code, str(error)) else "transient"
            if not retry.wait(kind, str(error)):
                raise
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, AttributeError) as error:
            # Timeouts and dropped connections, or a 200 response that is not a completion (empty content).
            if not retry.wait("transient", f"{type(error).__name__}: {str(error)[:200]}"):
                raise


class _ProviderError(RuntimeError):
    """An error object in a 200 OpenRouter response (e.g. the provider was overloaded mid-request)."""

    def __init__(self, error):
        error = error if isinstance(error, dict) else {"message": str(error)}
        code = error.get("code")
        self.code = code if isinstance(code, int) else int(code) if isinstance(code, str) and code.isdigit() else None
        super().__init__(f"provider error {error.get('code')}: {str(error.get('message', ''))[:300]} {json.dumps(error.get('metadata') or {})[:300]}")


# Prompts list the fields in this fixed order.
RECONSTRUCTION_FIELD_ORDER = (
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
)
RECONSTRUCTION_FIELDS = frozenset(RECONSTRUCTION_FIELD_ORDER)


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
                for field in RECONSTRUCTION_FIELD_ORDER
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


def agent_call(model, tokenizer, system: str, user: str, max_tokens: int, *, do_sample: bool = False,
               temperature: float | None = None, top_p: float | None = None) -> str:
    import torch
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
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            **({"temperature": temperature} if do_sample and temperature is not None else {}),
            **({"top_p": top_p} if do_sample and top_p is not None else {}),
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()


def final_answer_option(text: str) -> str:
    """Extract an explicit final option, including an out-of-range choice for audit."""
    pattern = r"final\s+answer\s*[:\-]?\s*(?:is\s*)?(?:option\s+)?(?:\*\*)?([A-Z])\b"
    matches = list(re.finditer(pattern, text, re.I))
    return matches[-1].group(1).upper() if matches else ""


def option(text: str, *, first: bool = False) -> str:
    final_choice = final_answer_option(text)
    if final_choice:
        return final_choice
    patterns = (
        r"(?:answer|option|choice)\s*[:\-]?\s*(?:is\s*)?(?:\*\*)?([A-E])\b",
        r"(?:^|\n)\s*(?:\*\*)?([A-E])(?:\*\*)?\s*[.):-]",
        r"(?:leaning|prefer|favor|support|choose|select)\s+(?:toward\s+|for\s+|option\s+)?([A-E])\b",
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
        r"[^.\n]{0,100}?(?:option\s+)?([A-E])\b",
        r"(?:option\s+)?([A-E])\b[^.\n]{0,40}(?:most likely|strongest contender|best choice)",
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


# Turn-level steps shared by main() and rubric-validation reruns
# (src/collaboration/rerun.py), so a rerun follows exactly the same protocol.


def doctor_turn_prompt(case: dict, dialogue: list[dict], turn: int, rounds: int) -> str:
    final_turn_instruction = (
        "\nThis is the final collaboration turn. Give your best-supported final answer explicitly."
        if turn == rounds
        else ""
    )
    return f"{case_text(case)}\n\nGrounding constraint: Use only the question, answer choices, dialogue, and established domain knowledge. If no patient vignette or event description is provided, do not invent one or add symptoms, injuries, tests, treatments, or events.\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nRespond naturally as the physician. This is discussion turn {turn} of {rounds}.{final_turn_instruction}"


FINAL_LABEL_ONLY_INSTRUCTION = (
    'This is the final decision. Select the option best supported by the complete discussion. '
    'Output exactly one line: Final answer: option <{labels}>. '
    'Replace the placeholder with only the selected option label. '
    'Do not include reasoning, answer text, Markdown, a STATUS line, or any other text.')


def final_decision_prompt(case: dict, dialogue: list[dict], *, label_only: bool = False) -> str:
    labels = "-".join(case["options"])
    if label_only:
        return f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\n{FINAL_LABEL_ONLY_INSTRUCTION.format(labels=labels)}"
    return f"{case_text(case)}\n\nConversation so far:\n{dialogue_text(dialogue)}\n\nThis is the final decision. State the option best supported by the complete discussion. End with exactly: Final answer: option <{labels}> - <answer text>"


def belief_agent_prompt(
    case: dict,
    dialogue: list[dict],
    turn: int,
    rounds: int,
    claim_order: str,
    *,
    reconstruct,
    judge,
):
    """Reconstruction -> turn-local Q1/Q2 judgment -> agent prompt (turn >= 2)."""
    reconstruction = reconstruct(case, dialogue)
    if reconstruction.get("parse_status") != "valid":
        return None, reconstruction, None, "reconstruction_failure"
    clean_reconstruction = {k: reconstruction[k] for k in RECONSTRUCTION_FIELD_ORDER}
    reconstructed_context = json.dumps(clean_reconstruction, ensure_ascii=False, indent=2)
    judgment_prompt = build_local_judgment_prompt(
        question=case["question"],
        options=case["options"],
        reconstruction=clean_reconstruction,
        scenario=case.get("scenario", ""),
        claim_order=claim_order,
    )
    judgment = parse_local_judgment(judge(judgment_prompt))
    if judgment.get("parse_status") != "valid":
        return None, reconstruction, judgment, "judgment_failure"
    clean_judgment = map_judgment_to_roles(judgment, claim_order)
    clean_judgment = {
        name: {key: value for key, value in clean_judgment[name].items()}
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
        max_rounds=rounds,
        is_final_round=turn == rounds,
    )
    return agent_prompt, reconstruction, judgment, "valid"


def stop_reason(turn: int, doctor: str, response: str, *, allow_mutual_conclusion: bool = True):
    """Early-termination rule after an agent turn, or None to continue."""
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
            return "mutual_agreement"
        if allow_mutual_conclusion and signals_conclusion(doctor) and signals_conclusion(response):
            return "mutual_conclusion"
    return None


def main() -> None:
    a = args()  # validate CLI arguments before loading heavy libraries/models
    random.seed(a.seed)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if a.agent_temperature < 0:
        raise ValueError("agent-temperature must be non-negative")
    root = ROOT
    if a.adapter and a.agent_backend != "local":
        raise ValueError("--adapter is supported only with --agent-backend local")
    adapter_path = None
    if a.adapter:
        requested_adapter = Path(a.adapter).expanduser()
        adapter_path = requested_adapter if requested_adapter.is_absolute() else root / requested_adapter
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"PEFT adapter is missing adapter_config.json: {adapter_path}")
    if a.start_index < 0:
        raise ValueError("start-index must be non-negative")
    rubric_guidance, rubric_set_version = load_rubric_guidance(
        root / a.rubric_set if a.rubric_set else None
    )
    all_cases = load(root / a.input)[a.start_index : a.start_index + a.max_cases]
    if not 0 <= a.shard_index < a.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    cases = all_cases[a.shard_index :: a.num_shards]
    doctor_system = (root / a.doctor_prompt).read_text(encoding="utf-8")
    agent_system = (root / "prompts/agent.txt").read_text(encoding="utf-8")
    judgment_system = (root / "prompts/local_epistemic_judgment.txt").read_text(
        encoding="utf-8"
    )
    if a.agent_backend == "local":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        # A PEFT adapter may contain tokenizer metadata written by a different
        # Transformers version. The adapter was trained on this base model, so
        # use the base tokenizer and load only LoRA weights from the adapter.
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.eval()

        def call_agent(system: str, user: str, max_tokens: int) -> str:
            return agent_call(model, tokenizer, system, user, max_tokens)
    else:
        def call_agent(system: str, user: str, max_tokens: int) -> str:
            return doctor_call(
                key,
                a.model,
                system,
                user,
                max_tokens,
                temperature=a.agent_temperature,
            )
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
                prompt = doctor_turn_prompt(case, dialogue, turn, a.rounds)
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
                    agent_prompt, reconstruction, judgment, judgment_status = belief_agent_prompt(
                        case,
                        dialogue,
                        turn,
                        a.rounds,
                        a.claim_order,
                        reconstruct=lambda c, d: reconstruct_call(
                            key, a.reconstructor_model, c, d, a.reconstruction_max_tokens
                        ),
                        judge=lambda p: call_agent(judgment_system, p, 384),
                    )
                    if judgment_status != "valid":
                        # reconstruction_failure or judgment_failure
                        termination_reason = judgment_status
                        print(f"{case['case_id']} {judgment_status}", flush=True)
                        break
                if rubric_guidance:
                    agent_prompt = agent_prompt + "\n\n" + rubric_guidance
                response = call_agent(agent_system, agent_prompt, a.max_tokens)
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
                reason = stop_reason(turn, doctor, response)
                if reason:
                    termination_reason = reason
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
            final_prompt = final_decision_prompt(case, dialogue)
            if a.final_decision_role == "doctor":
                final = doctor_call(
                    key, a.doctor_model, doctor_system, final_prompt, a.max_tokens
                )
            else:
                final = call_agent(agent_system, final_prompt, a.max_tokens)
            actual_rounds = sum(1 for item in dialogue if item["speaker"] == "agent")
            dialogue.append(
                {
                    "speaker": a.final_decision_role,
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
                "final_decision_role": a.final_decision_role,
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
                    "agent_adapter": str(adapter_path) if adapter_path is not None else None,
                },
                "generation": {
                    "doctor_temperature": 0.4,
                    "agent_backend": a.agent_backend,
                    "agent_adapter": str(adapter_path) if adapter_path is not None else None,
                    "agent_do_sample": False,
                    "agent_temperature": a.agent_temperature
                    if a.agent_backend == "openrouter"
                    else None,
                    "max_tokens": a.max_tokens,
                    "prompt_version": a.prompt_version,
                    "doctor_prompt": a.doctor_prompt,
                    "rubric_set": a.rubric_set or None,
                    "rubric_set_version": rubric_set_version or None,
                    "rubric_guidance": "conditional_applicability_and_required_behavior"
                    if rubric_guidance
                    else None,
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
