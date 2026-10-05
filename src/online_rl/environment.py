"""Doctor-Agent MedMCQA collaboration: chat prompts, traced Agent sampling and full rollouts (Doctor via OpenRouter)."""
from __future__ import annotations
from typing import Any
from src.belief import build_initial_agent_response_prompt, build_plain_agent_response_prompt
from src.collaboration.generator import case_text, dialogue_text, doctor_call, final_answer_option, final_decision_prompt, stop_reason

# Raw policy log-probabilities before temperature/top-k/top-p warpers are applied.
POLICY_LOGPROB_SEMANTICS = "raw_policy_before_sampling_warpers_v1"


def chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_generation_top_k(model: Any, requested: int | None) -> int:
    value = requested
    if value is None:
        value = getattr(getattr(model, "generation_config", None), "top_k", 0)
    value = 0 if value is None else int(value)
    if value < 0:
        raise ValueError("top-k must be non-negative; use 0 to disable top-k filtering")
    return value


def generate_with_trace(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int,
                        temperature: float, top_p: float, top_k: int) -> dict:
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    sequences = generated.sequences
    completion_ids = sequences[:, prompt_length:]
    transition_scores = model.compute_transition_scores(
        sequences,
        generated.scores,
        normalize_logits=True,
    )
    if generated.logits is None or len(generated.logits) != completion_ids.shape[1]:
        raise RuntimeError("generation did not return one raw-policy logit tensor per completion token")
    old_policy_logprobs = [
        float(F.log_softmax(step_logits.float(), dim=-1)[0, token_id].detach().cpu())
        for step_logits, token_id in zip(generated.logits, completion_ids[0])
    ]
    sampling_logprobs = transition_scores[0].float().detach().cpu().tolist()
    completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
    return {
        "prompt_input_ids": encoded["input_ids"][0].detach().cpu().tolist(),
        "prompt_attention_mask": encoded["attention_mask"][0].detach().cpu().tolist(),
        "completion_ids": completion_ids[0].detach().cpu().tolist(),
        "old_policy_logprobs": old_policy_logprobs,
        "sampling_logprobs": sampling_logprobs,
        "logprob_semantics": POLICY_LOGPROB_SEMANTICS,
        "sampling_temperature": temperature,
        "sampling_top_p": top_p,
        "sampling_top_k": top_k,
        "completion_text": completion_text,
    }


def discussion_prompt(case: dict, dialogue: list[dict], turn: int, rounds: int) -> str:
    final_turn = (
        " This is the final allowed discussion turn, so present your best-supported reasoning."
        if turn == rounds else ""
    )
    return (
        f"{case_text(case)}\n\n"
        "Grounding constraint: Use only the question, answer choices, dialogue, and established domain knowledge. "
        "If no patient vignette or event description is provided, do not invent one.\n\n"
        f"Conversation so far:\n{dialogue_text(dialogue)}\n\n"
        f"Respond naturally as the physician. This is discussion turn {turn} of at most {rounds}. "
        "State your current reasoning and what you need from the collaborator. If the main uncertainty has been "
        "resolved and you are ready to conclude, say so explicitly; a separate final-decision prompt will record "
        f"the answer.{final_turn}"
    )


def final_outcome(case: dict, option: str) -> float | None:
    correct = str((case.get("ground_truth") or {}).get("correct_option") or "").strip().upper()
    selected = str(option or "").strip().upper()
    if not correct or not selected:
        return None
    return float(selected == correct)


def generate_trajectory(model: Any, tokenizer: Any, case: dict, doctor_system: str, agent_system: str,
                        key: str, args) -> dict:
    dialogue: list[dict] = []
    agent_traces: list[dict] = []
    termination_reason = "max_agent_turns"
    for round_index in range(1, args.rounds + 1):
        doctor = doctor_call(
            key,
            args.doctor_model,
            doctor_system,
            discussion_prompt(case, dialogue, round_index, args.rounds),
            args.max_tokens_doctor,
        )
        dialogue.append({"speaker": "doctor", "stage": f"D{round_index}", "text": doctor, "round": round_index})
        if round_index == 1:
            user_prompt = build_initial_agent_response_prompt(
                scenario=case.get("scenario", ""), question=case["question"],
                options=case["options"], clinician_message=doctor,
            )
        else:
            user_prompt = build_plain_agent_response_prompt(
                scenario=case.get("scenario", ""), question=case["question"],
                options=case["options"], dialogue_history=dialogue_text(dialogue),
                current_round=round_index, max_rounds=args.rounds,
                is_final_round=round_index == args.rounds,
            )
        prompt = chat_prompt(tokenizer, agent_system, user_prompt)
        trace = generate_with_trace(
            model, tokenizer, prompt, args.max_completion_length,
            args.generation_temperature, args.top_p, args.top_k,
        )
        agent_index = len(dialogue)
        dialogue.append({
            "speaker": "agent", "stage": f"A{round_index}",
            "text": trace["completion_text"], "round": round_index,
            "trace_index": len(agent_traces),
        })
        agent_traces.append({
            "stage": f"A{round_index}",
            "dialogue_index": agent_index,
            "prompt_text": prompt,
            **trace,
        })
        reason = stop_reason(round_index, doctor, trace["completion_text"])
        if reason:
            termination_reason = reason
            break

    final_text = doctor_call(
        key,
        args.doctor_model,
        doctor_system,
        final_decision_prompt(case, dialogue),
        args.max_tokens_doctor,
    )
    dialogue.append({"speaker": "doctor", "stage": "FINAL", "text": final_text, "final": True})
    return {
        "dialogue": dialogue,
        "agent_traces": agent_traces,
        "final_decision_text": final_text,
        "final_option": final_answer_option(final_text),
        "num_discussion_rounds": len(agent_traces),
        "max_discussion_rounds": args.rounds,
        "termination_reason": termination_reason,
    }
