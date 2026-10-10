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
    # Keep sampled-token logprobs on-device until one CPU transfer. Gathering with a
    # tensor index avoids a device-to-host synchronization for every generated token.
    token_logprobs = [
        F.log_softmax(step_logits.float(), dim=-1).gather(-1, token_id.reshape(1, 1)).reshape(())
        for step_logits, token_id in zip(generated.logits, completion_ids[0])
    ]
    old_policy_logprobs = torch.stack(token_logprobs).detach().cpu().tolist() if token_logprobs else []
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
                        key: str, args, doctor=None, seed: int | None = None, agent_text=None, opening=None) -> dict:
    if opening is not None and doctor is None: raise ValueError('frozen openings need Doctor protocol v2')
    if doctor is not None:
        return generate_trajectory_v2(model, tokenizer, case, agent_system, args, doctor, seed, agent_text, opening)
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


def generate_trajectory_v2(model: Any, tokenizer: Any, case: dict, agent_system: str, args, doctor, seed: int | None, agent_text=None,
                           opening=None) -> dict:
    """One episode under Doctor protocol v2 (src/online_rl/doctor.py): seeded local Doctor turns, the hidden status cut
    from the visible dialogue, adaptive halting after a Doctor turn, then the usual FINAL. The Agent side is unchanged.
    opening(case, seed) -> D1: the frozen shared opening (src/online_rl/doctor_openings.py), read instead of generated."""
    from src.online_rl.doctor import doctor_seed, minimum_agent_turns
    if seed is None: raise ValueError('protocol-v2 episodes need an episode seed (it derives every Doctor seed)')
    cid = case['case_id']; dialogue: list[dict] = []; agent_traces: list[dict] = []; statuses: list[dict] = []
    termination_reason = "max_agent_turns"; prev_status = None
    private_needs = []
    for round_index in range(1, args.rounds + 1):
        ds = doctor_seed(seed, cid, 'doctor', round_index)
        d = opening(case, ds) if (round_index == 1 and opening is not None) else doctor.turn(case, dialogue, round_index, args.rounds, ds)
        if d.get('private_need') is not None: private_needs.append({**d['private_need'], 'audit': d.get('need_audit', {})})
        dialogue.append({"speaker": "doctor", "stage": f"D{round_index}", "text": d['visible'], "round": round_index})
        statuses.append({'stage': f'D{round_index}', 'status': d['status'], 'status_error': d['status_error'], 'seed': d['seed']})
        if round_index > 1:
            why = doctor.halt(agent_turns=round_index - 1, status=d['status'], prev_status=prev_status,
                              last_agent=dialogue[-2]['text'], options=case['options'])
            if why:
                termination_reason = why; break
        prev_status = d['status']
        if round_index == 1:
            user_prompt = build_initial_agent_response_prompt(scenario=case.get("scenario", ""), question=case["question"],
                                                              options=case["options"], clinician_message=d['visible'])
        else:
            user_prompt = build_plain_agent_response_prompt(scenario=case.get("scenario", ""), question=case["question"],
                                                            options=case["options"], dialogue_history=dialogue_text(dialogue),
                                                            current_round=round_index, max_rounds=args.rounds,
                                                            is_final_round=round_index == args.rounds)
        if agent_text is not None:                                   # the policy's own Agent action protocol (precommit_v1)
            from src.online_rl.integrated.continuation import derive_seed
            prompt = None; trace = {'completion_text': agent_text(case, [dict(x) for x in dialogue], round_index, derive_seed(seed, cid, 'agent', round_index))}
        else:
            prompt = chat_prompt(tokenizer, agent_system, user_prompt)
            trace = generate_with_trace(model, tokenizer, prompt, args.max_completion_length, args.generation_temperature, args.top_p, args.top_k)
        agent_index = len(dialogue)
        dialogue.append({"speaker": "agent", "stage": f"A{round_index}", "text": trace["completion_text"], "round": round_index,
                         "trace_index": len(agent_traces)})
        agent_traces.append({"stage": f"A{round_index}", "dialogue_index": agent_index, "prompt_text": prompt, **trace})
        reason = stop_reason(round_index, d['visible'], trace["completion_text"]) if round_index >= minimum_agent_turns(doctor) else None
        if reason:
            termination_reason = reason; break
    final_text = doctor.final(case, dialogue, doctor_seed(seed, cid, 'final', 0))
    dialogue.append({"speaker": "doctor", "stage": "FINAL", "text": final_text, "final": True})
    return {"dialogue": dialogue, "agent_traces": agent_traces, "final_decision_text": final_text,
            "final_option": final_answer_option(final_text), "num_discussion_rounds": len(agent_traces),
            "max_discussion_rounds": args.rounds, "termination_reason": termination_reason, "doctor_status": statuses,
            **({'private_needs': private_needs} if private_needs else {})}
