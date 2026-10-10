"""Frozen Doctor batching using HF's Gemma cache/mask path and per-request RNG.

The outer generate loop is greedy only because RowSample chooses each token
with its own torch.Generator first. No global RNG seeding or global multinomial.
"""
import time
import uuid
from src.online_rl.tree.batched_generation import GenerationBatcher, Request

VERSION = 'hf_doctor_batched_per_request_rng_v1'


def decode(model, tokenizer, requests, *, context=8192, token_budget=32768):
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
    device = next(model.parameters()).device
    prompts = [tokenizer(r.prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0].tolist() for r in requests]
    for ids, r in zip(prompts, requests):
        if not ids or r.max_tokens < 1 or len(ids) + r.max_tokens > context:
            raise ValueError('HF Doctor context budget exceeded; refusing truncation')
        if len(ids) + r.max_tokens > token_budget:
            raise ValueError('Doctor token budget cannot hold this single request')
    # Split a queue microbatch further if padding or requested output is large.
    groups = []; current = []
    for i in range(len(requests)):
        trial = current + [i]
        span = max(len(prompts[j]) for j in trial) + max(requests[j].max_tokens for j in trial)
        if current and (span > context or span * len(trial) > token_budget):
            groups.append(current); current = [i]
        else: current = trial
    if current: groups.append(current)
    results = [None] * len(requests)
    defaults = model.generation_config
    eos = defaults.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos_ids.discard(None)
    if not eos_ids: raise ValueError('Doctor requires an EOS token')
    pad = defaults.pad_token_id if defaults.pad_token_id is not None else tokenizer.pad_token_id
    if pad is None: pad = min(eos_ids)
    from src.common.live_gpu import activity
    for group in groups:
        started = time.monotonic(); batch_id = uuid.uuid4().hex
        width = max(len(prompts[i]) for i in group)
        ids = torch.tensor([[pad] * (width-len(prompts[i])) + prompts[i] for i in group], device=device)
        mask = torch.tensor([[0] * (width-len(prompts[i])) + [1]*len(prompts[i]) for i in group], device=device)
        rngs = [torch.Generator(device=device).manual_seed(requests[i].seed) for i in group]
        lengths = [0] * len(group); finished = [False] * len(group)
        class RowSample(LogitsProcessor):
            def __call__(self, input_ids, scores):
                selected = []
                for row, i in enumerate(group):
                    if finished[row] or lengths[row] >= requests[i].max_tokens:
                        selected.append(min(eos_ids)); continue
                    probabilities = (scores[row].float() / .4).softmax(-1)
                    token = int(torch.multinomial(probabilities, 1, generator=rngs[row]))
                    selected.append(token); lengths[row] += 1
                    finished[row] = token in eos_ids
                forced = torch.full_like(scores, -float('inf'))
                forced[torch.arange(len(group), device=device), torch.tensor(selected, device=device)] = 0
                return forced
        config = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
            max_new_tokens=max(requests[i].max_tokens for i in group),
            bos_token_id=defaults.bos_token_id, eos_token_id=eos, pad_token_id=pad)
        with activity("doctor_generate"), torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask, generation_config=config,
                logits_processor=LogitsProcessorList([RowSample()]), use_model_defaults=False, logits_to_keep=1)
        seconds = time.monotonic()-started
        for row, i in enumerate(group):
            tokens = output[row, width:width+lengths[row]].tolist()
            if len(prompts[i]) + len(tokens) >= context: raise ValueError('HF Doctor context full')
            body = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not body: raise ValueError('HF Doctor returned an empty reply')
            results[i] = dict(body=body, input_tokens=len(prompts[i]), output_tokens=len(tokens),
                batch_id=batch_id, batch_size=len(group), padded_context=width, request_seconds=seconds,
                batch_started_monotonic=started, sampling_protocol=VERSION)
    return results
