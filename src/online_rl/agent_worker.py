"""Execution only: one Agent generation worker process per non-primary GPU (SHERPA_AGENT_PROCESS_WORKERS=1).

The in-process replicas of SharedPolicy share one Python interpreter, so concurrent generations serialize on the GIL.
Here replica k >= 1 is its own process on its own GPU, holding exactly one copy of the base model + LoRA, built the same way
as the in-process replica (bf16, gradient checkpointing + input grads enabled, eval) and running the same
generate_with_trace call. The primary replica (cuda:0, the trained model) stays in the trainer process.

Protocol (multiprocessing Pipe, one outstanding request per worker; the parent holds that replica's lock):
  {'op': 'load', 'state': {name: cpu tensor}, 'use_cache': bool}            -> {'adapter_sha256': sha}
  {'op': 'generate', 'arm', 'adapter_sha256', 'prompt', 'prompt_input_ids', 'seed', 'max_new_tokens', 'segment'}
                                                                             -> {'out': generate_with_trace dict, 'gen_s': s}
  {'op': 'stop'}
Any mismatch (arm, adapter SHA, prompt token ids) or exception is sent back as {'error': ...} and the worker exits:
the parent raises, so a run never continues on a wrong or unsynchronized adapter."""
from __future__ import annotations
import os


def worker_main(conn, physical_gpu: str, init: dict) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = physical_gpu               # before any CUDA call: this process sees one GPU
    import time, traceback
    try:
        import torch
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.online_rl import environment as ST
        from src.online_rl.integrated.continuation import adapter_sha256
        if torch.cuda.device_count() != 1: raise RuntimeError(f'worker for GPU {physical_gpu} sees {torch.cuda.device_count()} GPUs')
        dev = torch.device('cuda:0'); lc = init['lora']
        tok = AutoTokenizer.from_pretrained(init['base_model'], trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        b = AutoModelForCausalLM.from_pretrained(init['base_model'], dtype=torch.bfloat16, trust_remote_code=True)
        m = get_peft_model(b, LoraConfig(task_type='CAUSAL_LM', r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc.get('dropout', 0.0),
                                         target_modules=lc['target_modules'], use_rslora=lc.get('use_rslora', True)))
        m.to(dev); m.gradient_checkpointing_enable(); m.enable_input_require_grads(); m.eval()
        sha = None                                                  # no generation before the first load
        conn.send({'ready': True, 'pid': os.getpid(), 'device_name': torch.cuda.get_device_name(dev)})
    except BaseException:
        conn.send({'error': traceback.format_exc()}); return
    while True:
        msg = conn.recv()
        try:
            op = msg['op']
            if op == 'stop': return
            if op == 'load':
                set_peft_model_state_dict(m, {k: t.to(dev) for k, t in msg['state'].items()}); m.config.use_cache = msg['use_cache']
                sha = adapter_sha256(m); conn.send({'adapter_sha256': sha}); continue
            if op != 'generate': raise RuntimeError(f'unknown op {op!r}')
            if msg['arm'] != init['arm']: raise RuntimeError(f"worker of arm {init['arm']!r} got a request of arm {msg['arm']!r}")
            if sha is None or msg['adapter_sha256'] != sha:
                raise RuntimeError(f"adapter mismatch: worker {sha and sha[:12]} vs expected {msg['adapter_sha256'][:12]} ({msg['segment']})")
            m.eval(); t0 = time.time()
            with torch.cuda.device(dev): torch.cuda.manual_seed(msg['seed'])        # this GPU's generator only
            out = ST.generate_with_trace(m, tok, msg['prompt'], msg['max_new_tokens'], init['temperature'], init['top_p'], init['top_k'])
            torch.cuda.synchronize(dev); gen_s = time.time() - t0
            if out['prompt_input_ids'] != msg['prompt_input_ids']: raise RuntimeError(f"prompt token ids differ ({msg['segment']})")
            conn.send({'out': out, 'gen_s': gen_s})
        except BaseException:
            conn.send({'error': traceback.format_exc()}); return
