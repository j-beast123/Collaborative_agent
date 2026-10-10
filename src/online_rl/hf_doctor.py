"""Local, unquantized Gemma Doctor. No model/network/GPU work on construction.

Reuse the Doctor discussion/FINAL/halting methods. One frozen BF16 model on
cuda:0 serializes all Doctor roles. Per-call RNG touches only Doctor's GPU.
"""
from contextlib import contextmanager
import json
import os
import threading
import time

from src.common import usage
from src.online_rl.doctor import (OllamaDoctor, DoctorUnavailable, PROTOCOL_VERSION,
    V2_OPENING, V2_TURN, FINAL_TURN, STATUS_INSTRUCTION)
from src.online_rl.integrated.records import sha256_json, sha256_text


@contextmanager
def doctor_rng(torch, device, seed):
    # Do not call torch.manual_seed/fork_rng during generation: their CPU/global
    # effects could race with Assistant work on GPU1. GPU0 has one locked owner.
    index = torch.device(device).index
    with torch.cuda.device(index):
        state = torch.cuda.get_rng_state(index)
        try:
            torch.cuda.default_generators[index].manual_seed(int(seed))
            yield
        finally:
            torch.cuda.set_rng_state(state, index)


class HFDoctor(OllamaDoctor):
    def __init__(self, backend, halting, system, max_tokens):
        if backend.get('engine') != 'hf_local' or backend.get('device') != 'cuda:0':
            raise ValueError('HF Doctor requires hf_local on dedicated cuda:0')
        if backend.get('torch_dtype') != 'bfloat16' or backend.get('quantization') != 'none':
            raise ValueError('HF Doctor requires unquantized BF16')
        if backend.get('local_files_only') is not True or backend.get('attention_implementation') != 'sdpa':
            raise ValueError('HF Doctor requires local-only weights and SDPA')
        self.backend, self.system, self.max_tokens = backend, system, max_tokens
        self.halting = halting or {'enabled': False, 'min_agent_turns': 2}
        self.min_agent_turns = int(self.halting.get('min_agent_turns', 2))
        self.stable_choice_no_objection_after_turns = self.halting.get('stable_choice_no_objection_after_turns')
        late = self.stable_choice_no_objection_after_turns
        if late is not None and (type(late) is not int or late < self.min_agent_turns):
            raise ValueError('Invalid stable-choice halting turn')
        self.options = {**backend['options'], 'num_predict': max_tokens}
        if any(self.options.get(k) != v for k, v in {'temperature': .4, 'top_p': 1., 'top_k': 0,
                'min_p': 0., 'repeat_penalty': 1., 'presence_penalty': 0., 'frequency_penalty': 0., 'num_ctx': 8192}.items()):
            raise ValueError('HF Doctor requires the existing Doctor decoding values')
        self.final_response = backend['final_response']
        if self.final_response != {'mode': 'label_only_v1', 'max_tokens': 32}:
            raise ValueError('HF Doctor requires the existing label-only FINAL budget')
        from src.collaboration.generator import FINAL_LABEL_ONLY_INSTRUCTION
        self.identity = {k: backend[k] for k in ('engine', 'model', 'model_digest', 'quantization',
            'torch_dtype', 'transformers_version', 'attention_implementation', 'sampling_protocol')}
        self.identity.update(api='transformers_local_generate', options=self.options,
            source_model=backend.get('source_model'), protocol_version=PROTOCOL_VERSION,
            seed_policy='derive_seed(base, key, role, index) mod 2^31-1',
            chat_format='single user turn: system + blank line + user (local HF Gemma-3 template)',
            system_sha256=sha256_text(system), turn_prompt_sha256=sha256_text(V2_OPENING + V2_TURN + FINAL_TURN + STATUS_INSTRUCTION),
            halting=self.halting, final_response={**self.final_response, 'instruction_sha256': sha256_text(FINAL_LABEL_ONLY_INSTRUCTION)},
            context_policy='reject if input plus requested output exceeds 8192; never truncate',
            generation_config_policy='fresh_generation_config_with_use_model_defaults_false_v1',
            generation_errors='fail immediately, no zero/outcome substitution; caller format-repair policies unchanged')
        self.model = f"hf_local/{backend['model']}@{backend['model_digest'][:12]}"
        self.protocol = sha256_json(self.identity)
        self._lock = threading.RLock(); self._chat_metadata = threading.local()
        self._net = self._tok = None; self._verified = False

    def verify(self, require_loaded=False):
        from importlib.metadata import version
        from src.online_rl.tree.hf_doctor_setup import inventory
        actual = inventory(self.backend['model'])
        problems = []
        if actual['model_digest'] != self.backend['model_digest'] or actual['files'] != self.backend['file_manifest']:
            problems.append('Local HF Doctor file identity changed')
        if version('transformers') != self.backend['transformers_version']:
            problems.append('Transformers version differs from pinned HF Doctor runtime')
        self._verified = not problems
        if require_loaded and not problems:
            with self._lock: self._load()
        return {'engine': 'hf_local', 'digest': actual['model_digest'], 'problems': problems,
                'loaded': self._net is not None, 'device': self.backend['device']}

    def _load(self):
        if self._net is not None: return
        if not self._verified and self.verify()['problems']:
            raise DoctorUnavailable('HF Doctor identity/runtime verification failed')
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        self._tok = AutoTokenizer.from_pretrained(self.backend['model'], local_files_only=True, trust_remote_code=False)
        self._tok.get_chat_template()  # Fail before model loading if no instruction template exists.
        # Eager startup before any rollout thread; preserve initialization RNG.
        with torch.random.fork_rng(devices=[0]):
            self._net = Gemma3ForConditionalGeneration.from_pretrained(self.backend['model'],
                local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
                attn_implementation='sdpa')
        self._net.eval(); self._net.requires_grad_(False)

    def generation_config(self, max_tokens):
        from transformers import GenerationConfig
        defaults = self._net.generation_config
        # Start fresh so model-specific sampling/penalty defaults cannot silently
        # override the experiment's decoding. Keep only model token boundaries.
        return GenerationConfig(do_sample=True, temperature=.4, top_p=1., top_k=0, min_p=0.,
            repetition_penalty=1., num_beams=1, max_new_tokens=max_tokens, use_cache=True,
            bos_token_id=defaults.bos_token_id, eos_token_id=defaults.eos_token_id,
            pad_token_id=defaults.pad_token_id if defaults.pad_token_id is not None else self._tok.pad_token_id)

    def _chat(self, user, seed, max_tokens=None, system_override=None):
        from src.common.live_gpu import activity
        if system_override is not None and (not isinstance(system_override, str) or not system_override.strip()):
            raise ValueError('Doctor system_override must be a non-empty string')
        system = self.system if system_override is None else system_override
        budget = self.max_tokens if max_tokens is None else max_tokens
        if type(budget) is not int or budget < 1: raise ValueError('Doctor token budget must be a positive integer')
        messages = [{'role': 'user', 'content': f'{system}\n\n{user}'}]
        start = time.monotonic()
        with self._lock:
            self._load()
            import torch
            n_in = n_out = None; error = None; request_start = time.monotonic()
            try:
                prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self._tok(prompt, add_special_tokens=False, return_tensors='pt').to(self.backend['device'])
                n_in = int(inputs['input_ids'].shape[-1])
                if n_in + budget > self.options['num_ctx']:
                    raise DoctorUnavailable('HF Doctor context budget exceeded; refusing truncation')
                with activity('doctor_generate'), torch.inference_mode(), doctor_rng(torch, self.backend['device'], seed):
                    tokens = self._net.generate(**inputs, generation_config=self.generation_config(budget),
                                                use_model_defaults=False, logits_to_keep=1)
                completion = tokens[0, n_in:]; n_out = int(completion.numel())
                body = self._tok.decode(completion, skip_special_tokens=True).strip()
                if n_in + n_out >= self.options['num_ctx']: raise DoctorUnavailable('HF Doctor context full')
                if not body: raise DoctorUnavailable('HF Doctor returned an empty reply')
                return body
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                raise DoctorUnavailable('HF Doctor generation failed: ' + error) from exc
            finally:
                seconds = time.monotonic() - request_start
                self._chat_metadata.last = {'input_tokens': n_in, 'output_tokens': n_out,
                    'transport_retries': 0, 'request_seconds': seconds, 'retry_wait_seconds': 0.,
                    'queue_wait_seconds': request_start-start, 'error': error}
                usage.record('collaboration', self.model, {'prompt_tokens': n_in, 'completion_tokens': n_out, 'cost': 0.},
                    latency_seconds=time.monotonic()-start, request_seconds=seconds, retry_wait_seconds=0.)
                path = os.environ.get('SHERPA_HF_DOCTOR_AUDIT')
                if path:
                    with open(path, 'a') as stream:
                        stream.write(json.dumps({'model': self.model, 'stage': usage.current_stage(),
                            'seed': seed, 'messages_sha256': sha256_json(messages), 'max_tokens': budget,
                            **self._chat_metadata.last}) + '\n')
