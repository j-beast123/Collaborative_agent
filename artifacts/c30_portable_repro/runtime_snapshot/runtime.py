"""GPU components: one shared LoRA policy (the sampling model is the trained model) and the optimizer/checkpoint base
of the GRPO updater."""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def seeded_lora(base, lc: dict):
    """The fresh LoRA adapter of `base`, created under a private torch RNG seeded with lc['init_seed']: its initial weights
    depend only on (base, LoRA config, seed), not on the global RNG state or on anything drawn before, and the global RNG
    state is left as it was. Same architecture and initialization scheme as before; only the random draw is pinned."""
    import torch
    from peft import LoraConfig, get_peft_model
    seed = lc.get('init_seed')
    if not isinstance(seed, int) or isinstance(seed, bool): raise ValueError(f'policy.lora.init_seed must be an integer, got {seed!r}')
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc.get('dropout', 0.0),
                                               target_modules=lc['target_modules'], use_rslora=lc.get('use_rslora', True)))


def _segment(policy, name: str) -> tuple:
    """The segment-identity argument of _generate, passed only to process workers; the default path keeps its call shape."""
    return (name,) if getattr(policy, '_procs', None) is not None else ()


class SharedPolicy:
    """Loads base model + LoRA once; generate() samples with the current adapter, the updater trains the same object."""
    def __init__(self, cfg: dict, key: str, *, device: str = 'cuda:0'):
        import torch
        from peft import LoraConfig, get_peft_model, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.online_rl import environment as ST
        self.ST = ST
        self.torch, self.cfg, self.key = torch, cfg, key; p = cfg['policy']
        self.tok = AutoTokenizer.from_pretrained(p['base_model'], trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        # `torch_dtype` is accepted by both the pinned Transformers 4.53.x
        # runtime and newer releases. `dtype` is forwarded to some remote-code
        # model constructors by 4.53.x instead of being consumed here.
        base = AutoModelForCausalLM.from_pretrained(p['base_model'], torch_dtype=torch.bfloat16, trust_remote_code=True)
        lc = p['lora']
        if p.get('init_adapter'): self.model = PeftModel.from_pretrained(base, p['init_adapter'], is_trainable=True)
        elif lc.get('init_seed') is not None: self.model = seeded_lora(base, lc)
        else:                                                         # legacy configs without policy.lora.init_seed: unseeded
            self.model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc.get('dropout', 0.0),
                                                         target_modules=lc['target_modules'], use_rslora=lc.get('use_rslora', True)))
        self.model.to(torch.device(device)); self.model.gradient_checkpointing_enable(); self.model.enable_input_require_grads()
        ro = cfg['rollout']
        self.args = Namespace(rounds=ro['rounds'], doctor_model=ro['doctor_model'], max_tokens_doctor=ro['max_tokens_doctor'], max_completion_length=ro['max_completion_length'],
                              generation_temperature=ro['temperature'], top_p=ro['top_p'], top_k=self.ST.resolve_generation_top_k(self.model, ro.get('top_k')))
        self.agent_system = (ROOT / ro.get('agent_system_prompt_file', 'prompts/agent.txt')).read_text()
        self.doctor_system = (ROOT / ro.get('doctor_system_prompt_file', 'prompts/user_simulator.txt')).read_text()
        from src.online_rl.doctor import make_doctor
        self.doctor = make_doctor(ro, self.doctor_system)              # None: legacy OpenRouter Doctor
        import threading; self.gen_lock = threading.RLock()            # one generation at a time on the shared model
        self.precommit = ro.get('agent_protocol') == 'precommit_v1'
        if ro.get('agent_protocol') not in (None, 'precommit_v1'): raise ValueError(f"unknown rollout.agent_protocol {ro['agent_protocol']!r}")
        self._init_replicas(base_model=p['base_model'], lc=lc, init_adapter=p.get('init_adapter'))

    # ---------- execution only: Agent replicas (SHERPA_AGENT_REPLICAS, default 1 = the single shared model) ----------
    # Replica i is the same base model + LoRA on cuda:i (cuda:0 = the trained primary). Every logical generation is still ONE
    # sequence with its own seed; it runs on whichever replica is free, under that replica's lock, seeding only that GPU's
    # generator, so concurrent generations on other GPUs cannot touch its random stream. Replicas copy the primary's LoRA
    # weights whenever they change (mark_dirty) and their adapter hashes are verified (sync_replicas). No batch decoding.
    def _init_replicas(self, base_model: str, lc: dict, init_adapter):
        import os, queue, threading
        torch = self.torch; n = int(os.environ.get('SHERPA_AGENT_REPLICAS', '1'))
        if n < 1 or (n > 1 and n > torch.cuda.device_count()): raise ValueError(f'SHERPA_AGENT_REPLICAS={n} but {torch.cuda.device_count()} visible GPUs')
        if n > 1 and self.model.device != torch.device('cuda:0'):
            raise ValueError('Multiple Agent replicas require the primary on cuda:0')
        self.n_replicas, self.models, self.devices = n, [self.model], [self.model.device]
        self._rlocks = [threading.Lock() for _ in range(n)]; self._free = queue.Queue(); self._sync_lock = threading.Lock()
        self._weights_version, self._synced_version = 0, -1; self.exec_log = os.environ.get('SHERPA_EXEC_LOG'); self._log_lock = threading.Lock()
        self._procs = None; self._synced_sha = None
        if n > 1 and os.environ.get('SHERPA_AGENT_PROCESS_WORKERS') == '1':
            self._init_process_workers(n, base_model, lc)
            for i in range(n): self._free.put(i)
            return
        if n > 1:
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import AutoModelForCausalLM
            for k in range(1, n):
                b = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
                m = (PeftModel.from_pretrained(b, init_adapter, is_trainable=True) if init_adapter else
                     get_peft_model(b, LoraConfig(task_type='CAUSAL_LM', r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc.get('dropout', 0.0),
                                                  target_modules=lc['target_modules'], use_rslora=lc.get('use_rslora', True))))
                m.to(torch.device(f'cuda:{k}')); m.gradient_checkpointing_enable(); m.enable_input_require_grads(); m.eval()
                self.models.append(m); self.devices.append(torch.device(f'cuda:{k}'))
            self.sync_replicas()
        for i in range(n): self._free.put(i)

    # SHERPA_AGENT_PROCESS_WORKERS=1 (execution only, src/online_rl/agent_worker.py): replica k >= 1 is a separate process on the
    # k-th visible GPU with its own single model copy; replica 0 is still the primary (trained) model in this process. Same
    # sampling call, seeds, sync barrier and hash checks; every worker request carries arm, adapter SHA, prompt ids, seed,
    # max_new_tokens and segment identity, and the worker refuses any mismatch.
    def _init_process_workers(self, n: int, base_model: str, lc: dict):
        import multiprocessing as mp, os
        from src.online_rl.agent_worker import worker_main
        vis = os.environ.get('CUDA_VISIBLE_DEVICES')
        if not vis or len(vis.split(',')) < n: raise ValueError('process workers need CUDA_VISIBLE_DEVICES listing every Agent GPU')
        self.arm = os.environ.get('SHERPA_ARM') or self.cfg.get('condition') or self.cfg.get('output_dir', '')
        a = self.args; ctx = mp.get_context('spawn'); self._procs = [None]
        init = {'arm': self.arm, 'base_model': base_model, 'lora': lc, 'temperature': a.generation_temperature, 'top_p': a.top_p, 'top_k': a.top_k}
        for k in range(1, n):
            gpu = vis.split(',')[k]; parent, child = ctx.Pipe()
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu                 # inherited by the spawned child; restored below
            try: p = ctx.Process(target=worker_main, args=(child, gpu, init), daemon=True); p.start()
            finally: os.environ['CUDA_VISIBLE_DEVICES'] = vis
            child.close()                                            # a dead worker then reads as EOF, never a hang
            self._procs.append({'proc': p, 'conn': parent, 'gpu': gpu})
        self.device_names = [self.torch.cuda.get_device_name(0)]
        for w in self._procs[1:]:
            try: r = w['conn'].recv()
            except EOFError: raise RuntimeError(f"Agent worker on GPU {w['gpu']} died during start (exit code {w['proc'].join(5) or w['proc'].exitcode})")
            if 'error' in r: raise RuntimeError(f"Agent worker on GPU {w['gpu']} failed to start:\n{r['error']}")
            w['pid'] = r['pid']; self.device_names.append(r['device_name'])
        self.devices = [self.torch.device('cuda:0')] + [f"proc:gpu{w['gpu']}" for w in self._procs[1:]]
        import atexit; atexit.register(self._stop_workers)
        self.sync_replicas()

    def _stop_workers(self):
        for w in (self._procs or [])[1:]:
            try: w['conn'].send({'op': 'stop'})
            except Exception: pass

    def _worker_call(self, i: int, msg: dict) -> dict:
        w = self._procs[i]
        try: w['conn'].send(msg); r = w['conn'].recv()
        except (EOFError, OSError) as e: raise RuntimeError(f"Agent worker on GPU {w['gpu']} died ({type(e).__name__})")
        if 'error' in r: raise RuntimeError(f"Agent worker on GPU {w['gpu']}:\n{r['error']}")
        return r

    def _sync_process_workers(self, expected_sha256):
        from peft import get_peft_model_state_dict
        from src.online_rl.integrated.continuation import adapter_sha256
        sd = {k: t.detach().to('cpu') for k, t in get_peft_model_state_dict(self.model).items()}
        msg = {'op': 'load', 'state': sd, 'use_cache': self.model.config.use_cache}
        for w in self._procs[1:]:
            try: w['conn'].send(msg)
            except OSError as e: raise RuntimeError(f"Agent worker on GPU {w['gpu']} died ({type(e).__name__})")
        hashes = [adapter_sha256(self.model)]
        for i, w in enumerate(self._procs[1:], 1):
            try: r = w['conn'].recv()
            except (EOFError, OSError) as e: raise RuntimeError(f"Agent worker on GPU {w['gpu']} died ({type(e).__name__})")
            if 'error' in r: raise RuntimeError(f"Agent worker on GPU {w['gpu']}:\n{r['error']}")
            hashes.append(r['adapter_sha256'])
        if len(set(hashes)) != 1: raise RuntimeError(f'Agent replica adapter mismatch: {hashes}')
        if expected_sha256 is not None and hashes[0] != expected_sha256:
            raise RuntimeError(f'Agent adapter {hashes[0][:12]} differs from the committed policy {expected_sha256[:12]}')
        self._synced_sha = hashes[0]
        return {'replicas': self.n_replicas, 'devices': [str(d) for d in self.devices], 'device_names': self.device_names,
                'adapter_sha256': hashes, 'agent_workers_mode': 'process', 'worker_pids': [w['pid'] for w in self._procs[1:]]}

    def seed_all(self, seed: int):
        """The legacy global seed before an episode / continuation. With replicas every generation seeds its own GPU, and a
        global reseed from another thread would disturb a running generation, so it is skipped."""
        if self.n_replicas == 1: self.torch.manual_seed(seed)

    def mark_dirty(self):
        """The primary's LoRA weights changed (optimizer step, checkpoint load): replicas re-sync before the next generation."""
        self._weights_version += 1

    def sync_replicas(self, expected_sha256: str | None = None) -> dict:
        """Barrier: wait for every replica to be idle, copy the primary LoRA weights to every replica, and verify that every
        replica's adapter hash equals the primary's (and expected_sha256 when given)."""
        from src.online_rl.integrated.continuation import adapter_sha256
        with self._sync_lock:
            for l in self._rlocks: l.acquire()
            try:
                v = self._weights_version
                if self._procs is not None:
                    r = self._sync_process_workers(expected_sha256); self._synced_version = v; return r
                if self.n_replicas > 1:
                    from peft import get_peft_model_state_dict, set_peft_model_state_dict
                    sd = get_peft_model_state_dict(self.model)
                    for m, d in zip(self.models[1:], self.devices[1:]):
                        set_peft_model_state_dict(m, {k: t.to(d) for k, t in sd.items()}); m.config.use_cache = self.model.config.use_cache
                hashes = [adapter_sha256(m) for m in self.models]
                if len(set(hashes)) != 1: raise RuntimeError(f'Agent replica adapter mismatch: {hashes}')
                if expected_sha256 is not None and hashes[0] != expected_sha256:
                    raise RuntimeError(f'Agent adapter {hashes[0][:12]} differs from the committed policy {expected_sha256[:12]}')
                self._synced_version = v
                return {'replicas': self.n_replicas, 'devices': [str(d) for d in self.devices],
                        'device_names': [self.torch.cuda.get_device_name(d) for d in self.devices], 'adapter_sha256': hashes}
            finally:
                for l in self._rlocks: l.release()

    def on_primary(self, fn):
        """fn (teacher-forced log-probs on the trained primary) serialized with generation on the primary GPU."""
        if self.n_replicas == 1: return fn
        def run(*a, **k):
            with self._rlocks[0]: return fn(*a, **k)
        return run

    def _log_call(self, rec: dict):
        if not self.exec_log: return
        import json
        with self._log_lock, open(self.exec_log, 'a') as f: f.write(json.dumps(rec) + '\n')

    def identity(self): return {'base_model': self.cfg['policy']['base_model'], 'lora': self.cfg['policy']['lora'], 'init_adapter': self.cfg['policy'].get('init_adapter'),
                                'agent_prompt': 'prompts/agent.txt', 'doctor_prompt': 'prompts/user_simulator.txt', 'rollout': self.cfg['rollout']}

    def generate(self, case: dict, seed: int, opening=None) -> dict:
        """One evaluation episode; opening = the frozen shared D1 table of the split (src/online_rl/doctor_openings.py)."""
        self.model.eval(); self.seed_all(seed)
        return self.ST.generate_trajectory(self.model, self.tok, case, self.doctor_system, self.agent_system, self.key, self.args,
                                           doctor=self.doctor, seed=seed, agent_text=self.agent_text if self.precommit else None, opening=opening)

    # Continuations reuse src/prompt_opt/harness.Env bound to THIS policy; rounds and Doctor model come from self.args (config).
    def _H(self):
        from src.prompt_opt import harness as H
        return H

    def agent_user_prompt(self, case, prefix, t): return self._H().Env.agent_user_prompt(self, case, prefix, t)

    def agent(self, user, seed):
        if self.n_replicas > 1 or self.cfg['rollout'].get('agent_sampling_protocol') == 'per_row_rng_batch_v1':
            return self._generate(self.ST.chat_prompt(self.tok, self.agent_system, user), self.args.max_completion_length, seed)['completion_text']
        with self.gen_lock:                                       # seed + generate atomically (threaded reruns)
            self.model.eval(); return self._H().Env.agent(self, user, seed)

    def natural(self, case, prefix, turn, seed): return self.agent(self.agent_user_prompt(case, prefix, turn), seed)

    def continue_from(self, case, prefix, turn, fixed_action, seed, turn_cache=None):
        self.model.eval(); return self._H().Env.continue_from(self, case, prefix, turn, fixed_action, seed, turn_cache=turn_cache)

    def sample_completions(self, case, prefix, turn, seeds) -> list[dict]:
        """G completions from EXACTLY one frozen prompt (normal policy prompt for h_t; no rubric). Each trace carries the
        prompt string / token ids, so identical prompt hashes can be asserted before the loss."""
        return [self.agent_turn(case, prefix, turn, s) for s in seeds]

    def _generate(self, prompt: str, max_new_tokens: int, seed: int, segment: str = 'agent') -> dict:
        a = self.args
        if self.cfg['rollout'].get('agent_sampling_protocol') == 'per_row_rng_batch_v1':
            import os
            from src.online_rl.agent_batch import Batcher
            # Called concurrently: initialize exactly once under the existing model lock.
            if not hasattr(self,'_batcher'):
                with self.gen_lock:
                    if not hasattr(self,'_batcher'):
                        capacity=int(os.environ.get('SHERPA_AGENT_BATCH_SIZE',self.cfg.get('five_gpu',{}).get('agent_batch_size',1)))
                        if capacity not in (1,2,4,8):raise ValueError('Agent batch size must be 1, 2, 4 or 8')
                        self._batcher=Batcher(self,capacity)
            return self._batcher.submit(prompt,max_new_tokens,seed)
        if self._procs is not None: return self._generate_process(prompt, max_new_tokens, seed, segment)
        if self.n_replicas == 1:
            with self.gen_lock:                                   # seed + generate atomically (threaded reruns)
                self.model.eval(); self.torch.manual_seed(seed)
                from src.common.live_gpu import activity
                with activity('agent_generate'):
                    return self.ST.generate_with_trace(self.model, self.tok, prompt, max_new_tokens, a.generation_temperature, a.top_p, a.top_k)
        import time
        if self._synced_version != self._weights_version: self.sync_replicas()
        t0 = time.time(); i = self._free.get(); t1 = time.time()
        try:
            with self._rlocks[i]:
                if self._synced_version != self._weights_version: raise RuntimeError('Agent replica used before synchronization')
                m = self.models[i]; m.eval(); t2 = time.time()
                with self.torch.cuda.device(self.devices[i]): self.torch.cuda.manual_seed(seed)    # this GPU's generator only
                out = self.ST.generate_with_trace(m, self.tok, prompt, max_new_tokens, a.generation_temperature, a.top_p, a.top_k)
                self.torch.cuda.synchronize(self.devices[i]); t3 = time.time()
        finally:
            self._free.put(i)
        self._log_call({'t': t0, 'replica': i, 'wait_s': round(t1 - t0, 4), 'gen_s': round(t3 - t2, 4), 'max_new': max_new_tokens,
                        'tokens': len(out['completion_ids']), 'seed': seed})
        return out

    def _generate_process(self, prompt: str, max_new_tokens: int, seed: int, segment: str) -> dict:
        """_generate with process workers: replica 0 generates here on the primary exactly as the in-process path does;
        replica k >= 1 receives the full request identity and generates in its own process."""
        import time
        a = self.args
        if self._synced_version != self._weights_version: self.sync_replicas()
        t0 = time.time(); i = self._free.get(); t1 = time.time()
        try:
            with self._rlocks[i]:
                if self._synced_version != self._weights_version: raise RuntimeError('Agent replica used before synchronization')
                if i == 0:
                    m = self.model; m.eval(); t2 = time.time()
                    with self.torch.cuda.device(self.devices[0]): self.torch.cuda.manual_seed(seed)
                    out = self.ST.generate_with_trace(m, self.tok, prompt, max_new_tokens, a.generation_temperature, a.top_p, a.top_k)
                    self.torch.cuda.synchronize(self.devices[0]); t3 = time.time()
                else:
                    ids = self.tok(prompt, return_tensors='pt', truncation=True)['input_ids'][0].tolist(); t2 = time.time()
                    r = self._worker_call(i, {'op': 'generate', 'arm': self.arm, 'adapter_sha256': self._synced_sha, 'prompt': prompt,
                                              'prompt_input_ids': ids, 'seed': seed, 'max_new_tokens': max_new_tokens, 'segment': segment})
                    out = r['out']; t3 = time.time()
        finally:
            self._free.put(i)
        self._log_call({'t': t0, 'replica': i, 'wait_s': round(t1 - t0, 4), 'gen_s': round(t3 - t2, 4), 'max_new': max_new_tokens,
                        'tokens': len(out['completion_ids']), 'seed': seed, 'mode': 'process'})
        return out

    def agent_turn(self, case, prefix, turn, seed) -> dict:
        """One Agent action at (case, dialogue prefix, turn). Legacy: one completion of the Agent prompt. precommit_v1
        (src/online_rl/agent_protocol.py): a blind private assessment z, then the visible reply conditioned on the state and
        z; both are segments of the same action."""
        user = self.agent_user_prompt(case, prefix, turn); base = self.ST.chat_prompt(self.tok, self.agent_system, user)
        if not self.precommit:
            return {'stage': f'A{turn}', 'prompt_text': base, 'seed': seed, **self._generate(base, self.args.max_completion_length, seed)}
        from src.online_rl import agent_protocol as AP
        sa, sb = AP.stage_seeds(seed, case['case_id'])
        pa = self.ST.chat_prompt(self.tok, AP.ASSESS_SYSTEM, AP.blind_case_prompt(case)); z = self._generate(pa, AP.ASSESS_MAX_TOKENS, sa, *_segment(self, f'A{turn}/private_assessment'))
        pb = self.ST.chat_prompt(self.tok, self.agent_system, user + AP.HIDDEN_BLOCK.format(assessment=z['completion_text'].strip()))
        r = self._generate(pb, self.args.max_completion_length, sb, *_segment(self, f'A{turn}/visible_reply'))
        seg = lambda role, prompt, t, s: {'role': role, 'prompt_text': prompt, 'seed': s,
                                         **{k: t[k] for k in ('prompt_input_ids', 'completion_ids', 'old_policy_logprobs', 'completion_text')}}
        return {'stage': f'A{turn}', 'prompt_text': pb, 'seed': seed, **r, 'private_assessment': z['completion_text'],
                'state_prompt_hash': AP.state_hash(z['prompt_input_ids'], base), 'agent_protocol': AP.VERSION,
                'segments': [seg('private_assessment', pa, z, sa), seg('visible_reply', pb, r, sb)]}

    def agent_text(self, case, prefix, turn, seed) -> str:
        """The visible reply of agent_turn (continuations, evaluation episodes)."""
        return self.agent_turn(case, prefix, turn, seed)['completion_text']

    def guided(self, *a, **k):
        raise RuntimeError('the Agent policy never sees a rubric')


class GRPOUpdater:
    """Optimizer, checkpoint save/load; the update itself is src/online_rl/integrated/grpo.IntegratedGRPOUpdater."""

    def __init__(self, policy: SharedPolicy, cfg: dict):
        self.policy, self.o = policy, cfg['optimizer']; torch = policy.torch
        self.trainable = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable, lr=self.o['learning_rate'])
        self.steps = 0

    def identity(self): return dict(self.o)

    def save(self, path: Path) -> str:
        path.mkdir(parents=True, exist_ok=True); self.policy.model.save_pretrained(path / 'adapter')
        self.policy.torch.save({'optimizer': self.optimizer.state_dict(), 'steps': self.steps, 'torch_rng': self.policy.torch.get_rng_state()}, path / 'trainer_state.pt')
        return str(path)

    def load(self, path: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        p = Path(path); set_peft_model_state_dict(self.policy.model, load_file(str(p / 'adapter' / 'adapter_model.safetensors')))
        getattr(self.policy, 'mark_dirty', lambda: None)()
        st = self.policy.torch.load(p / 'trainer_state.pt', weights_only=False); self.optimizer.load_state_dict(st['optimizer']); self.steps = st['steps']
        self.policy.torch.set_rng_state(st['torch_rng'])
