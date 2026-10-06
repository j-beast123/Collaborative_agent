"""Device-separated policy components. See gpu_integration.md for runner wiring.
GPU IDs are logical IDs within CUDA_VISIBLE_DEVICES; model replicas share weights, not objects.
"""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
import json
import os
import shutil
import tempfile
import uuid
ROOT = Path(__file__).resolve().parents[2]


class SharedPolicy:
    """Loads base model + LoRA once; generate() samples with the current adapter, the updater trains the same object."""
    def __init__(self, cfg: dict, key: str, *, device: str | None = None, trainable: bool = True):
        from src.online_rl.cuda_environment import configure_cuda_linker
        configure_cuda_linker(ROOT)
        import torch
        from peft import LoraConfig, get_peft_model, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.online_rl import environment as ST
        self.ST = ST
        self.trainable = trainable
        self.device = torch.device(device or cfg.get('gpu', {}).get('train_device', 'cuda:4'))
        if self.device.type != 'cuda':
            raise ValueError('This component requires a CUDA device')
        self.policy_version = 'initial'
        self.torch, self.cfg, self.key = torch, cfg, key; p = cfg['policy']
        self.tok = AutoTokenizer.from_pretrained(p['base_model'], trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(p['base_model'], dtype=torch.bfloat16, trust_remote_code=True)
        lc = p['lora']
        self.model = (PeftModel.from_pretrained(base, p['init_adapter'], is_trainable=trainable) if p.get('init_adapter') else
                      get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc.get('dropout', 0.0),
                                                      target_modules=lc['target_modules'], use_rslora=lc.get('use_rslora', True))))
        self.model.to(self.device)
        if trainable:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        else:
            self.model.requires_grad_(False)
        self.model.eval()
        ro = cfg['rollout']
        self.args = Namespace(rounds=ro.get('rounds', 5), doctor_model=ro['doctor_model'], max_tokens_doctor=ro['max_tokens_doctor'], max_completion_length=ro['max_completion_length'],
                              generation_temperature=ro['temperature'], top_p=ro['top_p'], top_k=self.ST.resolve_generation_top_k(self.model, ro.get('top_k')))
        self.agent_system = (ROOT / 'prompts/agent.txt').read_text(); self.doctor_system = (ROOT / 'prompts/user_simulator.txt').read_text()

    def identity(self): return {'base_model': self.cfg['policy']['base_model'], 'lora': self.cfg['policy']['lora'], 'init_adapter': self.cfg['policy'].get('init_adapter'),
                                'agent_prompt': 'prompts/agent.txt', 'doctor_prompt': 'prompts/user_simulator.txt', 'rollout': self.cfg['rollout']}

    def generate(self, case: dict, seed: int) -> dict:
        self.model.eval(); self.torch.manual_seed(seed)
        with self.torch.cuda.device(self.device), self.torch.inference_mode():
            return self.ST.generate_trajectory(self.model, self.tok, case, self.doctor_system, self.agent_system, self.key, self.args)

    # Continuations reuse src/prompt_opt/harness.Env bound to THIS policy; rounds and Doctor model come from self.args (config).
    def _H(self):
        from src.prompt_opt import harness as H
        return H

    def agent_user_prompt(self, case, prefix, t): return self._H().Env.agent_user_prompt(self, case, prefix, t)

    def agent(self, user, seed):
        self.model.eval()
        with self.torch.cuda.device(self.device), self.torch.inference_mode():
            return self._H().Env.agent(self, user, seed)

    def natural(self, case, prefix, turn, seed): return self.agent(self.agent_user_prompt(case, prefix, turn), seed)

    def continue_from(self, case, prefix, turn, fixed_action, seed):
        self.model.eval()
        with self.torch.cuda.device(self.device), self.torch.inference_mode():
            return self._H().Env.continue_from(self, case, prefix, turn, fixed_action, seed)

    def sample_completions(self, case, prefix, turn, seeds) -> list[dict]:
        """G completions from EXACTLY one frozen prompt (normal policy prompt for h_t; no rubric). Each trace carries the
        prompt string / token ids, so identical prompt hashes can be asserted before the loss."""
        self.model.eval(); a = self.args
        prompt = self.ST.chat_prompt(self.tok, self.agent_system, self.agent_user_prompt(case, prefix, turn)); out = []
        for s in seeds:
            self.torch.manual_seed(s)
            with self.torch.cuda.device(self.device), self.torch.inference_mode():
                tr = self.ST.generate_with_trace(self.model, self.tok, prompt, a.max_completion_length, a.generation_temperature, a.top_p, a.top_k)
            out.append({'stage': f'A{turn}', 'prompt_text': prompt, 'seed': s, **tr, 'policy_version': self.policy_version})
        return out

    def reload_adapter(self, checkpoint: str):
        """Reload only at a batch barrier; never while a case/rerun is active."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        path = Path(checkpoint)
        marker = path / 'COMPLETE.json'
        if not marker.is_file():
            raise RuntimeError(f'Incomplete checkpoint: {path}')
        info = json.loads(marker.read_text())
        set_peft_model_state_dict(self.model, load_file(str(path / 'adapter' / 'adapter_model.safetensors')))
        self.policy_version = info['policy_version']
        self.model.eval()
        return self.policy_version

    def guided(self, *a, **k):
        raise RuntimeError('the Agent policy never sees a rubric')


class GRPOUpdater:
    """Optimizer, checkpoint save/load; the update itself is src/online_rl/integrated/grpo.IntegratedGRPOUpdater."""

    def __init__(self, policy: SharedPolicy, cfg: dict):
        if not policy.trainable:
            raise ValueError('Updater requires a trainable policy replica')
        self.policy, self.o = policy, cfg['optimizer']; torch = policy.torch
        self.completed_cases = set()
        self.checkpoint_every_cases = cfg.get('checkpoint', {}).get('every_cases', 30)
        if self.checkpoint_every_cases <= 0:
            raise ValueError('every_cases must be positive')
        self.trainable = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable, lr=self.o['learning_rate'])
        self.steps = 0

    def identity(self): return dict(self.o)

    def save(self, path: Path) -> str:
        """Publish a complete, immutable checkpoint by same-filesystem rename."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        tmp = Path(tempfile.mkdtemp(prefix=path.name + '.pending-', dir=path.parent))
        version = uuid.uuid4().hex
        try:
            self.policy.model.save_pretrained(tmp / 'adapter', safe_serialization=True)
            self.policy.torch.save({
                'optimizer': self.optimizer.state_dict(), 'steps': self.steps,
                'completed_cases': sorted(self.completed_cases),
                'torch_rng': self.policy.torch.get_rng_state(),
                'cuda_rng': self.policy.torch.cuda.get_rng_state(self.policy.device),
            }, tmp / 'trainer_state.pt')
            (tmp / 'COMPLETE.json').write_text(json.dumps({
                'policy_version': version, 'steps': self.steps,
                'completed_cases': len(self.completed_cases),
                'policy': self.policy.identity(),
            }, ensure_ascii=False, indent=2))
            os.rename(tmp, path)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self.policy.policy_version = version
        return str(path)

    def mark_case_completed(self, case_id: str, checkpoint_root: Path) -> str | None:
        """Runner calls AFTER the case's generation/validation/scoring/update commit.
        Count cases, not leaves, reruns or optimizer steps. Calls must be serialized.
        """
        if case_id in self.completed_cases:
            return None
        self.completed_cases.add(case_id)
        count = len(self.completed_cases)
        if count % self.checkpoint_every_cases:
            return None
        try:
            return self.save(Path(checkpoint_root) / f'case_{count:06d}')
        except BaseException:
            self.completed_cases.remove(case_id)
            raise

    def load(self, path: str):
        p = Path(path)
        self.policy.reload_adapter(path)
        st = self.policy.torch.load(p / 'trainer_state.pt', map_location='cpu', weights_only=False)
        self.optimizer.load_state_dict(st['optimizer'])
        self.steps = st['steps']
        self.completed_cases = set(st.get('completed_cases', []))
        self.policy.torch.set_rng_state(st['torch_rng'])
        if 'cuda_rng' in st:
            self.policy.torch.cuda.set_rng_state(st['cuda_rng'], self.policy.device)


def build_train_policy(cfg: dict, key: str) -> SharedPolicy:
    return SharedPolicy(cfg, key, device=cfg.get('gpu', {}).get('train_device', 'cuda:4'), trainable=True)


def build_rollout_policy(cfg: dict, key: str, worker_index: int) -> SharedPolicy:
    devices = cfg.get('gpu', {}).get('rollout_devices', ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    return SharedPolicy(cfg, key, device=devices[worker_index], trainable=False)
