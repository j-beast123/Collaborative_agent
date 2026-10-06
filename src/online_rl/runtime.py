"""GPU components: one shared LoRA policy (the sampling model is the trained model) and the optimizer/checkpoint base
of the GRPO updater."""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


class SharedPolicy:
    """Loads base model + LoRA once; generate() samples with the current adapter, the updater trains the same object."""
    def __init__(self, cfg: dict, key: str):
        import torch
        from peft import LoraConfig, get_peft_model, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.online_rl import environment as ST
        self.ST = ST
        self.torch, self.cfg, self.key = torch, cfg, key; p = cfg['policy']
        self.tok = AutoTokenizer.from_pretrained(p['base_model'], trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(p['base_model'], dtype=torch.bfloat16, trust_remote_code=True)
        lc = p['lora']
        self.model = (PeftModel.from_pretrained(base, p['init_adapter'], is_trainable=True) if p.get('init_adapter') else
                      get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=lc['r'], lora_alpha=lc['alpha'], lora_dropout=lc.get('dropout', 0.0),
                                                      target_modules=lc['target_modules'], use_rslora=lc.get('use_rslora', True))))
        self.model.to(torch.device('cuda:0')); self.model.gradient_checkpointing_enable(); self.model.enable_input_require_grads()
        ro = cfg['rollout']
        self.args = Namespace(rounds=ro['rounds'], doctor_model=ro['doctor_model'], max_tokens_doctor=ro['max_tokens_doctor'], max_completion_length=ro['max_completion_length'],
                              generation_temperature=ro['temperature'], top_p=ro['top_p'], top_k=self.ST.resolve_generation_top_k(self.model, ro.get('top_k')))
        self.agent_system = (ROOT / 'prompts/agent.txt').read_text(); self.doctor_system = (ROOT / 'prompts/user_simulator.txt').read_text()

    def identity(self): return {'base_model': self.cfg['policy']['base_model'], 'lora': self.cfg['policy']['lora'], 'init_adapter': self.cfg['policy'].get('init_adapter'),
                                'agent_prompt': 'prompts/agent.txt', 'doctor_prompt': 'prompts/user_simulator.txt', 'rollout': self.cfg['rollout']}

    def generate(self, case: dict, seed: int) -> dict:
        self.model.eval(); self.torch.manual_seed(seed)
        return self.ST.generate_trajectory(self.model, self.tok, case, self.doctor_system, self.agent_system, self.key, self.args)

    # Continuations reuse src/prompt_opt/harness.Env bound to THIS policy; rounds and Doctor model come from self.args (config).
    def _H(self):
        from src.prompt_opt import harness as H
        return H

    def agent_user_prompt(self, case, prefix, t): return self._H().Env.agent_user_prompt(self, case, prefix, t)

    def agent(self, user, seed): self.model.eval(); return self._H().Env.agent(self, user, seed)

    def natural(self, case, prefix, turn, seed): return self.agent(self.agent_user_prompt(case, prefix, turn), seed)

    def continue_from(self, case, prefix, turn, fixed_action, seed): self.model.eval(); return self._H().Env.continue_from(self, case, prefix, turn, fixed_action, seed)

    def sample_completions(self, case, prefix, turn, seeds) -> list[dict]:
        """G completions from EXACTLY one frozen prompt (normal policy prompt for h_t; no rubric). Each trace carries the
        prompt string / token ids, so identical prompt hashes can be asserted before the loss."""
        self.model.eval(); a = self.args
        prompt = self.ST.chat_prompt(self.tok, self.agent_system, self.agent_user_prompt(case, prefix, turn)); out = []
        for s in seeds:
            self.torch.manual_seed(s)
            tr = self.ST.generate_with_trace(self.model, self.tok, prompt, a.max_completion_length, a.generation_temperature, a.top_p, a.top_k)
            out.append({'stage': f'A{turn}', 'prompt_text': prompt, 'seed': s, **tr})
        return out

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
        st = self.policy.torch.load(p / 'trainer_state.pt', weights_only=False); self.optimizer.load_state_dict(st['optimizer']); self.steps = st['steps']
        self.policy.torch.set_rng_state(st['torch_rng'])
