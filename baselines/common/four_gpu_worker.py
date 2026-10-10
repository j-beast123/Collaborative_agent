"""Two Assistant actors; Doctor HTTP endpoints live on two separate GPUs."""
import copy
import os
from src.online_rl.tree import five_gpu_worker as original
from baselines.common.reference import tensor_hash

_SWITCH=None


class InitialAdapterSwitch:
    """Immutable CPU initial tensors; time-multiplex one frozen base at an idle barrier."""
    def __init__(self,policy,expected_initial):
        self.policy=policy
        self.names=[n for n,p in policy.model.named_parameters() if p.requires_grad]
        if not self.names or tensor_hash(policy.model,self.names)!=expected_initial:raise RuntimeError('Initial reference hash differs')
        self.initial={n:p.detach().cpu().clone() for n,p in policy.model.named_parameters() if n in self.names}
        self.expected_initial=expected_initial
    def response(self,case,prefix,turn,seed,expected_current):
        from src.online_rl.tree.records import to_dialogue
        policy=self.policy;model=policy.model;params=dict(model.named_parameters())
        if tensor_hash(model,self.names)!=expected_current:raise RuntimeError('Reference worker current policy differs')
        current={n:params[n].detach().cpu().clone() for n in self.names}
        flags={n:p.requires_grad for n,p in params.items()};training=model.training
        actor=policy.pool_actor;old_expected=actor.expected;old_generate=policy._generate
        if old_expected!=expected_current:raise RuntimeError('Reference actor not prepared')
        import torch
        with actor.changing_policy():
            try:
                with policy.gen_lock,torch.no_grad():
                    for n in self.names:params[n].copy_(self.initial[n])
                    for p in params.values():p.requires_grad_(False)
                    policy.mark_dirty();model.eval()
                if tensor_hash(model,self.names)!=self.expected_initial:raise RuntimeError('Initial adapter installation failed')
                # Bypass pooled dispatch: only this idle actor generates; no Doctor call.
                policy._generate=actor.handlers['agent']
                trace=policy.agent_turn(case,to_dialogue(prefix),turn,seed)
                if tensor_hash(model,self.names)!=self.expected_initial:raise RuntimeError('Frozen initial reference mutated')
                return trace
            finally:
                policy._generate=old_generate
                with policy.gen_lock,torch.no_grad():
                    for n in self.names:params[n].copy_(current[n])
                    for n,p in params.items():p.requires_grad_(flags[n])
                    policy.mark_dirty();model.train(training)
                if tensor_hash(model,self.names)!=expected_current:raise RuntimeError('Current adapter restoration failed')
                actor.expected=old_expected


def initialize(cfg,worker,key,pool_spec,existing_policy=None):
    global _SWITCH
    local=copy.deepcopy(cfg)
    if existing_policy is None:
        os.environ['CUDA_VISIBLE_DEVICES']=cfg['four_gpu']['assistant_physical_gpus'][worker]
        from baselines.common.policy_loader import load_policy
        local['rollout']['doctor_backend']['endpoint']=cfg['five_gpu']['doctor_urls'][worker]
        policy=load_policy(local,key,device='cuda:0')
    else:policy=existing_policy
    # Passing an existing policy bypasses the original five-device initializer.
    original.initialize(local,worker,key,pool_spec,existing_policy=policy)
    problems=policy.doctor.verify()['problems']
    if problems:raise RuntimeError('Doctor identity: '+'; '.join(problems))
    if worker==0 and cfg['comparison_method']['method']=='online_rubrics':
        expected=cfg['comparison_method'].get('initial_policy_sha256')
        if expected is None:
            import json
            from pathlib import Path
            expected=json.loads((Path(cfg['output_dir'])/'initial_policy.json').read_text())['initial_policy_sha256']
        _SWITCH=InitialAdapterSwitch(policy,expected)


def prepare(checkpoint,expected):return original.prepare(checkpoint,expected)
def run_many(jobs,checkpoint,expected):return original.run_many(jobs,checkpoint,expected)


def reference_response(case,prefix,turn,seed,expected):
    if _SWITCH is None:raise RuntimeError('Initial reference not initialized on Assistant worker 0')
    from src.common.live_gpu import activity,note
    note(phase='initial_reference')
    with activity('initial_reference_adapter_swap_and_generation'):
        return _SWITCH.response(case,prefix,turn,seed,expected)
