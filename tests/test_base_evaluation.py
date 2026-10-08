"""BASE episodes disable LoRA and precommit and preserve training state on failure."""
import json
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch
from src.online_rl.tree.base_evaluation import BaseEvaluation,original_base_policy


def policy():
    model=SimpleNamespace(training=True,config=SimpleNamespace(use_cache=False,_commit_hash='revision'),adapter_on=True)
    model.eval=lambda:setattr(model,'training',False)
    model.train=lambda v:setattr(model,'training',v)
    @contextmanager
    def disabled():
        model.adapter_on=False
        try:yield
        finally:model.adapter_on=True
    model.disable_adapter=disabled
    p=SimpleNamespace(torch=torch,model=model,precommit=True,agent_system='agent',doctor=SimpleNamespace(protocol='doctor'))
    def generate(case,seed):
        assert not p.model.adapter_on and not p.precommit
        torch.manual_seed(seed);torch.rand(3)
        return {'dialogue':[],'final_option':'A','num_discussion_rounds':3}
    p.generate=generate
    return p


def test_base_context_restores_rng_mode_and_precommit_after_error():
    p=policy();rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with original_base_policy(p):
            assert not p.model.adapter_on and not p.precommit
            torch.rand(20);p.model.config.use_cache=True
            raise RuntimeError('episode failed')
    assert p.model.adapter_on and p.precommit and p.model.training and not p.model.config.use_cache
    assert torch.equal(rng,torch.get_rng_state())


def test_base_results_resume_without_repeating_completed_cases(tmp_path):
    p=policy();cases=[{'case_id':str(i),'ground_truth':{'correct_option':'A'}} for i in range(3)]
    cfg={'policy':{'base_model':'original'},'rollout':{'rounds':5,'max_completion_length':400,'temperature':.7,
        'top_p':.9,'top_k':None,'max_tokens_doctor':256,'halting':{'min_agent_turns':3}},
        'five_gpu':{'eval_seed':0,'eval_sha256':'data'}}
    b=BaseEvaluation(tmp_path,cases,p,cfg);assert b.step()
    b=BaseEvaluation(tmp_path,cases,p,cfg)
    while b.step():pass
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['evaluated']==3 and summary['correct']==3 and summary['status']=='complete'
    assert summary['optimizer_steps']==0 and not summary['precommit'] and summary['adapter']=='disabled'
    assert p.model.adapter_on and p.precommit
