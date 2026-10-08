import json
from pathlib import Path
from src.online_rl.cases import load_cases
from src.online_rl.guards import question_sha256
from src.online_rl.tree.ollama_policy import OllamaAgent


def test_pilot_has_four_optimization_one_heldout_fixed_agent():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/online_rl/tree_pilot5.json').read_text())
    train=load_cases(root/cfg['data']['train']);heldout=load_cases(root/cfg['data']['heldout'])
    assert len(train)==4 and len(heldout)==1
    assert cfg['pilot']['prompt_only'] and cfg['schedule']['cases']==4
    assert cfg['rollout']['rounds']==5 and cfg['rollout']['branching']==2
    assert cfg['pilot']['prompt_model']=='openai/gpt-5.6-sol'
    assert not {question_sha256(c['question']) for c in train}&{question_sha256(c['question']) for c in heldout}


def test_ollama_agent_has_honest_text_only_trace():
    class Policy:
        def agent_user_prompt(self,*args):return 'prefix'
        def agent(self,user,seed):return f'{user}:{seed}'
    trace=OllamaAgent(Policy()).sample({},[],1,42)
    assert trace['completion_text']=='prefix:42'
    assert trace['seed']==42
    assert 'completion_ids' not in trace and 'old_policy_logprobs' not in trace


def test_ollama_uses_explicit_qwen_template(monkeypatch):
    from src.online_rl.tree.ollama_policy import OllamaPolicy
    import io
    import urllib.request
    seen=[]
    def urlopen(request,timeout):
        seen.append(json.loads(request.data))
        assert request.full_url.endswith('/api/generate')
        return io.BytesIO(json.dumps({'response':'Ready.','done_reason':'stop'}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',urlopen)
    policy=OllamaPolicy.__new__(OllamaPolicy)
    policy.endpoint='http://local';policy.model_name='qwen';policy.agent_system='system'
    policy.cfg={'rollout':{'temperature':.7,'top_p':.9,'max_completion_length':2048}}
    assert policy.agent('question',42)=='Ready.'
    assert seen[0]['raw'] is True
    assert '<|im_start|>system' in seen[0]['prompt']
    assert seen[0]['prompt'].endswith('<think>\n\n</think>\n\n')
