"""Frozen local Ollama policy for prompt-only pilots; no synthetic token/logprob traces."""
from argparse import Namespace
from pathlib import Path
import json
import urllib.request
from src.online_rl.integrated.records import sha256_json

class OllamaPolicy:
    def __init__(self,cfg,key):
        root=Path(__file__).resolve().parents[3]
        self.cfg,self.key=cfg,key; self.trainable=False
        self.endpoint=cfg['pilot']['ollama_url'].rstrip('/');self.model_name=cfg['pilot']['ollama_model']
        self.judgment_system=(root/'prompts/local_epistemic_judgment.txt').read_text();self.agent_system=(root/'prompts/agent.txt').read_text();self.doctor_system=(root/'prompts/user_simulator.txt').read_text()
        ro=cfg['rollout'];self.args=Namespace(rounds=ro['rounds'],doctor_model=ro['doctor_model'],max_tokens_doctor=ro['max_tokens_doctor'])
        with urllib.request.urlopen(self.endpoint+'/api/tags',timeout=10) as response:
            models=json.load(response)['models']
        model=next(m for m in models if m['name']==self.model_name)
        self.digest=model['digest'];self.details=model['details']

    def identity(self):
        return {'backend':'ollama','model':self.model_name,'digest':self.digest,'details':self.details,
                'agent_system':self.agent_system,'rollout':self.cfg['rollout'],'thinking':False,'chat_template':'qwen3_raw_disabled_thinking_v1','belief':self.cfg.get('belief'),'judgment_system':self.judgment_system}

    def policy_sha(self): return sha256_json(self.identity())

    def agent_user_prompt(self,case,prefix,turn):
        if self.cfg.get('belief',{}).get('enabled'):
            from src.online_rl.tree.belief import prepare
            return prepare(self,case,prefix,turn)[0]
        from src.prompt_opt.harness import Env
        return Env.agent_user_prompt(self,case,prefix,turn)

    def agent(self,user,seed,*,system=None,greedy=False,max_tokens=None):
        ro=self.cfg['rollout']
        prompt='<|im_start|>system\n'+(system or self.agent_system)+'<|im_end|>\n<|im_start|>user\n'+user+'\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        payload={'model':self.model_name,'stream':False,'raw':True,'prompt':prompt,
                 'options':{'seed':seed,'temperature':0 if greedy else ro['temperature'],'top_p':ro['top_p'],
                            'num_predict':max_tokens or ro['max_completion_length'],'num_ctx':16384,'num_gpu':-1}}
        req=urllib.request.Request(self.endpoint+'/api/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=240) as response:
            result=json.load(response)
        if result.get('done_reason')=='length':
            raise ValueError('Qwen reply truncated: raise max_completion_length')
        text=result['response']
        if not text.strip(): raise ValueError('Qwen returned no reply')
        return text

class OllamaAgent:
    def __init__(self,policy):self.policy=policy
    def sample(self,case,prefix,turn,seed):
        packet=None
        if getattr(self.policy, 'cfg', {}).get('belief',{}).get('enabled'):
            from src.online_rl.tree.belief import prepare
            prompt,packet=prepare(self.policy,case,prefix,turn)
        else:
            prompt=self.policy.agent_user_prompt(case,prefix,turn)
        return {'prompt_text':prompt,'completion_text':self.policy.agent(prompt,seed),'seed':seed,
                'pre_response_belief':packet,'trace_semantics':'text_only_frozen_policy; no training logprobs'}
