"""Resumable original-base-model dialogue evaluation: no LoRA, belief, rubric or learning."""
import json
import time
from contextlib import contextmanager
from pathlib import Path
from src.online_rl.integrated.records import atomic_write_json, sha256_json, sha256_text
from src.online_rl.environment import final_outcome


@contextmanager
def original_base_policy(policy):
    """Borrow the one GPU4 model without changing the learned adapter or RNG state."""
    torch=policy.torch
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mode=policy.model.training;cache=policy.model.config.use_cache;precommit=policy.precommit
    try:
        policy.precommit=False
        policy.model.eval()
        # PEFT disables every adapter in this context; frozen base weights stay original.
        with policy.model.disable_adapter():yield
    finally:
        policy.precommit=precommit;policy.model.train(mode);policy.model.config.use_cache=cache
        torch.set_rng_state(cpu)
        if gpu is not None:torch.cuda.set_rng_state_all(gpu)


class BaseEvaluation:
    def __init__(self,root,cases,policy,cfg,*,gpu=4):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.cases=cases;self.policy=policy;self.cfg=cfg;self.gpu=gpu
        ro=cfg['rollout']
        self.identity={'mode':'original_base_no_precommit_v1','base_model':cfg['policy']['base_model'],
            'base_revision':getattr(policy.model.config,'_commit_hash',None),
            'base_weights':'original_frozen_base','adapter':'disabled','agent_protocol':'plain_dialogue_v1',
            'rubrics':False,'optimizer_steps':0,
            'agent_prompt_sha256':sha256_text(policy.agent_system),'doctor_protocol_sha256':policy.doctor.protocol,
            'rollout':{k:ro[k] for k in ('rounds','max_completion_length','temperature','top_p','top_k','max_tokens_doctor')},
            'halting':ro['halting'],'seed':cfg['five_gpu']['eval_seed'],
            'data_sha256':cfg['five_gpu']['eval_sha256'],'case_ids':[c['case_id'] for c in cases]}
        manifest=self.root/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=self.identity:
            raise RuntimeError('BASE evaluation identity changed; use a new BASE output directory')
        atomic_write_json(manifest,self.identity)
        self.digest=sha256_json(self.identity)
        self._summarize()

    def _rows(self):
        rows=[json.loads(p.read_text()) for p in sorted((self.root/'cases').glob('case_*.json'))]
        for row in rows:
            if row['evaluation_sha256']!=self.digest or row['case_id']!=self.cases[row['index']]['case_id']:
                raise RuntimeError('BASE result identity differs')
        return rows

    def _summarize(self):
        rows=self._rows();n=len(rows)
        summary={'mode':self.identity['mode'],'dataset_cases':len(self.cases),'evaluated':n,
                 'correct':sum(r['outcome']==1 for r in rows),'unparsed':sum(r['outcome'] is None for r in rows),
                 'accuracy':sum(r['outcome']==1 for r in rows)/n if n else None,
                 'mean_agent_turns':sum(r['num_discussion_rounds'] for r in rows)/n if n else None,
                 'mean_episode_seconds':sum(r['seconds'] for r in rows)/n if n else None,
                 'status':'complete' if n==len(self.cases) else 'pending',
                 'evaluation_sha256':self.digest,'optimizer_steps':0,'adapter':'disabled','precommit':False}
        atomic_write_json(self.root/'summary.json',summary)
        return summary

    def step(self):
        for i,case in enumerate(self.cases):
            path=self.root/'cases'/f'case_{i:06d}.json'
            if path.exists():continue
            start=time.monotonic();seed=self.identity['seed']+i
            print(json.dumps({'stage':'BASE_eval','gpu':self.gpu,'index':i,'case_id':case['case_id'],'total':len(self.cases)}),flush=True)
            from src.common.live_gpu import activity, note
            note(phase='BASE_eval',gpu=self.gpu,case_index=i,case_id=case['case_id'])
            with activity('BASE_eval'), original_base_policy(self.policy):
                episode=self.policy.generate(case,seed)
            result={'index':i,'case_id':case['case_id'],'seed':seed,'evaluation_sha256':self.digest,
                    **episode,'outcome':final_outcome(case,episode['final_option']),'seconds':time.monotonic()-start}
            atomic_write_json(path,result)
            summary=self._summarize()
            print(json.dumps({'stage':'BASE_case_done',**summary}),flush=True)
            return True
        self._summarize();return False
