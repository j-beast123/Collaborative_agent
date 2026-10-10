"""Reference generation leased only at a complete batch barrier; no extra GPU."""
import json
import time
from contextlib import contextmanager
from pathlib import Path
from baselines.common.reference import FrozenInitialReference
from baselines.common.rubrics import public_case,public_history
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.gpu_pool import assert_idle


class PooledInitialReference:
    def __init__(self,cfg,output,*,base_revision,initial_policy_sha256):
        self.local=FrozenInitialReference(cfg,output,base_revision=base_revision,initial_policy_sha256=initial_policy_sha256)
        self.identity={**self.local.identity,'execution':'idle_assistant_initial_adapter_swap_v1',
            'reference_physical_gpu':cfg['four_gpu']['assistant_physical_gpus'][0]}
        self.bound=None;self.actual_initial_hash=initial_policy_sha256
    def observe_initial(self,value):
        if self.actual_initial_hash is not None and self.actual_initial_hash!=value:raise RuntimeError('Reference initial identity differs')
        self.actual_initial_hash=value
    @contextmanager
    def at_barrier(self,executor,expected,pool):
        if self.bound is not None:raise RuntimeError('Reference already bound')
        assert_idle(pool);self.bound=(executor,expected,pool)
        try:yield
        finally:self.bound=None;assert_idle(pool)
    def response(self,case,prefix,turn,seed):
        if self.bound is None:raise RuntimeError('Reference only allowed after all five case rollouts finish')
        executor,expected,pool=self.bound;assert_idle(pool)
        if prefix[-1][2]!=turn:raise ValueError('History turn differs')
        if not self.actual_initial_hash:raise RuntimeError('Initial model hash must be measured before reference generation')
        request={'case':public_case(case),'history':public_history(prefix),'turn':turn,'seed':seed,'model_and_generation':{**self.identity,'measured_initial_policy_sha256':self.actual_initial_hash}}
        key=sha256_json(['initial_reference',request])
        def fetch():
            from baselines.common.four_gpu_worker import reference_response
            began=time.time();event={'status':'interrupted','cache_key':key,'input_sha256':sha256_json(request)}
            try:
                trace=executor.submit(reference_response,public_case(case),prefix,turn,seed,expected).result()
                text=trace['completion_text']
                if not isinstance(text,str) or not text.strip():raise ValueError('Empty reference response')
                counts={role:sum(len(s['completion_ids']) for s in trace.get('segments',[]) if s.get('role')==role)
                        for role in ('private_assessment','visible_reply')}
                value={'text':text,'private_tokens':counts['private_assessment'],'visible_tokens':counts['visible_reply'],
                    'cache_key':key,'input_sha256':sha256_json(request)}
                event={'status':'ok',**{k:v for k,v in value.items() if k!='text'}}
                return value
            except Exception as exc:event.update(status='failed',error=type(exc).__name__);raise
            finally:
                event.update(seconds=time.time()-began,doctor_calls=0,seed=seed,physical_gpu=self.identity['reference_physical_gpu'],
                    reference_adapter_sha256=self.actual_initial_hash,current_adapter_sha256=expected,
                    execution=self.identity['execution'])
                self.local.out.mkdir(parents=True,exist_ok=True)
                with (self.local.out/'reference_generation.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        value=self.local.cache.call('initial_reference',request,fetch)
        if value.get('cache_key')!=key or value.get('input_sha256')!=sha256_json(request):raise RuntimeError('Reference cache identity differs')
        return value
    def close(self):
        if self.bound is not None:raise RuntimeError('Reference still bound at close')
