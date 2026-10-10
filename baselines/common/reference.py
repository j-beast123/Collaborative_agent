"""One cached Agent-only reply from a separate, immutable INITIAL policy copy."""
import copy
import hashlib
import json
import time
from pathlib import Path
from src.online_rl.integrated.records import sha256_json
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.records import to_dialogue
from baselines.common.protocol import ROOT
from baselines.common.rubrics import public_case, public_history


def tensor_hash(model,names):
    params=dict(model.named_parameters());digest=hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode());digest.update(params[name].detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


class FrozenInitialReference:
    def __init__(self,cfg,output,*,base_revision,initial_policy_sha256,device='cuda:5',factory=None):
        self.cfg=copy.deepcopy(cfg);self.out=Path(output);self.device=device;self.factory=factory
        self.policy=None;self.names=None
        self.cache=CallCache(self.out/'cache/reference_responses.jsonl')
        from src.online_rl import agent_protocol as ap
        ro=cfg['rollout']
        prompt=ROOT/ro.get('agent_system_prompt_file','prompts/agent.txt')
        self.identity={'protocol':'initial_adapter_agent_only_reference_v1','policy':cfg['policy'],
            'base_model_revision':base_revision,'initial_policy_sha256':initial_policy_sha256,
            'agent_prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),
            'private_protocol_sha256':sha256_json(ap.IDENTITY),'rollout':ro,
            'runtime_source_sha256':hashlib.sha256((ROOT/'src/online_rl/runtime.py').read_bytes()).hexdigest(),
            'sampling_source_sha256':hashlib.sha256((ROOT/'src/online_rl/agent_batch.py').read_bytes()).hexdigest()}

    def _load(self):
        if self.policy is not None:return
        import torch,os
        from src.online_rl.runtime import SharedPolicy
        if os.environ.get('SHERPA_AGENT_REPLICAS','1')!='1':raise RuntimeError('Reference requires one isolated replica')
        # Model construction must not advance the training policy's RNG streams.
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self.policy=(self.factory(self.cfg,self.device) if self.factory else
                         SharedPolicy(self.cfg,'',device=self.device))
        model=self.policy.model
        self.names=[name for name,p in model.named_parameters() if p.requires_grad]
        if not self.names or tensor_hash(model,self.names)!=self.identity['initial_policy_sha256']:
            raise RuntimeError('Reference adapter is not the verified initial adapter')
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
        # No updater is constructed. This object has no current-checkpoint loading path.

    def response(self,case,prefix,turn,seed):
        case=public_case(case);history=public_history(prefix)
        if prefix[-1][2]!=turn:raise ValueError('Reference history/turn mismatch')
        request={'case':case,'history':history,'turn':turn,'seed':seed,'model_and_generation':self.identity}
        key=sha256_json(['initial_reference',request])
        def generate():
            started=time.time();before=None
            event={'status':'interrupted','cache_key':key,'input_sha256':sha256_json(request)}
            try:
                self._load();before=tensor_hash(self.policy.model,self.names)
                if before!=self.identity['initial_policy_sha256']:raise RuntimeError('Frozen reference changed')
                # One precommit action: private assessment + visible reply. No Doctor call, FINAL, or future rollout.
                trace=self.policy.agent_turn(case,to_dialogue(prefix),turn,seed)
                after=tensor_hash(self.policy.model,self.names)
                if after!=before:raise RuntimeError('Reference weights mutated during generation')
                text=trace['completion_text']
                if not isinstance(text,str) or not text.strip():raise ValueError('Empty reference reply')
                counts={r:sum(len(s['completion_ids']) for s in trace.get('segments',[]) if s.get('role')==r)
                        for r in ('private_assessment','visible_reply')}
                value={'text':text,'private_tokens':counts['private_assessment'],'visible_tokens':counts['visible_reply'],
                       'input_sha256':sha256_json(request),'cache_key':key}
                event={'status':'ok',**{k:v for k,v in value.items() if k!='text'}}
            except Exception as exc:
                event={'status':'failed','error':type(exc).__name__,'cache_key':key,'input_sha256':sha256_json(request)}
                raise
            finally:
                self.out.mkdir(parents=True,exist_ok=True)
                event.update(seconds=time.time()-started,seed=seed,device=self.device,doctor_calls=0,
                             reference_adapter_sha256=before,physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'))
                with (self.out/'reference_generation.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            return value
        result=self.cache.call('initial_reference',request,generate)
        if result.get('cache_key')!=key or result.get('input_sha256')!=sha256_json(request):raise RuntimeError('Reference cache identity differs')
        return result

    def close(self):
        if self.policy is not None and hasattr(self.policy,'_batcher'):self.policy._batcher.close()
        self.policy=None


def _worker(connection,cfg,output,revision,initial_hash,physical_gpu):
    import os,fcntl
    os.environ['CUDA_VISIBLE_DEVICES']=str(physical_gpu)
    os.environ['SHERPA_AGENT_REPLICAS']='1'
    runtime_out=Path(output)/'reference_runtime';runtime_out.mkdir(parents=True,exist_ok=True)
    os.environ['SHERPA_EXEC_LOG']=str(runtime_out/'agent_execution.jsonl')
    os.environ['API_USAGE_LOG']=str(runtime_out/'api_usage.jsonl')
    from src.common.live_gpu import configure,note
    configure(runtime_out,physical_gpu);note(phase='frozen_initial_reference')
    lock=open(f'/tmp/sherpa_tree_physical_gpu{physical_gpu}.lock','a')
    ref=None
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ref=FrozenInitialReference(cfg,output,base_revision=revision,initial_policy_sha256=initial_hash,device='cuda:0')
        ref.cache=CallCache(None)  # parent is the sole durable-cache writer
        while True:
            message=connection.recv()
            if message is None:break
            try:connection.send({'ok':True,'value':ref.response(*message)})
            except Exception as exc:connection.send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
    except Exception as exc:
        connection.send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
    finally:
        if ref:ref.close()
        lock.close();connection.close()


class ReferenceProcess:
    """Dedicated physical GPU outside the unchanged five training GPUs; lazy load."""
    def __init__(self,cfg,output,*,base_revision,initial_policy_sha256,physical_gpu):
        self.local=FrozenInitialReference(cfg,output,base_revision=base_revision,initial_policy_sha256=initial_policy_sha256)
        self.identity=self.local.identity;self.physical_gpu=str(physical_gpu);self.process=None;self.connection=None
    def response(self,case,prefix,turn,seed):
        request={'case':public_case(case),'history':public_history(prefix),'turn':turn,'seed':seed,'model_and_generation':self.identity}
        key=sha256_json(['initial_reference',request])
        def fetch():
            if self.process is None:
                import multiprocessing
                ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe()
                self.process=ctx.Process(target=_worker,args=(child,self.local.cfg,str(self.local.out),
                    self.identity['base_model_revision'],self.identity['initial_policy_sha256'],self.physical_gpu),daemon=True)
                self.process.start();child.close();self.connection=parent
            self.connection.send((public_case(case),prefix,turn,seed))
            if not self.connection.poll(1800):raise TimeoutError('Reference response timed out; resume from cache')
            result=self.connection.recv()
            if not result['ok']:raise RuntimeError('Frozen reference failed: '+result['error'])
            value=result['value']
            if value.get('cache_key')!=key or value.get('input_sha256')!=sha256_json(request):raise RuntimeError('Reference IPC identity differs')
            return value
        value=self.local.cache.call('initial_reference',request,fetch)
        if value.get('cache_key')!=key or value.get('input_sha256')!=sha256_json(request):raise RuntimeError('Reference cache identity differs')
        return value
    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try:self.connection.send(None)
                except (BrokenPipeError,EOFError):pass
                self.process.join(timeout=30)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=10)
            self.connection.close();self.process=None
