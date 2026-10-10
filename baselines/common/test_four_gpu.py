"""CPU-only scheduler/barrier/adapter/registry checks; no model, API or CUDA execution."""
import copy
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager,ExitStack
from concurrent.futures import Future,ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from baselines.common.test_methods import TinyModel,fixture,FakeBackend
from baselines.common.four_gpu import batch_slots
from baselines.common.four_gpu_worker import InitialAdapterSwitch
from baselines.common.pooled_reference import PooledInitialReference
from baselines.common.reference import tensor_hash
from baselines.common.rubrics import RubricEngine
from baselines.common.rubric_launch import configuration,RubricHook
from baselines.common.runner import isolated_runner
from src.online_rl.integrated.continuation import adapter_sha256
from src.online_rl.integrated.grpo import IntegratedGRPOUpdater
from src.online_rl.tree.team import node_credit
from src.online_rl.tree.five_gpu import batch_slots as old_slots


class Actor:
    def __init__(self,expected):self.expected=expected;self.handlers={'agent':lambda *a:None};self.active=0
    @contextmanager
    def changing_policy(self):
        if self.active:raise RuntimeError('active actor')
        self.expected=None;yield
    def close(self):pass


class Executor:
    def __init__(self,*a,**kw):pass
    def submit(self,fn,*a,**kw):
        future=Future()
        try:future.set_result(fn(*a,**kw))
        except BaseException as e:future.set_exception(e)
        return future
    def shutdown(self,**kw):pass


class FourGPUTests(unittest.TestCase):
    def test_two_workers_keep_24_original_five_case_batches(self):
        seen=[]
        for cursor in range(0,120,5):
            slots=batch_slots(cursor,120);original=old_slots(cursor,120,30,5,1)
            self.assertEqual([s['index'] for s in slots],[s['index'] for s in original])
            self.assertEqual([sum(s['worker']==w for s in slots) for w in range(2)],[3,2])
            seen.extend(s['index'] for s in slots)
        self.assertEqual(seen,list(range(120)))
        configs=[json.loads(Path(f'baselines/{m}/method.json').read_text()) for m in ('rar','online_rubrics')]
        self.assertEqual(configs[0]['doctor_physical_gpus']+configs[0]['assistant_physical_gpus'],['0','1','2','3'])
        self.assertEqual(configs[1]['doctor_physical_gpus']+configs[1]['assistant_physical_gpus'],['4','5','6','7'])
        self.assertIsNone(configs[1]['reference_physical_gpu'])

    def test_adapter_swap_restores_current_even_when_generation_fails(self):
        model=TinyModel();initial=adapter_sha256(model)
        policy=SimpleNamespace(model=model,gen_lock=threading.RLock(),mark_dirty=lambda:None,_generate=lambda *a:'pooled')
        switch=InitialAdapterSwitch(policy,initial)
        with torch.no_grad():model.adapter.add_(2.)
        current=adapter_sha256(model);policy.pool_actor=Actor(current);calls=[]
        def generate(case,prefix,turn,seed):
            self.assertEqual(tensor_hash(model,switch.names),initial)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertIsNone(policy.pool_actor.expected)
            self.assertIs(policy._generate,policy.pool_actor.handlers['agent'])
            calls.append(seed)
            if seed==2:raise ValueError('synthetic failure')
            return {'completion_text':'initial only'}
        policy.agent_turn=generate;old_generate=policy._generate
        args=({'case_id':'test'},(('doctor','D1',1,'history'),),1)
        self.assertEqual(switch.response(*args,1,current)['completion_text'],'initial only')
        with self.assertRaises(ValueError):switch.response(*args,2,current)
        self.assertEqual(adapter_sha256(model),current)
        self.assertIs(policy._generate,old_generate);self.assertEqual(policy.pool_actor.expected,current)
        self.assertTrue(model.training);self.assertTrue(model.adapter.requires_grad)
        policy.pool_actor.active=1
        with self.assertRaises(RuntimeError):switch.response(*args,3,current)
        self.assertEqual(calls,[1,2])

    def test_pooled_reference_requires_idle_barrier_and_reuses_cache(self):
        model=TinyModel();trees,receipt,cases=fixture(model)
        with tempfile.TemporaryDirectory() as temp:
            cfg=configuration('online_rubrics',temp,receipt,None,adapter_sha256(model),'synthetic')
            ref=PooledInitialReference(cfg,temp,base_revision='synthetic',initial_policy_sha256=adapter_sha256(model))
            args=(cases['case0'],trees[0].nodes[0].prefix,1,11)
            with self.assertRaises(RuntimeError):ref.response(*args)
            pool={'lock':threading.RLock(),'leases':{}}
            trace={'completion_text':'initial','segments':[{'role':'private_assessment','completion_ids':[1,2]}]}
            with patch('baselines.common.four_gpu_worker.reference_response',return_value=trace) as generate:
                with ref.at_barrier(Executor(),'current',pool):
                    first=ref.response(*args);self.assertEqual(ref.response(*args),first)
                self.assertEqual(generate.call_count,1)
                self.assertNotIn('ground_truth',generate.call_args.args[0])
            self.assertIsNone(ref.bound)
            pool['leases']['x']={}
            with self.assertRaises(RuntimeError):
                with ref.at_barrier(Executor(),'current',pool):pass

    def test_concurrent_methods_generate_one_shared_initial_rubric(self):
        trees,_,cases=fixture(TinyModel());backend=FakeBackend()
        with tempfile.TemporaryDirectory() as temp:
            rar=RubricEngine('rar',backend,backend,Path(temp)/'rar',weights={'w_outcome':0.,'w_rubric':1.},initial_registry=Path(temp)/'registry')
            from baselines.common.test_methods import FakeReference
            online=RubricEngine('online_rubrics',backend,backend,Path(temp)/'online',weights={'w_outcome':0.,'w_rubric':1.},
                reference=FakeReference(),initial_registry=Path(temp)/'registry')
            with ThreadPoolExecutor(2) as ex:
                pending=[ex.submit(e.initial,cases['case0'],trees[0].nodes[0]) for e in (rar,online)]
                self.assertEqual(pending[0].result(),pending[1].result())
            self.assertEqual(len(backend.calls),1)
            self.assertEqual(len(rar.initial_cache.store),1);self.assertEqual(len(online.initial_cache.store),1)

    def test_real_four_gpu_runner_one_cpu_mock_batch_and_resume(self):
        model=TinyModel();trees,receipt,cases=fixture(model);initial=adapter_sha256(model)
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            backend=FakeBackend();engine=RubricEngine('rar',backend,backend,Path(temp)/'rewards',weights={'w_outcome':0.,'w_rubric':1.})
            cfg=configuration('rar',temp,receipt,engine,initial,'synthetic')
            cfg['schedule']['cases']=5;cfg['five_gpu']['eval_cases']=3  # synthetic final checkpoint, no clinical data
            hook=RubricHook(cfg,receipt,engine,cases);runner=isolated_runner(hook)
            policies=[];evaluated=[]
            def policy_factory(*a,**k):
                p=SimpleNamespace(model=TinyModel(),torch=torch)
                def generate(case,seed):
                    evaluated.append(seed)
                    return {'dialogue':[],'final_option':'A' if seed!=1 else '', 'num_discussion_rounds':2}
                p.generate=generate;policies.append(p);return p
            def initialize(cfg,worker,key,pool,existing_policy):existing_policy.pool_actor=Actor(None)
            def prepare(checkpoint,expected):
                policies[-1].pool_actor.expected=expected
                self.assertEqual(adapter_sha256(policies[-1].model),expected)
            def records(jobs,checkpoint,expected):
                self.assertEqual(adapter_sha256(policies[-1].model),expected)
                return [{'case_id':trees[j['index']].case_id,'case_cursor':j['index'],'tree':asdict(trees[j['index']]),
                    'policy_sha256':expected,'prompt_sha256':cfg['rubric']['prompt_sha256'],
                    'agent_targets':node_credit(1.,0.)(trees[j['index']],{},{})} for j in jobs]
            def save(updater,path,proof):
                Path(path).mkdir(parents=True);(Path(path)/'COMPLETE.json').write_text(json.dumps(proof))
                (Path(path)/'cpu_adapter.json').write_text(json.dumps(updater.policy.model.adapter.detach().tolist()))
            def load(updater,path):
                with torch.no_grad():updater.policy.model.adapter.copy_(torch.tensor(json.loads((Path(path)/'cpu_adapter.json').read_text())))
            # Only hardware/model/checkpoint I/O is mocked; actual run loop, reward
            # hook, original GRPO update and durable batch receipts execute on CPU.
            bindings=[('scripts.online_rl.run_tree.load_key',lambda:'synthetic'),('src.online_rl.runtime.SharedPolicy',policy_factory),
                ('src.online_rl.integrated.grpo.make_updater',lambda policy,cfg:IntegratedGRPOUpdater(policy,cfg)),
                ('src.online_rl.cases.load_cases',lambda path:list(cases.values())[:3] if str(path).endswith('/test') else list(cases.values()))]
            for path,value in bindings:stack.enter_context(patch(path,value))
            for name,value in [('require_assistant_cuda',lambda:None),('preflight',lambda cfg:[]),('training_order',lambda *a:list(cases)),
                ('save_checkpoint',save),('load_checkpoint',load),('ProcessPoolExecutor',Executor),('ThreadPoolExecutor',Executor),
                ('create_pool',lambda *a:{'lock':threading.RLock(),'leases':{},'directory':str(Path(temp)/'pool')})]:
                stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch.object(runner.multiprocessing,'get_context',lambda *a:SimpleNamespace(Manager=lambda:SimpleNamespace(shutdown=lambda:None))))
            for name,value in [('initialize',initialize),('prepare',prepare),('run_many',records)]:stack.enter_context(patch.object(runner.worker,name,value))
            runner.run(cfg,max_batches=1)
            state=json.loads((Path(temp)/'run_state.json').read_text())
            self.assertEqual(state['case_cursor'],5);self.assertEqual(state['batch_cursor'],1)
            self.assertNotEqual(state['policy_sha256'],initial)
            proof=json.loads((Path(state['checkpoint'])/'COMPLETE.json').read_text())
            self.assertEqual(proof['update']['optimizer_steps'],1)
            # Resume without another batch: loads the committed current adapter.
            runner.run(cfg,max_batches=0)
            self.assertEqual(adapter_sha256(policies[-1].model),state['policy_sha256'])
            self.assertFalse((Path(temp)/'evaluation').exists())
            from baselines.common.final_eval import evaluate_completed_run
            summary=evaluate_completed_run(runner,cfg,method_identity=hook.fingerprint)
            self.assertEqual(summary['accuracy'],2/3)
            evaluate_completed_run(runner,cfg,method_identity=hook.fingerprint)
            self.assertEqual(sorted(evaluated),[0,1,2])


if __name__=='__main__':unittest.main()
