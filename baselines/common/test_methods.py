"""Synthetic CPU integration; no provider, Qwen, Doctor, pilot or clinical evaluation."""
import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.judges.llm import ChatResult, ChatRequest
from src.online_rl.integrated import grpo
from src.online_rl.integrated.continuation import adapter_sha256
from src.online_rl.integrated.records import atomic_write_json
from src.online_rl.tree.records import AgentNode, Leaf, Tree
from src.online_rl.tree.team import node_credit
from baselines.common.rubrics import RubricEngine, criteria, public_history
from baselines.common.rubric_launch import configuration, RubricHook, bind
from baselines.common.runner import isolated_runner
from baselines.common.final_eval import evaluate_completed_run
from baselines.common.costs import summarize
from baselines.common.backend import AuditedBackend
from baselines.common.reference import FrozenInitialReference, tensor_hash
from baselines.common.protocol import mixture


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.adapter=torch.nn.Parameter(torch.zeros(8))
        self.config=SimpleNamespace(use_cache=True);self.disabled=False
    @contextmanager
    def disable_adapter(self):
        self.disabled=True
        try:yield
        finally:self.disabled=False
    def forward(self,input_ids,attention_mask,use_cache,logits_to_keep):
        logits=self.adapter*(0 if self.disabled else 1)
        return SimpleNamespace(logits=logits.reshape(1,1,-1).expand(1,logits_to_keep,-1))


def fixture(model):
    policy=adapter_sha256(model);trees=[]
    for case in range(5):
        nodes=[];leaves=[]
        for sibling in range(4):
            nid=str(sibling)
            trace={'agent_protocol':'precommit_v1','state_prompt_hash':'same',
                'segments':[{'prompt_input_ids':[1],'completion_ids':[sibling+1]},
                            {'prompt_input_ids':[1,sibling+1],'completion_ids':[sibling+1]}],
                'private_assessment':'SECRET_PRIVATE'}
            trace['behavior_logprobs']=grpo.behavior_logprobs(model,[trace],16)[0]
            dialogue=(('doctor','D1',1,'Synthetic physician'),('agent','A1',1,f'reply{sibling}'))
            nodes.append(AgentNode(nid,None,f'case{case}',1,dialogue[:1],f'reply{sibling}',trace,policy,0,'stop',True,None))
            leaves.append(Leaf(nid,f'case{case}',(nid,),dialogue,'A','A',1.,'stop'))
        trees.append(Tree(f'case{case}',policy,4,4,tuple(nodes),tuple(leaves)))
    receipt={'all_verified':True,'train_case_order':[t.case_id for t in trees],
        'files':{k:{'actual_sha256':'synthetic_'+k,'path':'/synthetic/'+k} for k in ('train','train_order','test')}}
    cases={t.case_id:{'case_id':t.case_id,'question':'synthetic','options':{'A':'answer'},
        'ground_truth':{'correct_option':'A','correct_text':'TRAIN_GOLD_SECRET'}} for t in trees}
    return trees,receipt,cases


class FakeBackend:
    def __init__(self,role='both'):self.calls=[];self.role=role
    def identity(self):return {'model':'synthetic_'+self.role,'revision':'test'}
    def complete(self,request):
        payload=json.loads(request.user);self.calls.append((request.prompt_version,payload))
        if request.prompt_version.endswith('initial_rubric'):
            value={'criteria':[{'text':f'criterion{i}','category':'Essential'} for i in range(7)]}
        elif request.prompt_version.endswith('pairwise_extract'):
            value={'differences':'synthetic pair evidence','criteria':[{'text':'undesirable behavior','weight':-2}]}
        elif request.prompt_version.endswith('deduplicate'):value={'keep_ids':['e0']}
        else:
            marker=int(payload['target_response'][-1])
            value={'grades':[{'id':r['id'],'satisfied':int(r['id'][1:])<=marker if r['id'].startswith('c') else marker==0} for r in payload['rubric']]}
        return ChatResult('ok',json.dumps(value),'synthetic','synthetic','test',{},0.,1)


class FakeReference:
    identity={'initial_policy_sha256':'initial_ONLY','revision':'synthetic'}
    def __init__(self):self.calls=[]
    def response(self,case,prefix,turn,seed):
        self.calls.append((case,prefix,turn,seed))
        return {'text':'INITIAL_ONLY_RESPONSE','cache_key':'synthetic','input_sha256':'synthetic'}


class MethodTests(unittest.TestCase):
    def test_approved_doctor_pin_covers_opening_assets_and_resume(self):
        from baselines.common.protocol import apply_common,settings
        from baselines.common.doctor_assets import EXPECTED
        from src.online_rl.tree.experiment import load_config
        original=load_config('configs/experiments/reward_prompt_matrix/reward_sweep/r0_o1_outcome_only.json')
        before=copy.deepcopy(original);updated=apply_common(original)
        expected='88c2e99b4a44ef8a4c855315927ae99f6ea77a06f4ec24ad2d08929f818f5f00'
        self.assertEqual(original,before)
        self.assertEqual(EXPECTED,expected)
        self.assertEqual(updated['rollout']['doctor_backend']['model_digest'],expected)
        self.assertEqual(updated['five_gpu']['opening_identity']['model_digest'],expected)
        self.assertNotEqual(original['five_gpu']['opening_identity'],updated['five_gpu']['opening_identity'])
        model=TinyModel();_,receipt,_=fixture(model)
        with tempfile.TemporaryDirectory() as temp:
            cfg=configuration('rar',temp,receipt,None,adapter_sha256(model),'synthetic')
            bind(cfg)
            changed=copy.deepcopy(cfg)
            changed['comparison_method']['doctor_identity']['model_digest']=before['rollout']['doctor_backend']['model_digest']
            with self.assertRaises(RuntimeError):bind(changed,create=False)
        for method in ('rar','online_rubrics'):
            spec=json.loads(Path(f'baselines/{method}/method.json').read_text())
            self.assertEqual(spec['doctor_digest'],expected)
            for key in ('output_dir','initial_rubric_source','initial_rubric_registry'):
                self.assertIn(settings()['doctor']['cache_namespace'],spec[key])

    def test_existing_doctor_listener_is_read_only_rejected(self):
        from baselines.common.doctors import assert_endpoints_unused
        header='header\n'
        row='0: 0100007F:2DD3 00000000:0000 0A 0:0 0:0 0 0 0 0\n'  # 11731
        with tempfile.TemporaryDirectory() as temp,patch.object(Path,'read_text',return_value=header+row),patch('os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'11731.*untouched'):
                assert_endpoints_unused({'output_dir':temp,'five_gpu':{'doctor_urls':['http://127.0.0.1:11731']}})
            kill.assert_not_called()

    def test_existing_doctor_pid_is_not_adopted_or_stopped(self):
        from baselines.common.doctors import assert_endpoints_unused
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'doctor_servers';root.mkdir();(root/'gpu0.pid').write_text(str(os.getpid()))
            native=Path.read_text
            def read(path,*args,**kwargs):return 'header\n' if str(path).startswith('/proc/net/') else native(path,*args,**kwargs)
            with patch.object(Path,'read_text',read),patch('os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'Existing Doctor PID'):
                    assert_endpoints_unused({'output_dir':temp,'five_gpu':{'doctor_urls':['http://127.0.0.1:11731']}})
                kill.assert_not_called()

    def test_backend_requests_pinned_id_and_audits_mismatch_cost(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'TEST_REWARD_KEY':'synthetic'}):
            audit=Path(temp)/'reward_api_attempts.jsonl'
            backend=AuditedBackend(audit_path=audit,base_url='https://openrouter.ai/api/v1',model='family',
                revision='provider/family-fixed',api_key_env='TEST_REWARD_KEY',max_retries=2)
            sent=[]
            def response(req,timeout):
                sent.append(json.loads(req.data))
                value={'id':'req-1','model':'wrong-model','usage':{'cost':.01},'choices':[{'message':{'content':'{}'}}]}
                data=io.BytesIO(json.dumps(value).encode());data.headers={};return data
            with patch('urllib.request.urlopen',side_effect=response):
                result=backend.complete(ChatRequest('system','user','test',{}))
            self.assertEqual(sent[0]['model'],'provider/family-fixed')
            self.assertFalse(sent[0]['provider']['allow_fallbacks'])
            self.assertEqual(len(sent),1)
            self.assertEqual(result.status,'failed')
            self.assertEqual(json.loads(audit.read_text())['cost_usd'],.01)
            self.assertEqual(json.loads(audit.read_text())['request_id'],'req-1')

    def test_final_only_evaluation_resume_and_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);atomic_write_json(out/'run_state.json',{'case_cursor':120,'policy_sha256':'final','checkpoint':str(out/'checkpoints/checkpoint_000120')})
            cfg={'output_dir':temp,'schedule':{'cases':120},'five_gpu':{'eval_sha256':'test','eval_seed':0,'eval_cases':3,'checkpoint_evaluation_enabled':False}}
            generated=[]
            runner=SimpleNamespace()
            def run(config,eval_only):
                self.assertTrue(eval_only)
                queue=runner.EvaluationQueue(out/'evaluation',[{'case_id':f'test{i}'} for i in range(3)],{'seed':0})
                for n in (30,60,90,120):queue.register(out/f'checkpoints/checkpoint_{n:06d}',n,'final')
                def play(manifest,case,seed):
                    generated.append(seed);return {'outcome':None if seed==1 else 1,'num_discussion_rounds':2}
                queue.run_all_parallel(play,workers=1)
            runner.run=run
            first=evaluate_completed_run(runner,cfg,method_identity='method')
            self.assertEqual(first['accuracy'],2/3);self.assertEqual(first['unparsed'],1)
            evaluate_completed_run(runner,cfg,method_identity='method')
            self.assertEqual(sorted(generated),[0,1,2])
            self.assertEqual([p.name for p in (out/'evaluation').glob('checkpoint_*')],['checkpoint_000120'])
            with self.assertRaises(RuntimeError):evaluate_completed_run(runner,cfg,method_identity='different')

    def test_cost_unknown_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(summarize([temp])['total_reward_api_cost_usd'])

    def test_both_methods_reach_original_grpo_directly(self):
        for method in ('rar','online_rubrics'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as temp:
                model=TinyModel();trees,receipt,cases=fixture(model)
                generator=FakeBackend('generator');grader=FakeBackend('grader')
                reference=FakeReference() if method=='online_rubrics' else None
                engine=RubricEngine(method,generator,grader,Path(temp)/'rewards',weights={'w_outcome':0.,'w_rubric':1.},reference=reference)
                cfg=configuration(method,temp,receipt,engine,adapter_sha256(model),'synthetic')
                hook=RubricHook(cfg,receipt,engine,cases);runner=isolated_runner(hook)
                records=[{'case_id':t.case_id,'case_cursor':i,'tree':asdict(t),'policy_sha256':t.policy_sha256,
                    'prompt_sha256':cfg['rubric']['prompt_sha256'],'agent_targets':node_credit(1.,0.)(t,{},{})} for i,t in enumerate(trees)]
                beforetrees=copy.deepcopy([r['tree'] for r in records]);before=adapter_sha256(model)
                runner.experiment.check_batch(cfg,records)
                captured=[];original=grpo.grpo_loss
                def capture(m,rows,advantages,*args):
                    captured.extend(advantages);return original(m,rows,advantages,*args)
                with patch.object(grpo,'grpo_loss',side_effect=capture):
                    result=runner.update_batch(grpo.IntegratedGRPOUpdater(SimpleNamespace(torch=torch,model=model),cfg),records,before,lambda:adapter_sha256(model))
                self.assertTrue(result['optimizer_step']);self.assertEqual(result['optimizer_steps'],1)
                self.assertNotEqual(before,adapter_sha256(model));self.assertEqual(beforetrees,[r['tree'] for r in records])
                expected=[(i+1)/7-(2/7 if method=='online_rubrics' and i==0 else 0) for i in range(4)]
                mean=sum(expected)/4
                self.assertEqual(len(captured),20)
                for observed,wanted in zip(captured[:4],expected):self.assertAlmostEqual(observed,wanted-mean)
                self.assertEqual(len([x for x in generator.calls if x[0].endswith('initial_rubric')]),5)
                for kind,payload in generator.calls:
                    if kind.endswith('initial_rubric'):
                        self.assertIn('train_reference_answer',payload)
                        self.assertNotIn('reply',json.dumps(payload));self.assertNotIn('SECRET_PRIVATE',json.dumps(payload))
                for kind,payload in grader.calls:
                    self.assertEqual(set(payload),{'public_history','target_response','rubric'})
                    for hidden in ('TRAIN_GOLD_SECRET','SECRET_PRIVATE','outcome','INITIAL_ONLY_RESPONSE'):
                        self.assertNotIn(hidden,json.dumps(payload))
                if reference:
                    self.assertEqual(len(reference.calls),5)
                    pair=[p for k,p in generator.calls if k.endswith('pairwise_extract')][0]
                    self.assertEqual(pair['reference_response'],'INITIAL_ONLY_RESPONSE')
                    self.assertIn(pair['current_response'],['reply'+str(i) for i in range(4)])
                    self.assertEqual(records[0]['rubric_adaptation']['groups']['root']['online']['pair_count'],1)
                # Same artifacts on resume, including fixed selected current sibling.
                calls=(len(generator.calls),len(grader.calls))
                targets,report=engine.score(trees[0],cases['case0'])
                self.assertEqual(targets,records[0]['agent_targets']);self.assertEqual(report,records[0]['rubric_adaptation'])
                self.assertEqual(calls,(len(generator.calls),len(grader.calls)))
                bind(cfg);bind(copy.deepcopy(cfg),create=False)
                for field in ('actual_reward_weights','engine','initial_policy_sha256','base_model_revision'):
                    changed=copy.deepcopy(cfg);changed['comparison_method'][field]='changed'
                    with self.assertRaises(RuntimeError):bind(changed)

    def test_outcome_only_unchanged_and_no_engine(self):
        model=TinyModel();trees,receipt,cases=fixture(model)
        with tempfile.TemporaryDirectory() as temp:
            cfg=configuration('outcome_only',temp,receipt,None,adapter_sha256(model),'synthetic')
            records=[{'case_id':t.case_id,'case_cursor':i,'tree':asdict(t),'policy_sha256':t.policy_sha256,
                'prompt_sha256':cfg['rubric']['prompt_sha256'],'agent_targets':node_credit(1.,0.)(t,{},{})} for i,t in enumerate(trees)]
            original=copy.deepcopy(records)
            RubricHook(cfg,receipt,None,cases).check_batch(cfg,records)
            self.assertEqual(records,original)

    def test_future_changes_outcome_but_not_rubric(self):
        trees,_,cases=fixture(TinyModel());tree=trees[0]
        # Different numbers of descendant terminal conversations are permitted;
        # rubric is assigned to the candidate, never averaged over these leaves.
        extra=replace(tree.leaves[0],leaf_id='extra',outcome=0.,final_text='SECRET_FUTURE')
        changed=replace(tree,leaves=tuple([replace(l,outcome=None) if l.leaf_id=='3' else l for l in tree.leaves]+[extra]))
        with tempfile.TemporaryDirectory() as temp:
            generator=FakeBackend();grader=FakeBackend()
            engine=RubricEngine('rar',generator,grader,temp,weights={'w_outcome':0.,'w_rubric':1.})
            first,report=engine.score(tree,cases['case0']);n=len(generator.calls)+len(grader.calls)
            second,report2=engine.score(changed,cases['case0'])
            self.assertEqual(report,report2);self.assertEqual(n,len(generator.calls)+len(grader.calls))
            self.assertEqual(second['nodes']['0']['outcome_component'],.5)
            self.assertEqual(second['nodes']['0']['R_A'],first['nodes']['0']['R_A'])
            self.assertEqual(second['nodes']['0']['reward'],first['nodes']['0']['reward'])
            self.assertIsNone(second['nodes']['3']['reward'])

    def test_shared_initial_rubric_and_history_sensitive_cache(self):
        trees,_,cases=fixture(TinyModel());tree=trees[0]
        with tempfile.TemporaryDirectory() as temp:
            gen=FakeBackend();grade=FakeBackend()
            rar=RubricEngine('rar',gen,grade,Path(temp)/'rar',weights={'w_outcome':0.,'w_rubric':1.})
            initial=rar.initial(cases['case0'],tree.nodes[0]);calls=len(gen.calls)
            online=RubricEngine('online_rubrics',gen,grade,Path(temp)/'online',weights={'w_outcome':0.,'w_rubric':1.},
                reference=FakeReference(),initial_rubric_source=Path(temp)/'rar/cache/initial_rubrics.jsonl')
            self.assertEqual(initial,online.initial(cases['case0'],tree.nodes[0]));self.assertEqual(calls,len(gen.calls))
            later=replace(tree.nodes[0],depth=2,prefix=tree.nodes[0].prefix+(('agent','A1',1,'earlier'),('doctor','D2',2,'new needs')))
            online.initial(cases['case0'],later);self.assertEqual(calls+1,len(gen.calls))
            self.assertEqual(gen.calls[-1][1]['turn'],2)

    def test_real_frozen_reference_is_initial_after_current_update(self):
        model=TinyModel();trees,receipt,cases=fixture(model)
        with tempfile.TemporaryDirectory() as temp:
            initial=adapter_sha256(model);cfg=configuration('online_rubrics',temp,receipt,None,initial,'synthetic')
            produced=[];models=[]
            def factory(cfg,device):
                policy=SimpleNamespace(model=TinyModel());models.append(policy.model)
                def agent(case,prefix,turn,seed):
                    produced.append((case,prefix,turn,seed))
                    return {'completion_text':'initial reply','segments':[{'role':'private_assessment','completion_ids':[1,2]},
                        {'role':'visible_reply','completion_ids':[3]}]}
                policy.agent_turn=agent
                return policy  # No Doctor or continuation method exists.
            ref=FrozenInitialReference(cfg,temp,base_revision='synthetic',initial_policy_sha256=initial,device='cpu',factory=factory)
            with torch.no_grad():model.adapter.add_(1.)  # current policy has moved
            args=(cases['case0'],trees[0].nodes[0].prefix,1,19)
            with patch('torch.cuda.is_available',return_value=False):first=ref.response(*args)
            self.assertEqual(first,ref.response(*args));self.assertEqual(len(produced),1)
            self.assertTrue(all(not p.requires_grad for p in models[0].parameters()))
            self.assertEqual(tensor_hash(models[0],ref.names),initial);self.assertNotEqual(adapter_sha256(model),initial)
            self.assertNotIn('ground_truth',produced[0][0]);self.assertEqual(first['private_tokens'],2)
            ref.response(*args[:-1],20);self.assertEqual(len(produced),2)
            ref.close()
            resumed=FrozenInitialReference(cfg,temp,base_revision='synthetic',initial_policy_sha256=initial,device='cpu',factory=lambda *a:self.fail('Cache should avoid model loading'))
            self.assertEqual(resumed.response(*args),first)

    def test_bad_judge_is_not_cached_and_does_not_drop_leaf(self):
        trees,_,cases=fixture(TinyModel())
        with tempfile.TemporaryDirectory() as temp:
            backend=FakeBackend();engine=RubricEngine('rar',backend,backend,temp,weights={'w_outcome':0.,'w_rubric':1.})
            rubric=engine.initial(cases['case0'],trees[0].nodes[0])
            with patch.object(backend,'complete',return_value=ChatResult('ok','{"grades":[]}','fake','fake','test',{},0.,1)):
                with self.assertRaises(ValueError):engine.grade(cases['case0'],trees[0].nodes[0],rubric)
            self.assertEqual(len(engine.cache.store),0)
            self.assertIsInstance(engine.grade(cases['case0'],trees[0].nodes[0],rubric),float)

    def test_invalid_rubric_and_outcome_mixture_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):RubricEngine('rar',FakeBackend(),FakeBackend(),temp,weights={'w_outcome':.5,'w_rubric':.5})
        for weight in (True,0,11,-11,float('nan')):
            with self.assertRaises(ValueError):criteria({'criteria':[{'text':'test','weight':weight}]},signed=True)
        with self.assertRaises(KeyError):mixture({'agent_reward':{}})
        self.assertEqual(mixture({'agent_reward':{'w_rubric':.3,'w_outcome':.7}}),{'w_rubric':.3,'w_outcome':.7})
        with self.assertRaises(ValueError):public_history((('doctor','FINAL',1,'A'),))

    def test_launch_setup_reuses_local_recipe_without_external_manifest(self):
        from baselines.common.rubric_launch import setup
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)/'model';model.mkdir()
            (model/'generation_config.json').write_text('{"top_k":20}')
            runtime={'base_model_path':str(model),'initialization_registry':str(Path(temp)/'registry')}
            local={'path':str(model),'snapshot_revision':'b'*40,'identity_sha256':'c'*64}
            for method in ('outcome_only','rar','online_rubrics'):
                spec=json.loads(Path(f'baselines/{method}/method.json').read_text());spec['output_dir']=str(Path(temp)/method)
                _,synthetic,_=fixture(TinyModel())
                with patch('baselines.common.rubric_launch.audit_data',return_value=synthetic),patch('baselines.common.local_settings.model_identity',return_value=local):
                    cfg,receipt,engine,ref,errors=setup(method,spec,runtime)
                self.assertEqual(errors,[])
                self.assertFalse(cfg['rubric']['enabled']);self.assertFalse(cfg['prompt_update']['enabled'])
                self.assertEqual(cfg['policy']['base_model'],str(model))
                self.assertIsNone(cfg['comparison_method']['initial_policy_sha256'])
                self.assertEqual(cfg['comparison_method']['common_settings']['effective_agent_top_k'],20)
                self.assertEqual(cfg['agent_reward']['w_rubric'],0.)  # rollout diagnostic only
                if engine:
                    self.assertEqual(engine.weights,{'w_outcome':0.,'w_rubric':1.})
                    self.assertNotEqual(engine.generator.model,engine.grader.model)
                if ref:self.assertIsNone(ref.bound)
                self.assertFalse(Path(spec['output_dir']).exists())

    def test_initial_hash_shared_across_methods_and_checked_on_resume(self):
        from baselines.common.initialization import record_initial,expected_initial
        from baselines.common.pooled_reference import PooledInitialReference
        with tempfile.TemporaryDirectory() as temp:
            _,receipt,_=fixture(TinyModel())
            runtime={'initialization_registry':str(Path(temp)/'registry'),'local_model_identity':{'snapshot':'same'},'package_versions':{'torch':'same'}}
            configs=[configuration(m,Path(temp)/m,receipt,None,None,'same',runtime=runtime) for m in ('rar','online_rubrics')]
            self.assertIsNone(expected_initial(configs[0]))
            first=record_initial(configs[0],'a'*64)
            self.assertEqual(expected_initial(configs[1]),'a'*64)
            self.assertEqual(first,record_initial(configs[1],'a'*64))
            self.assertEqual(first,record_initial(configs[0],'a'*64))
            with self.assertRaises(RuntimeError):record_initial(configs[1],'b'*64)
            ref=PooledInitialReference(configs[1],Path(temp)/'ref',base_revision='same',initial_policy_sha256=None)
            identity=copy.deepcopy(ref.identity)
            ref.observe_initial('a'*64)
            self.assertEqual(identity,ref.identity)
            with self.assertRaises(RuntimeError):ref.observe_initial('b'*64)

    def test_legacy_model_loader_keeps_bf16_and_restores_keyword_adapter(self):
        import transformers
        from baselines.common.policy_loader import load_policy
        cls=transformers.AutoModelForCausalLM
        dtype=torch.bfloat16
        def factory(cfg,key,device):return cls.from_pretrained('synthetic',dtype=dtype)
        with patch.object(transformers,'__version__','4.51.1'),patch.object(cls,'from_pretrained',return_value='model') as native,patch('src.online_rl.runtime.SharedPolicy',side_effect=factory):
            before=cls.__dict__['from_pretrained']
            self.assertEqual(load_policy({},'key',device='cpu'),'model')
            native.assert_called_once_with('synthetic',torch_dtype=dtype)
            self.assertIs(cls.__dict__['from_pretrained'],before)
            native.side_effect=RuntimeError('synthetic failure')
            with self.assertRaises(RuntimeError):load_policy({},'key',device='cpu')
            self.assertIs(cls.__dict__['from_pretrained'],before)


if __name__=='__main__':unittest.main()
