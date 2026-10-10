"""CPU-only scheduling, reward equivalence, credential routing and failure tests."""
import copy
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from baselines.common.test_methods import TinyModel, fixture, FakeBackend, FakeReference
from baselines.common.rubrics import RubricEngine
from baselines.parallel_api.engine import ParallelRubricEngine
from baselines.parallel_api.backend import OpenAIGraderBackend, ParallelAuditedBackend
from baselines.parallel_api.openai_profile import GRADE_PARAMS
from baselines.parallel_api.io import append_json
from src.judges.llm import ChatRequest


def tree_fixture(depth=3):
    trees, receipt, cases = fixture(TinyModel()); base = trees[0]; nodes = []; leaves = []
    def add(parent, level, prefix):
        for j in range(4):
            template = base.nodes[j]; nid = str(j) if parent is None else parent+'.'+str(j)
            node = replace(template, node_id=nid, parent_id=parent, depth=level, prefix=prefix,
                           trace={**template.trace, 'state_prompt_hash':str(prefix)})
            nodes.append(node)
            dialogue = prefix + (('agent','A'+str(level),level,node.response),)
            if level < depth:
                add(nid, level+1, dialogue+(('doctor','D'+str(level+1),level+1,'physician '+nid),))
            else:
                path = tuple('.'.join(nid.split('.')[:i]) for i in range(1, level+1))
                leaves.append(replace(base.leaves[j], leaf_id=nid, path=path, dialogue=dialogue))
    add(None, 1, base.nodes[0].prefix)
    return replace(base, nodes=tuple(nodes), leaves=tuple(leaves)), cases[base.case_id]


class ObservedBackend(FakeBackend):
    def __init__(self, role):
        super().__init__(role); self.lock=threading.Lock(); self.active=0; self.peak=0
    def complete(self, request):
        with self.lock: self.active+=1; self.peak=max(self.peak,self.active)
        try:
            time.sleep(.01)
            return super().complete(request)
        finally:
            with self.lock: self.active-=1


class ObservedReference(FakeReference):
    def __init__(self): super().__init__(); self.active=0; self.peak=0; self.thread_ids=[]
    def response(self,*args):
        self.active+=1; self.peak=max(self.peak,self.active); self.thread_ids.append(threading.get_ident())
        try:return super().response(*args)
        finally:self.active-=1


class ParallelTests(unittest.TestCase):
    def test_same_targets_reports_order_and_reference_seed(self):
        tree,case=tree_fixture(2)
        for method in ('rar','online_rubrics'):
            with tempfile.TemporaryDirectory() as temp:
                ref1=FakeReference() if method=='online_rubrics' else None
                ref2=ObservedReference() if ref1 else None
                serial=RubricEngine(method,FakeBackend('generator'),FakeBackend('grader'),Path(temp)/'serial',
                    weights={'w_outcome':0.,'w_rubric':1.},reference=ref1)
                gen=ObservedBackend('generator'); grade=ObservedBackend('grader')
                parallel=ParallelRubricEngine(method,gen,grade,Path(temp)/'parallel',
                    weights={'w_outcome':0.,'w_rubric':1.},reference=ref2,api_workers=20,grader_workers=100)
                expected=serial.score(tree,case); observed=parallel.score(tree,case)
                self.assertEqual(observed,expected)
                self.assertGreater(grade.peak,1); self.assertLessEqual(grade.peak,100)
                if ref2:
                    self.assertEqual(ref1.calls,ref2.calls);self.assertEqual(ref2.peak,1)
                    self.assertEqual(set(ref2.thread_ids),{threading.get_ident()})
                    self.assertIs(parallel.reference,ref2)
                calls=(len(gen.calls),len(grade.calls))
                self.assertEqual(parallel.score(tree,case),expected)
                self.assertEqual(calls,(len(gen.calls),len(grade.calls)))

    def test_limits_twenty_generators_and_hundred_judges(self):
        tree,case=tree_fixture(3)
        with tempfile.TemporaryDirectory() as temp:
            gen=ObservedBackend('generator')
            class WideGrader(ObservedBackend):
                def __init__(self):super().__init__('grader');self.release=threading.Event()
                def complete(self, request):
                    with self.lock:
                        self.active+=1;self.peak=max(self.peak,self.active)
                        if self.active>=30:self.release.set()
                    try:
                        if not self.release.wait(10):raise RuntimeError('Judge concurrency did not reach 30')
                        return FakeBackend.complete(self,request)
                    finally:
                        with self.lock:self.active-=1
            grade=WideGrader()
            engine=ParallelRubricEngine('rar',gen,grade,temp,weights={'w_outcome':0.,'w_rubric':1.},
                api_workers=20,grader_workers=100)
            targets,_=engine.score(tree,case)
            self.assertEqual(len(targets['nodes']),84)
            self.assertGreater(gen.peak,1);self.assertLessEqual(gen.peak,20)
            self.assertGreater(grade.peak,20);self.assertLessEqual(grade.peak,100)

    def test_failure_drains_tasks_without_returning_reward(self):
        tree,case=tree_fixture(2)
        class Failed(ObservedBackend):
            def complete(self, request):
                result=super().complete(request)
                if json.loads(request.user).get('target_response')=='reply2':
                    return replace(result,status='failed',text=None,error='synthetic')
                return result
        with tempfile.TemporaryDirectory() as temp:
            grader=Failed('grader');ref=ObservedReference()
            engine=ParallelRubricEngine('online_rubrics',ObservedBackend('generator'),grader,temp,
                weights={'w_outcome':0.,'w_rubric':1.},reference=ref)
            with self.assertRaisesRegex(RuntimeError,'synthetic'):engine.score(tree,case)
            self.assertEqual(grader.active,0);self.assertIs(engine.reference,ref)

    def test_jsonl_concurrent_long_rows_are_intact(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'audit.jsonl'
            with ThreadPoolExecutor(max_workers=100) as pool:
                list(pool.map(lambda i:append_json(path,{'index':i,'text':'한'*10000}),range(200)))
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(sorted(r['index'] for r in rows),list(range(200)))
            self.assertTrue(all(len(r['text'])==10000 for r in rows))

    def test_openai_key_endpoint_and_request_schema(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'OPENAI_API_KEY':'fake-openai','OPENROUTER_API_KEY':'fake-router'}):
            backend=OpenAIGraderBackend(audit_path=Path(temp)/'audit.jsonl')
            seen=[]
            def reply(req, timeout):
                seen.append((req.full_url,req.get_header('Authorization'),json.loads(req.data)))
                stream=io.BytesIO(json.dumps({'id':'req-test','model':'gpt-5.6-luna','usage':{'prompt_tokens':5,'completion_tokens':2},
                    'choices':[{'message':{'content':'{"grades":[]}'}}]}).encode());stream.headers={};return stream
            with patch('urllib.request.urlopen',side_effect=reply):
                result=backend.complete(ChatRequest('system','user','grade',GRADE_PARAMS))
            self.assertEqual(result.status,'ok')
            url,auth,payload=seen[0]
            self.assertEqual(url,'https://api.openai.com/v1/chat/completions');self.assertEqual(auth,'Bearer fake-openai')
            self.assertEqual(payload['model'],'gpt-5.6-luna');self.assertEqual(payload['reasoning_effort'],'medium')
            self.assertEqual(payload['max_completion_tokens'],4096)
            for key in ('provider','usage','max_tokens','temperature'):self.assertNotIn(key,payload)
            text=(Path(temp)/'audit.jsonl').read_text();self.assertNotIn('fake-openai',text);self.assertNotIn('fake-router',text)
            row=json.loads(text);self.assertEqual(row['provider'],'openai');self.assertIsNone(row['cost_usd'])

    def test_rate_limit_retry_is_bounded_and_audited(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'OPENAI_API_KEY':'fake'}):
            backend=OpenAIGraderBackend(audit_path=Path(temp)/'audit.jsonl');backend.sleep=lambda _:None
            def fail(req,timeout):raise urllib.error.HTTPError(req.full_url,429,'rate limit',{},io.BytesIO(b'limited'))
            with patch('urllib.request.urlopen',side_effect=fail):
                result=backend.complete(ChatRequest('s','u','grade',GRADE_PARAMS))
            self.assertEqual(result.status,'failed');self.assertEqual(result.attempts,3)
            self.assertEqual(len((Path(temp)/'audit.jsonl').read_text().splitlines()),3)

    def test_parallel_rewards_reach_original_grpo(self):
        from functools import partial
        from baselines.common.test_methods import MethodTests
        with patch('baselines.common.test_methods.RubricEngine',partial(ParallelRubricEngine,api_workers=20,grader_workers=100)):
            MethodTests().test_both_methods_reach_original_grpo_directly()

    def test_shared_registry_prevents_duplicate_initial_generation(self):
        tree,case=tree_fixture(1)
        with tempfile.TemporaryDirectory() as temp:
            gen=ObservedBackend('generator'); registry=Path(temp)/'shared'
            engines=[ParallelRubricEngine('rar',gen,FakeBackend('grader'),Path(temp)/str(i),
                weights={'w_outcome':0.,'w_rubric':1.},initial_registry=registry) for i in range(2)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                values=list(pool.map(lambda engine:engine.initial(case,tree.nodes[0]),engines))
            self.assertEqual(values[0],values[1]);self.assertEqual(len(gen.calls),1)

    def test_missing_openai_key_names_correct_environment(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'OPENROUTER_API_KEY':'fake'},clear=True):
            backend=OpenAIGraderBackend(audit_path=Path(temp)/'audit.jsonl')
            with self.assertRaisesRegex(RuntimeError,'OPENAI_API_KEY'):backend.complete(ChatRequest('s','u','grade',GRADE_PARAMS))
            self.assertFalse((Path(temp)/'audit.jsonl').exists())


if __name__=='__main__':unittest.main()
