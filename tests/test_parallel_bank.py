"""Fake transport only: no API, model download or GPU execution."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import threading
import time
import urllib.error

import pytest
from src.online_rl.tree import parallel_bank as P
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.b200_runner import load_config, preflight
from src.online_rl.integrated.records import sha256_json
from test_member_intervention_bank import sources, functions, CASE
from test_b200_partner import Outcomes
from src.online_rl.tree.partner_verification import verify_fixed_source


def partition(candidates):
    groups = []
    for stratum in ('failure', 'success'):
        ids = [c['candidate_id'] for c in candidates if c['stratum'] == stratum]
        if ids: groups.append({'candidate_ids': ids, 'checks': {k: True for k in P.CHECKS}, 'reason': 'Same publicly requested relation and score semantics'})
    return {'clusters': groups, 'excluded': []}


def fixture(cache, *, fail_gate=False, fail_generate=None, reverse=False):
    leaves, nodes, tins = sources()
    base, _, calls = functions(False)
    seen, gates = [], []
    def generate(prompt, tin):
        # Candidate tasks may persist results, but no finalized bank is visible.
        assert not any(v['kind'] == 'canonical_bank_plan' for v in cache.store.values())
        time.sleep((3-int(tin.leaf_id[-1]) if reverse else int(tin.leaf_id[-1])) * .005)
        if tin.leaf_id == fail_generate: raise RuntimeError('injected generation outage')
        return base(prompt, tin)
    generate.model = 'openai/gpt-6-luna'
    generate.settings = {'reasoning_effort': 'none'}
    def consolidate(candidates):
        seen.append(copy.deepcopy(candidates))
        assert len(calls) == 3
        return partition(candidates)
    consolidate.model = 'openai/gpt-6-luna'; consolidate.settings = {'reasoning_effort': 'none'}
    def gate(tin, rub):
        assert any(v['kind'] == 'canonical_bank_plan' for v in cache.store.values())
        gates.append(tin.leaf_id)
        if fail_gate: raise RuntimeError('injected gate outage')
        return {'passed': True}
    gate.model = 'unchanged-gate'
    args = (CASE, leaves, tins, 'fixed prompt', 'prompt', generate, consolidate, gate, cache)
    return args, nodes, calls, seen, gates


def test_parallel_generation_sorted_single_owner_provenance_and_resume(tmp_path):
    cache = CallCache(tmp_path/'calls.jsonl')
    args, nodes, calls, seen, gates = fixture(cache, reverse=True)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: P.build_bank(*args, chunk_size=3), range(2)))
    assert sha256_json(results[0][2]) == sha256_json(results[1][2])
    canon, assignments, report = results[0]
    assert len(calls) == 3 and len(seen) == 1 and len(gates) == 1
    assert len(canon) == 1 and report['duplicates_removed'] == 2
    assert cache.calls['canonical_bank_plan'] == cache.calls['case_bank_complete'] == 1
    assert [c['candidate_id'] for c in seen[0]] == sorted(c['candidate_id'] for c in seen[0])
    rubric = next(iter(canon.values()))
    assert rubric.rubric_id == min(c['candidate_id'] for c in seen[0])
    assert rubric.leaf_id == next(c['source_trajectory'] for c in seen[0] if c['candidate_id'] == rubric.rubric_id)
    assert len(report['bank_plan']['original_candidates']) == 3
    assert all(a.intervention_spec['selected_agent_turn_id'] == 1 for a in assignments)
    backend = Outcomes([1,1,0])
    leaves = {l.leaf_id:l for l in args[1]}
    first = verify_fixed_source(CASE,nodes,leaves,canon,assignments,backend=backend,run_seed=0,policy_sha256='policy',cache=cache)
    assert len(backend.calls) == 2 and next(iter(first.values())).decision == 'KEEP'
    fresh = CallCache(cache.path)
    args = (*args[:-1], fresh)
    restored = P.build_bank(*args, chunk_size=3)
    verify_fixed_source(CASE,nodes,leaves,restored[0],restored[1],backend=backend,run_seed=0,policy_sha256='policy',cache=fresh)
    assert len(backend.calls)==2 and len(calls)==3 and len(seen)==len(gates)==1
    # A different completion order yields exactly the same consolidation input/source.
    other = fixture(CallCache(None), reverse=False)
    other_result = P.build_bank(*other[0], chunk_size=3)
    assert other[3] == seen
    assert other_result[2]['bank_plan'] == report['bank_plan']


def test_resume_after_bank_finalized_before_gate(tmp_path):
    cache = CallCache(tmp_path/'calls.jsonl')
    args, nodes, calls, seen, gates = fixture(cache, fail_gate=True)
    with pytest.raises(RuntimeError,match='gate outage'): P.build_bank(*args,chunk_size=3)
    assert cache.calls['canonical_bank_plan']==1 and 'case_bank_complete' not in cache.calls
    fresh=CallCache(cache.path)
    def gate(*a):return {'passed':True}
    gate.model='unchanged-gate'
    def forbidden(*a):raise AssertionError('completed work rerun')
    for func in (args[5],args[6]):
        pass
    result=P.build_bank(*args[:5],args[5],args[6],gate,fresh,chunk_size=3)
    assert len(calls)==3 and len(seen)==1
    assert fresh.calls.get('canonical_bank_plan',0)==0 and fresh.calls['case_bank_complete']==1
    assert result[2]['canonical_rubrics']==1


def test_resume_partial_candidate_generation(tmp_path):
    cache=CallCache(tmp_path/'calls.jsonl')
    args,_,calls,seen,_=fixture(cache,fail_generate='L1')
    with pytest.raises(RuntimeError,match='generation outage'):P.build_bank(*args,chunk_size=3)
    assert {x[1] for x in calls}=={'L0','L2'} and not seen
    fresh=CallCache(cache.path)
    base,_,newcalls=functions(False)
    base.model=args[5].model;base.settings=args[5].settings
    result=P.build_bank(*args[:5],base,partition,lambda *a:{'passed':True},fresh,chunk_size=3)
    assert [x[1] for x in newcalls]==['L1'] and result[2]['canonical_rubrics']==1


@pytest.mark.parametrize('bad',['stratum','duplicate','missing','rewrite','uncertain'])
def test_invalid_consolidation_cannot_finalize(bad):
    data=[{'candidate_id':'a','stratum':'failure'},{'candidate_id':'b','stratum':'failure'}]
    result=partition(data)
    if bad=='stratum':data[1]['stratum']='success'
    if bad=='duplicate':result['clusters'][0]['candidate_ids'].append('a')
    if bad=='missing':result['clusters'][0]['candidate_ids'].pop()
    if bad=='rewrite':result['clusters'][0]['new_rubric']='invented'
    if bad=='uncertain':result['clusters'][0]['checks']['same_need']=False
    with pytest.raises(ValueError):P.validate_partition(result,data)


def test_unclear_source_is_excluded_without_gate_or_verification():
    cache=CallCache(None);args,_,calls,seen,gates=fixture(cache)
    def exclude(candidates):
        return {'clusters':[], 'excluded':[{'candidate_id':c['candidate_id'],'reason':'Unclear public evidence'} for c in candidates]}
    canon,assignments,report=P.build_bank(*args[:6],exclude,*args[7:],chunk_size=3)
    assert not gates and all(not r.valid for r in canon.values())
    assert all(a.assignment=='CREATE_REJECTED' for a in assignments)
    assert len(report['bank_plan']['exclusions'])==3


class Response:
    status=200
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self):return json.dumps(self.body).encode()


def test_real_backend_serializes_model_reasoning_and_rejects_substitution(monkeypatch):
    from src.online_rl.tree.backends import api_rubric_generator
    from src.common import usage
    sent=[];return_model=['openai/gpt-6-luna']
    def urlopen(req,timeout):
        payload=json.loads(req.data);sent.append((usage.current_stage(),payload))
        content={'clusters':[],'excluded':[]} if usage.current_stage()=='rubric_consolidate' else {'name':'Explain','applicability_condition':'any turn','required_behavior':'explain','scores':{'0':'none','1':'partial','2':'full'}}
        return Response({'model':return_model[0],'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],'usage':{}})
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    cfg=load_config();settings={'max_tokens':2048,'temperature':.7,'reasoning_effort':'none','strict_provider_parameters':True}
    gen=api_rubric_generator('fake-key',cfg['rubric']['generator_model'],settings)
    gen('fixed',next(iter(sources()[2].values())))
    con=P.api_consolidator('fake-key',cfg['rubric']['consolidate']);con([])
    assert {s for s,_ in sent}=={'rubric_generator','rubric_consolidate'}
    for _,payload in sent:
        assert payload['model']=='openai/gpt-6-luna'
        assert payload['reasoning']=={'effort':'none'}
        assert payload['provider']=={'require_parameters':True,'allow_fallbacks':False}
        assert 'models' not in payload
    return_model[0]='other-model'
    with pytest.raises(RuntimeError,match='no substitution'):con([])


def test_global_transport_limit_all_roles_and_retry_releases_slot(monkeypatch,tmp_path):
    from src.common import api_limiter, retry, usage
    from src.online_rl.tree.backends import _openrouter_json
    monkeypatch.setattr(api_limiter,'_limiter',None)
    limiter=api_limiter.configure(3)
    monkeypatch.setenv('SHERPA_HTTP_AUDIT',str(tmp_path/'attempts.jsonl'))
    attempts={};lock=threading.Lock();sleep_states=[]
    def urlopen(req,timeout):
        name=json.loads(req.data)['messages'][1]['content']
        with lock:attempts[name]=attempts.get(name,0)+1;n=attempts[name]
        time.sleep(.005)
        if n==1:raise urllib.error.HTTPError(req.full_url,429,'busy',{'Retry-After':'0.01'},io.BytesIO(b'rate limit'))
        return Response({'model':'openai/gpt-6-luna','choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]})
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    original=retry.Retry.__init__
    def init(self,*a,**kw):
        original(self,*a,**kw)
        # A retry wait is outside the physical request: its own slot is free.
        self.sleep=lambda delay:sleep_states.append(delay)
        self.log=lambda _:None
    monkeypatch.setattr(retry.Retry,'__init__',init)
    roles=['rubric_generator','rubric_consolidate','rubric_gate','rubric_scorer']
    def one(i):
        return _openrouter_json('fake','openai/gpt-6-luna','system',str(i),{'max_tokens':200,'temperature':.7,'reasoning_effort':'none'}, {},'test',roles[i%4])
    with ThreadPoolExecutor(12) as pool:list(pool.map(one,range(12)))
    assert limiter.peak==3 and limiter.active==0 and limiter.attempts==24
    assert len(sleep_states)==12 and set(sleep_states)=={.01}
    rows=[json.loads(line) for line in (tmp_path/'attempts.jsonl').read_text().splitlines()]
    assert len(rows)==24 and {r['role'] for r in rows}==set(roles)
    assert max(r['global_peak_in_flight'] for r in rows)==3
    assert sum(r['http_status']==429 for r in rows)==12
    assert all(r['limiter_wait_seconds']>=0 for r in rows)
    # Stronger single-request check: no unrelated request can mask a held slot.
    with limiter.slot():pass
    def wait_check(delay):assert limiter.active==0
    r=retry.Retry('check',log=lambda _:None);r.sleep=wait_check;r.wait('rate_limit','busy','0')


def test_config_preserves_training_contract_and_pins_new_identity(monkeypatch):
    monkeypatch.setattr('src.online_rl.tree.b200_runner.physical_devices',lambda:['0','1'])
    cfg=load_config()
    assert all(e.startswith('Missing file:') for e in preflight(cfg,resources=False))
    assert cfg['execution']['global_api_max_in_flight']==100
    assert cfg['rubric']['candidate_batch_size']==8
    assert cfg['rollout']['rounds']==cfg['rollout']['branching']==4
    assert cfg['two_gpu']['batch_cases']==5 and not cfg['prompt_update']['enabled']
    assert cfg['agent_reward']['w_outcome']==0 and cfg['agent_reward']['w_rubric']==1
    assert cfg['fixed_accuracy']['enabled']
    from src.online_rl.tree.trainer import model_roles
    roles=model_roles(cfg)
    assert roles['generator_settings']['reasoning_effort']==roles['consolidate_settings']['reasoning_effort']=='none'
    assert roles['rubric_gate']=='openai/gpt-5.6-luna' and roles['rubric_scorer']=='openai/gpt-6-luna'


def test_new_bank_is_connected_to_trainer_accuracy_reward_and_resume(tmp_path,monkeypatch):
    from test_witness_trainer import make
    from test_b200_need_accuracy import Scorer
    from src.online_rl.tree.fixed_accuracy import always_applicable, ID
    from src.online_rl.tree.team import node_credit
    from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
    tr,_,_=make(tmp_path,'rubric_only')
    cfg=load_config()
    tr.cfg['fixed_accuracy']=cfg['fixed_accuracy']
    tr.cfg['rubric'].update(bank_mode=P.VERSION,candidate_batch_size=3)
    tr.parallel_bank=True;tr.member_intervention_version=MEMBER_INTERVENTION_VERSION
    gen,_,calls=functions(False)
    tr.env=replace(tr.env,generate_fn=gen,assign_fn=None,consolidate_fn=partition,
                   scorer=always_applicable(Scorer(partner_match=False)))
    tr.st=replace(tr.st,credit_assignment=node_credit(0,1))
    def verify(case,tree,feedback,ruled,canon,assignments):
        assert ID not in canon
        assert all('source_provenance' in r.rubric for r in canon.values())
        return {}
    monkeypatch.setattr(tr,'_verify_bounded',verify)
    rec=tr.step(apply_update=False,commit=False)
    assert rec['rubric_bank']['candidate_bank']['version']==P.VERSION
    tr.validate_member_interventions(rec)
    assert len(rec['fixed_criteria'])==1 and rec['rubric_weights']=={ID:1.0}
    assert all(r['reward']==r['R_A'] for r in rec['agent_targets']['nodes'].values())
    count=len(calls)
    repeat=tr.step(apply_update=False,commit=False)
    assert len(calls)==count
    assert repeat['rubric_bank']==rec['rubric_bank']


def test_http_unsupported_reasoning_is_not_retried_or_downgraded(monkeypatch):
    calls=[]
    def urlopen(req,timeout):
        calls.append(json.loads(req.data))
        raise urllib.error.HTTPError(req.full_url,400,'unsupported',{},io.BytesIO(b'unsupported reasoning none'))
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    con=P.api_consolidator('fake',load_config()['rubric']['consolidate'])
    with pytest.raises(urllib.error.HTTPError):con([])
    assert len(calls)==1 and calls[0]['reasoning']=={'effort':'none'}


def test_generator_cache_separates_model_and_reasoning():
    from src.online_rl.tree.posthoc import generate_rubric
    leaves,_,tins=sources();gen,_,calls=functions(False);cache=CallCache(None)
    gen.model='old-model';gen.settings={'reasoning_effort':'low'}
    generate_rubric('P','sha',tins['L0'],gen,cache)
    gen.model='openai/gpt-6-luna';gen.settings={'reasoning_effort':'none'}
    generate_rubric('P','sha',tins['L0'],gen,cache)
    generate_rubric('P','sha',tins['L0'],gen,cache)
    assert len(calls)==2


def test_actual_retry_sleep_holds_no_slot(monkeypatch):
    from src.common import api_limiter,retry
    from src.collaboration.generator import doctor_call
    monkeypatch.setattr(api_limiter,'_limiter',None)
    limiter=api_limiter.configure(1)
    attempts=[];waits=[]
    def urlopen(req,timeout):
        attempts.append(1)
        if len(attempts)==1:raise urllib.error.HTTPError(req.full_url,429,'busy',{'Retry-After':'0'},io.BytesIO(b'busy'))
        return Response({'choices':[{'message':{'content':'ok'}}]})
    original=retry.Retry.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        def sleep(delay):
            assert limiter.active==0
            waits.append(delay)
        self.sleep=sleep;self.log=lambda _:None
    monkeypatch.setattr(retry.Retry,'__init__',init)
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    assert doctor_call('fake','same-model','s','u',30)=='ok'
    assert waits==[0] and limiter.attempts==2
