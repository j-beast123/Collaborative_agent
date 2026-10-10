"""Synthetic CPU-only wiring checks, not semantic or clinical validation."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import pytest
from scripts.online_rl.run_tree import make_scorer, make_gate, evidence_gate_identity
from src.online_rl.tree.experiment import load_config
from src.online_rl.tree.luna import LunaDecisionClient
from src.online_rl.tree.applicability import state_applicability
from src.online_rl.tree.cache import CallCache
from src.online_rl.tree.scoring import score_nodes
from src.common.config import IntegrationUnavailable
from test_scoring_batch import CASE, KEPT, RUBRIC, node

CFG = 'configs/experiments/train120_max4_v7_rubric_only_b200.json'

class Transport:
    def __init__(self, mode='normal'):
        self.calls=[]; self.mode=mode
    def __call__(self,url,payload,key,timeout):
        assert url == 'https://api.openai.com/v1/chat/completions'
        assert payload['model'] == 'gpt-6-luna' and 'max_completion_tokens' in payload
        self.calls.append(copy.deepcopy(payload)); p=json.loads(payload['messages'][1]['content'])
        assert 'correct_option' not in json.dumps(p)
        if 'responses' not in p:
            assert 'target_agent_reply' not in json.dumps(p)
            assert 'minimum_performance' not in json.dumps(p['items'])
            ans={}
            for rid,item in p['items'].items():
                prefix=p['state']['dialogue_before_target_reply'];t=prefix[0]
                ans[rid]={
                  'doctor_condition':[{'clause':item['doctor_condition'],'status':'supported','evidence':[{'turn_id':t['turn_id'],'quote':t['text']}],'inspected':[],'note':'Synthetic support.'}],
                  'agent_condition':[{'clause':item['agent_condition'],'status':'not_required','evidence':[],'inspected':[],'note':'No extra state required.'}]}
        else:
            ans={}
            for response in p['responses']:
                reply=response['reply'];level=reply[-1] if reply[-1] in '012' else '2'
                ans[response['response_id']]={}
                for rid,r in p['rubric_items'].items():
                    refs=['score_anchors.0','score_anchors.1','score_anchors.2']
                    checks=[{'ref':ref,'met':str(i)==level,'cites':[0] if str(i)==level and level!='0' else [],'why':'Synthetic match.'} for i,ref in enumerate(refs)]
                    ans[response['response_id']][rid]={'level':level,'quotes':[reply],'checks':checks,'gap':'' if level=='2' else 'Written completion missing.','defect':''}
        return {'model': 'gpt-6-luna', 'choices':[{'finish_reason':'length' if self.mode=='truncated' else 'stop','message':{'content':json.dumps({'answers':ans})}}], 'usage':{}}

def setup(mode='normal'):
    cfg=load_config(CFG); t=Transport(mode)
    client=LunaDecisionClient(model=cfg['judges']['scorer_model'],key='OFFLINE',system=Path(cfg['judges']['scorer_prompt_file']).read_text(),max_tokens=4096,post=t,api_provider='openai')
    return cfg,t,client,make_scorer(cfg,client)

def test_sibling_alias_shared_app_and_actual_messages():
    _,t,_,s=setup();cache=CallCache(None)
    a=state_applicability(CASE,node(0),RUBRIC,s,cache)
    b=state_applicability(CASE,node(1),{**RUBRIC,'rubric_id':'bank_alias'},s,cache)
    assert a==b and len(t.calls)==1
    assert 'Target reply' not in json.dumps(t.calls[0])
    c=state_applicability({**CASE,'question':'Different question'},node(1),RUBRIC,s,cache)
    assert len(t.calls)==2 and a['provenance']!=c['provenance']

def test_native_scoring_keeps_zero_and_sibling_vector():
    _,t,_,s=setup();nodes=[replace(node(i),response=f'Observed response {v}') for i,v in enumerate([0,1,2,2])]
    rows=score_nodes(CASE,nodes,[KEPT],s,CallCache(None))
    assert [r.score for r in rows]==[0,0.5,1,1]
    assert sum('responses' not in json.loads(x['messages'][1]['content']) for x in t.calls)==1
    for x in t.calls:
        kind='behavior' if 'responses' in json.loads(x['messages'][1]['content']) else 'applicability'
        assert Path(f'prompts/sherpa/compact_{kind}_semantics_v2.txt').read_text() in x['messages'][0]['content']
        assert 'intervention_instruction' not in x['messages'][1]['content']
    from src.online_rl.integrated.metrics import advantages
    rewards=[r.score for r in rows]
    assert rewards==[0,0.5,1,1]
    assert advantages(rewards, 'mean_center')==[-0.625,-0.125,0.375,0.375]

def test_truncation_is_error_never_score_or_na():
    _,_,_,s=setup('truncated')
    with pytest.raises(IntegrationUnavailable) as e:s.review_state(CASE,[{'speaker':'doctor','text':'Please compare.'}],[RUBRIC])
    assert 'complete' in str(e.value)

def test_gate_uses_pinned_prompt_and_blind_compact_scorer():
    cfg,_,client,_=setup();client.system=Path(cfg['judges']['gate_prompt_file']).read_text()
    gate=make_gate(cfg,client,[CASE])
    assert gate.settings['blind_source_protocol']
    assert gate.settings['protocol'] == 'public_mc_gate_v3'
    assert evidence_gate_identity(cfg)['contract_sha256']==gate.settings['contract_sha256']


def test_hidden_assessments_do_not_split_sibling_applicability_or_leak():
    _,t,_,s=setup(); cache=CallCache(None)
    nodes=[replace(node(i), trace={'pre_response_belief': {'private': f'SECRET_{i}'}}, response=f'Observed response {i%3}') for i in range(4)]
    rows=score_nodes(CASE,nodes,[KEPT],s,cache)
    assert len(rows)==4 and len(t.calls)==5
    assert 'SECRET_' not in json.dumps(t.calls)
    calls=len(t.calls)
    score_nodes(CASE,nodes,[KEPT],s,cache)
    assert len(t.calls)==calls


def test_v7_config_is_separate_and_preserves_shared_training(monkeypatch):
    from src.online_rl.tree.b200_runner import load_config as load, preflight, validate_resume_identity
    from src.online_rl.integrated.records import sha256_json
    from src.online_rl.tree.fixed_accuracy import scoring_book
    monkeypatch.setattr('src.online_rl.tree.b200_runner.physical_devices', lambda: ['0','1'])
    cfg,base=load(CFG),load()
    assert all(e.startswith('Missing file:') for e in preflight(cfg, resources=False))
    for key in ('data','policy','rollout','feedback','schedule','optimizer','two_gpu','verification','agent_reward','execution'):
        assert cfg[key]==base[key], key
    assert cfg['output_dir']!=base['output_dir']
    assert cfg['fixed_accuracy']=={'enabled':False} and scoring_book([],cfg)==[]
    assert scoring_book([KEPT],cfg)==[KEPT]
    assert not cfg['doctor_need']['enabled'] and not cfg['prompt_update']['enabled']
    assert cfg['judges']['intervention_audit_model']=='gpt-6-luna'
    assert cfg['rubric']['generation']['reasoning_effort']==cfg['rubric']['consolidate']['reasoning_effort']=='none'
    with pytest.raises(RuntimeError,match='Config/code changed'):
        validate_resume_identity({'config_sha256':sha256_json(base),'source_sha256':'code'},cfg,'code')


def test_no_rubric_never_uses_outcome_and_unequal_scores_drive_advantage():
    from src.online_rl.tree.team import node_credit
    from src.online_rl.tree.reward import leaf_outcomes
    from src.online_rl.tree.records import Tree, Leaf
    from src.online_rl.integrated.metrics import advantages
    nodes=tuple(node(i) for i in range(4))
    for outcomes in ([0,0,0,0],[1,0,1,0],[1,1,1,1]):
        leaves=tuple(Leaf(n.node_id,CASE['case_id'],(n.node_id,),(),'','A',o,'max_agent_turns') for n,o in zip(nodes,outcomes))
        tree=Tree(CASE['case_id'],'policy-sha',1,4,nodes,leaves)
        assert all(r['reward'] is None for r in node_credit(0,1)(tree,leaf_outcomes(tree),{})['nodes'].values())
        ra=dict(zip([n.node_id for n in nodes],[0,.5,1,1]))
        rewards=[r['reward'] for r in node_credit(0,1)(tree,leaf_outcomes(tree),ra)['nodes'].values()]
        assert advantages(rewards,'mean_center')==[-.625,-.125,.375,.375]


def test_fixed_instruction_audit_rejection_does_not_execute_trials(monkeypatch,tmp_path):
    from test_b200_partner import Outcomes, verify
    from src.online_rl.tree import instruction_audit as audit
    from src.online_rl.integrated.records import sha256_json
    backend=Outcomes([1,1,1]);backend.instruction_auditor=object(); calls=[]
    def cached(auditor,cache,policy):
        def judge(case,prefix,rubric,spec):
            return cache.call('fake_instruction',{'prefix':sha256_json(prefix),'spec':spec},
                lambda: calls.append(spec.copy()) or {'passed':False,'reasons':['conclusion_prescribed']})
        return judge
    monkeypatch.setattr(audit,'cached_instruction_auditor',cached)
    cache=CallCache(tmp_path/'calls.jsonl')
    result=verify(backend,cache)
    assert result.decision=='DROP' and result.stop_reason=='instruction_rejected_before_trials'
    assert backend.calls==[] and len(calls)==1
    verify(backend,CallCache(cache.path))
    assert backend.calls==[] and len(calls)==1


def test_public_mc_scores_and_inconsistent_mc_fail_closed():
    _,t,_,scorer=setup()
    rubric={**RUBRIC,'minimum_performance':'Explain at least one distinction.',
            'completion_condition':'Connect the explanation to the decision.', 'criterion_version':'public_behavior_app_alignment_v5'}
    app={'user_side_context':RUBRIC['applicability_condition'],'agent_side_state':'No additional Agent state required.'}
    rubric.pop('applicability_condition')
    rubric['applicability']=app
    rubric['scores']={k:{**app,'behavior':v} for k,v in RUBRIC['scores'].items()}
    kept=replace(KEPT,rubric=rubric)
    original=t.__class__.__call__
    def with_mc(self,url,payload,key,timeout):
        body=original(self,url,payload,key,timeout)
        packet=json.loads(payload['messages'][1]['content'])
        if 'responses' in packet:
            answer=json.loads(body['choices'][0]['message']['content'])
            for rows in answer['answers'].values():
                for row in rows.values():
                    for ref,met in [('minimum_performance',row['level']!='0'),('completion_condition',row['level']=='2')]:
                        row['checks'].append({'ref':ref,'met':met,'cites':[0] if met else [],'why':'Synthetic M/C support.'})
            body['choices'][0]['message']['content']=json.dumps(answer)
        return body
    from unittest.mock import patch
    with patch.object(Transport,'__call__',with_mc):
        nodes=[replace(node(i),response=f'Observed response {i%3}') for i in range(4)]
        result=score_nodes(CASE,nodes,[kept],scorer,CallCache(None))
        assert [r.score for r in result]==[0,.5,1,0]
    from src.online_rl.tree.compact_audit import validate_response
    behavior=t.calls[-1]
    body=with_mc(t,'https://api.openai.com/v1/chat/completions',behavior,'fake',10)
    answer=json.loads(body['choices'][0]['message']['content'])
    row=next(iter(next(iter(answer['answers'].values())).values()))
    row['checks'][-1].update(met=True,cites=[0])
    body['choices'][0]['message']['content']=json.dumps(answer)
    assert validate_response(behavior,body)['status']!='valid_execution_contract'


def test_v7_generator_preserves_partial_source_and_exact_definitions(monkeypatch):
    from rubric_test_fixtures import _tin,_output
    from src.online_rl.tree import backends as B
    from src.online_rl.tree.feedback_grounding import REFLECTION_KIND
    from src.online_rl.tree.parallel_bank import api_consolidator
    cfg=load_config(CFG)
    tin=replace(_tin('failure'),feedback_kind=REFLECTION_KIND,feedback=json.dumps({'helpful':[],'unresolved':[]}))
    obj=_output(tin);c=obj['candidate_rubric'];a=obj['analysis']
    obj['intervention_instruction']='Compare both considerations and connect them to the decision criterion.'
    c.update(required_behavior=c['scores']['2']['behavior'],minimum_performance='Make the requested comparison.',
             completion_condition='Connect the comparison to the stated decision criterion.',criterion_version='public_behavior_app_alignment_v5')
    a['source_score']=1
    a['feedback_check']['claim_reviews']=[]
    a['score_construction_check'].update(minimum_ref='candidate_rubric.minimum_performance',completion_ref='candidate_rubric.completion_condition',
        instruction_alignment={'target_level':2,'instruction_ref':'intervention_instruction','rationale':'Requires both M and C.'})
    requests=[]
    def transport(key,model,system,user,settings,schema,name,role):
        requests.append((model,settings,role,system))
        return json.dumps(obj if role=='rubric_generator' else {'clusters':[],'excluded':[]}),{}
    monkeypatch.setattr(B,'_openrouter_json',transport)
    gen=B.api_rubric_generator('offline',cfg['rubric']['generator_model'],B.generation_settings(cfg['rubric']))
    result=gen(Path(cfg['rubric']['generator_prompt_file']).read_text(),tin)
    assert result['parse_error'] is None
    assert result['rubric']['minimum_performance']==c['minimum_performance']
    assert result['rubric']['source_provenance']['analysis']['source_score']==1
    assert result['rubric']['source_provenance']['source_anchor_contract_mismatch'] is True
    api_consolidator('offline',cfg['rubric']['consolidate'])([])
    assert len(requests)==2
    for model,settings,role,prompt in requests:
        assert model=='gpt-6-luna' and settings['api_provider']=='openai' and settings['reasoning_effort']=='none'
    assert gen.settings['local_schema_sha256']==cfg['rubric']['generation']['local_schema_sha256']


def test_case_pipeline_without_any_keep_excludes_all_groups_and_resumes(tmp_path,monkeypatch):
    from test_witness_trainer import make
    from test_parallel_bank import functions, partition
    from test_b200_need_accuracy import Scorer
    from src.online_rl.tree.team import node_credit
    from src.online_rl.tree.member_intervention import MEMBER_INTERVENTION_VERSION
    tr,_,_=make(tmp_path,'rubric_only')
    cfg=load_config(CFG)
    tr.cfg['fixed_accuracy']=cfg['fixed_accuracy']
    tr.cfg['rubric'].update(bank_mode=cfg['rubric']['bank_mode'],candidate_batch_size=3)
    tr.parallel_bank=True;tr.member_intervention_version=MEMBER_INTERVENTION_VERSION
    gen,_,calls=functions(False)
    tr.env=replace(tr.env,generate_fn=gen,assign_fn=None,consolidate_fn=partition,scorer=Scorer(partner_match=False))
    tr.st=replace(tr.st,credit_assignment=node_credit(0,1))
    monkeypatch.setattr(tr,'_verify_bounded',lambda *args: {})
    record=tr.step(apply_update=False,commit=False)
    assert 'fixed_criteria' not in record and record['rubric_weights']=={}
    assert all(n['reward'] is None for n in record['agent_targets']['nodes'].values())
    count=len(calls)
    repeat=tr.step(apply_update=False,commit=False)
    assert count==len(calls) and record['rubric_bank']==repeat['rubric_bank']
