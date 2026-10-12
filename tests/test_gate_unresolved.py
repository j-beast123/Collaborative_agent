from types import SimpleNamespace as NS
import pytest
from src.online_rl.tree import quality_gate as Q


def bind(monkeypatch,source):
    from src.online_rl.tree import pilot_evidence_scorer,cumulative_bank
    scorer=NS(protocol='test',score_behavior=lambda *a:{'source_criterion':source})
    monkeypatch.setattr(pilot_evidence_scorer,'evidence_scorer',lambda *a,**kw:scorer)
    monkeypatch.setattr(Q,'rubric_problems',lambda r:[])
    monkeypatch.setattr(cumulative_bank,'content_hash',lambda r:'hash')
    monkeypatch.setattr(Q,'public_gate_state',lambda *a:{'candidate_rubric':{},'case_context':{},
        'dialogue_before_target_turn':[],'source_agent_reply':{'text':'reply'},'selected_agent_turn_id':1})
    monkeypatch.setattr(Q,'validate_public_quality',lambda *a,**kw:[])
    monkeypatch.setattr(Q,'materialize_public_quality',lambda answer,state:(answer,[]))
    monkeypatch.setattr(Q,'public_rubric_field_catalog',lambda *a:{'name':'rubric'})
    monkeypatch.setattr(Q,'public_gate_reference_fields',lambda *a:{})
    answer={'checks':{'observability':{'passed':True},'boundary_coverage':{'passed':True}},'score_boundaries_clear':True}
    client=NS(model='fake',system='system',max_tokens=100,judge=lambda *a,**kw:({'candidate':answer},{}))
    gate=Q.public_mc_posthoc_gate(client,{'c':{}},protocol=Q.PUBLIC_GATE_PROTOCOL_V3,blind_scorer_override=scorer)
    return gate,NS(case_id='c',outcome='success',sha256='tin',leaf_id='l'),{'name':'rubric','scores':{'0':'none','1':'partial','2':'complete'}}


def test_gate_excludes_exhausted_source_without_indexing_missing_stage2(monkeypatch):
    source={'scoring_status':'execution_unresolved','hard':None,'probabilities':None,'stage2':None,
            'total_attempts':3,'reasons_by_attempt':[{'error':'C cannot be complete without M'}]*3}
    gate,tin,rubric=bind(monkeypatch,source)
    result=gate(tin,rubric)
    assert result['passed'] is False and result['status']=='execution_unresolved'
    assert result['reasons']==['source_execution_unresolved']
    assert result['source_anchor']['observed_level'] is None
    assert result['source_anchor']['matches'] is None
    assert result['evidence']['blind_source']['total_attempts']==3
    # The complete admission path rejects this candidate without raising.
    from src.online_rl.tree import posthoc
    from src.online_rl.tree.cache import CallCache
    monkeypatch.setattr(posthoc,'rubric_problems',lambda r:[])
    admitted=posthoc.generate_rubric('prompt','p',tin,lambda *a:{'rubric':rubric},CallCache(None),gate)
    assert not admitted.valid and admitted.gate['status']=='execution_unresolved'


@pytest.mark.parametrize('level,passed',[('2',True),('1',False),('0',False),('UNSCORABLE',False)])
def test_valid_source_levels_keep_the_existing_gate_rules(monkeypatch,level,passed):
    gate,tin,rubric=bind(monkeypatch,{'stage2':{'level':level},'evidence_validation':{}})
    result=gate(tin,rubric)
    assert result['passed'] is passed
    assert result['source_anchor']['observed_level']==level
