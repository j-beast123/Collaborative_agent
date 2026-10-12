from types import SimpleNamespace as NS
import json
import pytest
from src.online_rl.tree import bank_delta as D, backends as B


def fixture(monkeypatch):
    monkeypatch.setattr(D.CB,'criteria',lambda r:r)
    cache=NS(call=lambda kind,identity,fn:fn())
    context={'key':'fake','model':'gemini-test','settings':{'max_tokens':100,'temperature':0}}
    trainer=NS(cache=cache,env=NS(generate_fn=NS(chunk_context=context)),cfg={'judges':{'execution_retry_attempts':3}})
    pending={'a':NS(rubric={'meaning':'candidate'})}
    delta=[{'bank_id':'bank_b','content_sha256':'b','rubric':{'rubric':{'meaning':'existing'}}}]
    return trainer,pending,delta


def row(checks=None):
    return {'candidate_id':'a','bank_id':'bank_b','checks':checks or {k:True for k in D.CHECKS},'rationale':'same public conditions and thresholds'}


def test_delta_api_sees_only_candidate_criteria_and_given_delta(monkeypatch):
    trainer,pending,delta=fixture(monkeypatch);calls=[]
    def request(key,model,system,user,settings,schema,name,role):
        calls.append(json.loads(user));return json.dumps({'decisions':[row()]}),{}
    monkeypatch.setattr(B,'_generator_json',request)
    result=D.compare(trainer,pending,delta)
    assert result['aliases']['a']['bank_id']=='bank_b'
    assert set(calls[0]['new_bank_entries'])=={'bank_b'}
    assert set(calls[0])=={'candidates','new_bank_entries'}
    assert D.validate_alias(result['aliases']['a'],'a',{'bank_b':delta[0]})==delta[0]


def test_partial_equivalence_never_merges_after_budget_exhausted(monkeypatch):
    trainer,pending,delta=fixture(monkeypatch);calls=[]
    checks={k:True for k in D.CHECKS};checks['same_score_boundaries']=False
    def request(*args):calls.append(1);return json.dumps({'decisions':[row(checks)]}),{}
    monkeypatch.setattr(B,'_generator_json',request)
    result=D.compare(trainer,pending,delta)
    assert len(calls)==3 and result['status']=='execution_unresolved' and result['aliases']=={}


def test_missing_candidate_is_execution_unresolved_not_equivalence(monkeypatch):
    trainer,pending,delta=fixture(monkeypatch)
    monkeypatch.setattr(B,'_generator_json',lambda *a:('{"decisions":[]}',{}))
    assert D.compare(trainer,pending,delta)['aliases']=={}


def test_no_delta_makes_no_api_call(monkeypatch):
    trainer,pending,delta=fixture(monkeypatch)
    def forbidden(*args):raise AssertionError('no delta must not call API')
    monkeypatch.setattr(B,'_generator_json',forbidden)
    assert D.compare(trainer,pending,[])['status']=='no_comparison_needed'


def test_stale_alias_target_or_candidate_hash_is_rejected(monkeypatch):
    trainer,pending,delta=fixture(monkeypatch)
    alias={'protocol':D.VERSION,'candidate_content_sha256':'a','bank_id':'bank_b','bank_content_sha256':'b',
           'checks':{k:True for k in D.CHECKS},'rationale':'same','comparison_sha256':'proof'}
    with pytest.raises(RuntimeError):D.validate_alias(alias,'different',{'bank_b':delta[0]})
    with pytest.raises(RuntimeError):D.validate_alias(alias,'a',{})
