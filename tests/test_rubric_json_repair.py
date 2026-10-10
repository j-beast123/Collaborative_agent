"""No API/GPU: narrowly bounded JSON recovery retains strict rubric validation."""
import copy
import json
import pytest
from src.online_rl.tree import backends as B
from src.online_rl.tree.cache import CallCache
from test_rubric_quote_repair import example, setup


def broken_comma(good):
    return json.dumps(good,indent=2).replace(',\n  "source_case_id"','\n  "source_case_id"',1)


def test_local_missing_comma_restores_all_values_and_logs_exact_edit():
    _, good=example()
    broken=broken_comma(good)
    value,error,audit=B.parse_generator_json('```json\n'+broken+'\n```')
    assert value==good and error is None
    assert len(audit)==1 and len(audit[0]['edits'])==1
    repair=audit[0]
    assert repair['protocol']=='missing_structural_comma_v1'
    assert 'Expecting' in repair['original_error']
    assert json.loads(repair['repaired_json'])==good
    pos=repair['edits'][0]['offset']
    assert repair['repaired_json']==broken[:pos]+','+broken[pos:]


@pytest.mark.parametrize('bad', ['{"a":"unfinished', '{"a":1 "b":}', '{"a":"say "hello""}',
    '{"a":1,}', '{"a":1 "b":2 "c":3 "d":4 "e":5 "f":6}', '{"a":1 "a":2}'])
def test_ambiguous_truncated_duplicates_or_excessive_repairs_not_accepted(bad):
    value,error,audit=B.parse_generator_json(bad)
    assert value is None and error and not audit


def mock_generator(monkeypatch, responses):
    # Existing setup records real requested model/settings; replace its response
    # serialization only to deliver actual malformed bytes rather than a string.
    gen,prompt,_=setup(monkeypatch,[])
    sent=[]
    def send(key,model,system,user,settings,schema,name,role):
        sent.append(dict(model=model,system=system,user=user,settings=settings))
        raw=responses[len(sent)-1]
        return raw if isinstance(raw,str) else json.dumps(raw),{'usage':{'total_tokens':9}}
    monkeypatch.setattr(B,'_openrouter_json',send)
    return gen,prompt,sent


def test_missing_comma_costs_no_extra_request_and_resumes_success_cache(monkeypatch,tmp_path):
    tin,good=example();broken=broken_comma(good)
    gen,prompt,sent=mock_generator(monkeypatch,[broken])
    cache=CallCache(tmp_path/'calls.jsonl');identity={'settings':gen.settings,'leaf':tin.leaf_id}
    result=cache.call('rubric_generation',identity,lambda:gen(prompt,tin))
    assert result['parse_error'] is None and len(sent)==1
    assert result['raw']==broken and result['raw_attempts']==[broken]
    assert result['syntax_repairs'][0]['attempt']==1
    assert result['rubric']['source_provenance']['analysis']['source_score']==1
    assert CallCache(cache.path).call('rubric_generation',identity,lambda:pytest.fail('duplicate request'))==result


def test_syntax_failure_gets_bounded_final_repair_same_model_and_settings(monkeypatch):
    tin,good=example()
    invalid=json.dumps(good)[:-1]+',}'  # not a locally repairable missing comma
    gen,prompt,sent=mock_generator(monkeypatch,[invalid,invalid,good])
    result=gen(prompt,tin)
    assert result['parse_error'] is None and len(sent)==3
    assert 'fixing JSON syntax only' in sent[2]['user']
    assert result['usage_attempts']==[{'total_tokens':9}]*3
    assert len(result['validation_errors'])==2
    assert all(r['system']==prompt and r['model']=='gpt-6-luna' and r['settings']==sent[0]['settings'] for r in sent)


def test_still_invalid_json_stops_at_three_and_preserves_failure_audit(monkeypatch,tmp_path):
    tin,good=example();invalid=json.dumps(good)[:-1]+',}'
    gen,prompt,sent=mock_generator(monkeypatch,[invalid]*3)
    cache=CallCache(tmp_path/'calls.jsonl')
    with pytest.raises(B.RubricOutputIncomplete) as error:
        cache.call('rubric_generation',{},lambda:gen(prompt,tin))
    assert len(sent)==3 and not cache.store
    assert error.value.audit['raw_attempts']==[invalid]*3
    assert len(error.value.audit['validation_errors'])==3
    assert json.loads(cache.path.with_suffix('.errors.jsonl').read_text())['audit']==error.value.audit


def test_repaired_syntax_does_not_bypass_source_validation_or_add_fourth_call(monkeypatch):
    tin,good=example();invalid=json.dumps(good)[:-1]+',}'
    bad=copy.deepcopy(good);bad['analysis']['evidence_spans'][0]['quote']='Invented evidence'
    gen,prompt,sent=mock_generator(monkeypatch,[invalid,invalid,bad])
    with pytest.raises(B.RubricOutputIncomplete,match='exactly match'):gen(prompt,tin)
    assert len(sent)==3


def test_local_repair_cannot_accept_changed_source_id(monkeypatch):
    tin,bad=example();bad['source_case_id']='another-case'
    gen,prompt,sent=mock_generator(monkeypatch,[broken_comma(bad)]*2)
    with pytest.raises(B.RubricOutputIncomplete):gen(prompt,tin)
    assert len(sent)==2
