"""Offline invalid-partition repair, cache preservation and narrow resume acceptance."""
import copy
import json
from pathlib import Path
import pytest
from src.online_rl.tree import parallel_bank as P, b200_runner as R
from src.online_rl.tree.cache import CallCache
from src.online_rl.integrated.records import sha256_json
from test_parallel_bank import partition


def candidates():
    return [{'candidate_id':'A','stratum':'success'}, {'candidate_id':'B','stratum':'failure'}]


def setup(monkeypatch,tmp_path,values):
    cfg=R.load_config('configs/experiments/train120_max4_v7_rubric_only_b200.json')['rubric']['consolidate']
    requests=[]
    def send(key,model,system,user,settings,schema,name,role):
        requests.append({'model':model,'system':system,'user':json.loads(user),'settings':settings,'role':role})
        result=values[len(requests)-1]
        if isinstance(result,Exception):raise result
        return result if isinstance(result,str) else json.dumps(result),{}
    monkeypatch.setattr('src.online_rl.tree.backends._openrouter_json',send)
    audit=tmp_path/'audit.jsonl';monkeypatch.setenv('SHERPA_CONSOLIDATE_AUDIT',str(audit))
    return P.api_consolidator('never-log-this-key',cfg),requests,audit


def valid():
    return {'clusters':[partition([c])['clusters'][0] for c in candidates()], 'excluded':[]}


def test_mixed_strata_repaired_same_candidates_model_and_cache(tmp_path,monkeypatch):
    bad=valid();bad['clusters']=[{**bad['clusters'][0], 'candidate_ids':['A','B']}]; good=valid()
    con,sent,audit=setup(monkeypatch,tmp_path,[bad,good])
    cache=CallCache(tmp_path/'cache.jsonl');identity={'consolidator':con.settings,'candidates':candidates()}
    assert cache.call('case_consolidation',identity,lambda:con(candidates()))==good
    assert len(sent)==2 and sent[0]['user']==candidates()
    fix=sent[1]['user']
    assert fix['original_candidates']==candidates()
    assert 'success and failure' in fix['validation_error']
    assert json.loads(fix['previous_response'])==bad
    assert all(r['model']=='gpt-6-luna' and r['settings']['reasoning_effort']=='none' and r['settings']['api_provider']=='openai' for r in sent)
    assert sent[0]['system']==sent[1]['system']
    fresh=CallCache(cache.path)
    assert fresh.call('case_consolidation',identity,lambda:pytest.fail('completed valid partition repeated'))==good
    rows=[json.loads(line) for line in audit.read_text().splitlines()]
    assert [r['accepted'] for r in rows]==[False,True]
    assert 'never-log-this-key' not in audit.read_text()


@pytest.mark.parametrize('bad',['mixed', 'not json',{'clusters':{},'excluded':[]}, {'clusters':[],'excluded':[]}])
def test_bounded_failure_is_audited_not_cached_or_silently_changed(tmp_path,monkeypatch,bad):
    if bad == 'mixed':
        bad=valid();bad['clusters']=[{**bad['clusters'][0], 'candidate_ids':['A','B']}]
    con,sent,audit=setup(monkeypatch,tmp_path,[bad]*3)
    cache=CallCache(tmp_path/'cache.jsonl')
    with pytest.raises(RuntimeError,match='after two repair') as error:
        cache.call('case_consolidation',{},lambda:con(candidates()))
    assert len(sent)==3 and not cache.store
    assert len(error.value.audit['attempts'])==3
    assert (tmp_path/'cache.errors.jsonl').exists()
    assert all(not r['accepted'] for r in error.value.audit['attempts'])


def test_transport_failure_not_treated_as_format_repair(tmp_path,monkeypatch):
    con,sent,_=setup(monkeypatch,tmp_path,[RuntimeError('provider outage')])
    with pytest.raises(RuntimeError,match='provider outage'):con(candidates())
    assert len(sent)==1


def test_valid_first_response_and_settings_stay_legacy_compatible(tmp_path,monkeypatch):
    con,sent,_=setup(monkeypatch,tmp_path,[valid()])
    con(candidates());assert len(sent)==1
    cfg=R.load_config('configs/experiments/train120_max4_v7_rubric_only_b200.json')['rubric']['consolidate']
    expected={k:cfg[k] for k in ('max_tokens','temperature','reasoning_effort','strict_provider_parameters','api_provider','prompt_sha256')}
    assert con.settings=={**expected,'version':P.VERSION}


def test_actual_pinned_code_allows_only_this_operational_fix():
    receipt=json.loads((R.ROOT/'maintenance/consolidate_repair_compatibility.json').read_text())
    cfg=R.load_config();state={'config_sha256':sha256_json(cfg),'source_sha256':receipt['old_source_sha256']}
    current=R.code_identity()
    got=R.validate_resume_identity(state,cfg,current)
    assert got['effective_source_sha256']==current
    assert set(got['changed_files'])=={'src/online_rl/tree/parallel_bank.py','src/online_rl/tree/b200_runner.py'}
    for key in ('config_sha256','source_sha256'):
        with pytest.raises(RuntimeError,match='Config/code changed'):
            R.validate_resume_identity({**state,key:'different'},cfg,current)
    changed=copy.deepcopy(cfg);changed['optimizer']['learning_rate']=123
    with pytest.raises(RuntimeError,match='Config/code changed'):R.validate_resume_identity(state,changed,current)
    assert R.validate_resume_identity({**state,'source_sha256':current},cfg,current) is None


@pytest.mark.parametrize('edited',['src/online_rl/tree/parallel_bank.py','src/online_rl/tree/team.py'])
def test_compatibility_cannot_hide_any_other_edit(monkeypatch,edited):
    receipt=json.loads((R.ROOT/'maintenance/consolidate_repair_compatibility.json').read_text())
    cfg=R.load_config();state={'config_sha256':sha256_json(cfg),'source_sha256':receipt['old_source_sha256']}
    original=R.file_sha
    monkeypatch.setattr(R,'file_sha',lambda path:'different' if str(path.relative_to(R.ROOT))==edited else original(path))
    with pytest.raises(RuntimeError,match='Config/code changed'):
        R.validate_resume_identity(state,cfg,R.code_identity())
