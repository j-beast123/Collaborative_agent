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
    assert set(got['changed_files'])=={'src/online_rl/tree/parallel_bank.py','src/online_rl/tree/b200_runner.py','src/online_rl/tree/backends.py',
        'src/online_rl/hf_doctor.py','src/online_rl/doctor_batch.py','src/online_rl/tree/hf_doctor_setup.py'}
    for source in receipt.get('additional_sources', []):
        got=R.validate_resume_identity({**state,'source_sha256':source['old_source_sha256']},cfg,current)
        assert got['effective_source_sha256']==current
    for key in ('config_sha256','source_sha256'):
        with pytest.raises(RuntimeError,match='Config/code changed'):
            R.validate_resume_identity({**state,key:'different'},cfg,current)
    changed=copy.deepcopy(cfg);changed['optimizer']['learning_rate']=123
    with pytest.raises(RuntimeError,match='Config/code changed'):R.validate_resume_identity(state,changed,current)
    assert R.validate_resume_identity({**state,'source_sha256':current},cfg,current) is None


@pytest.mark.parametrize('edited',['src/online_rl/tree/parallel_bank.py','src/online_rl/tree/team.py','src/online_rl/doctor_batch.py'])
def test_compatibility_cannot_hide_any_other_edit(monkeypatch,edited):
    receipt=json.loads((R.ROOT/'maintenance/consolidate_repair_compatibility.json').read_text())
    cfg=R.load_config();state={'config_sha256':sha256_json(cfg),'source_sha256':receipt['old_source_sha256']}
    original=R.file_sha
    monkeypatch.setattr(R,'file_sha',lambda path:'different' if str(path.relative_to(R.ROOT))==edited else original(path))
    with pytest.raises(RuntimeError,match='Config/code changed'):
        R.validate_resume_identity(state,cfg,R.code_identity())


def test_coverage_error_identifies_missing_and_duplicate_locations():
    bad=valid();bad['clusters'][1]['candidate_ids']=['A']
    with pytest.raises(P.PartitionCoverageError) as error:P.validate_partition(bad,candidates())
    report=error.value.report
    assert report['expected_count']==report['returned_occurrences']==2
    assert report['unique_known_count']==1 and report['missing_ids']==['B']
    assert report['duplicate_ids']=={'A':['clusters[0].candidate_ids[0]','clusters[1].candidate_ids[0]']}
    assert 'missing=1' in str(error.value) and 'duplicate_ids=1' in str(error.value)


def test_targeted_correction_includes_exact_inventory_and_coverage(tmp_path,monkeypatch):
    bad=valid();bad['clusters'].pop()
    con,sent,audit=setup(monkeypatch,tmp_path,[bad,valid()])
    assert con(candidates())==valid()
    request=sent[1]['user']
    assert request['candidate_inventory']==candidates()
    assert request['membership_report']['missing_ids']==['B']
    assert request['membership_report']['duplicate_ids']=={}
    assert request['original_candidates']==candidates()
    assert 'COMPLETE' in request['instruction'] and 'not automatically exclude' in request['instruction']
    rows=[json.loads(s) for s in audit.read_text().splitlines()]
    assert rows[0]['membership_report']['missing_ids']==['B']
    assert rows[1]['membership_report']['missing_ids']==[]
    assert rows[0]['candidate_count']==2 and 'usage' in rows[0]


def test_diagnostic_covers_cluster_exclusion_overlap_unknowns_and_malformed():
    value={'clusters':[{'candidate_ids':['A','ZZ',7]}],
           'excluded':[{'candidate_id':'A','reason':'reason'},{'candidate_id':'ZZ','reason':'reason'}]}
    r=P.membership_report(value,candidates())
    assert r['missing_ids']==['B']
    assert r['duplicate_ids']['A']==['clusters[0].candidate_ids[0]','excluded[0].candidate_id']
    assert r['unknown_ids']['ZZ']==['clusters[0].candidate_ids[1]','excluded[1].candidate_id']
    assert r['invalid_memberships']==[{'location':'clusters[0].candidate_ids[2]','value':7}]
    assert P.membership_report(None,candidates())['missing_ids']==['A','B']


def test_256_candidates_exact_repair_and_no_implicit_singletons(tmp_path,monkeypatch):
    original=[{'candidate_id':f'candidate-{i:03d}','stratum':'failure'} for i in range(256)]
    good=partition(original);bad=copy.deepcopy(good)
    bad['clusters'][0]['candidate_ids'].remove('candidate-017')
    bad['clusters'][0]['candidate_ids'].append('candidate-099')
    con,sent,_=setup(monkeypatch,tmp_path,[bad,good])
    assert con(original)==good
    report=sent[1]['user']['membership_report']
    assert report['missing_ids']==['candidate-017']
    assert list(report['duplicate_ids'])==['candidate-099']
    assert len(sent[1]['user']['candidate_inventory'])==256
    assert len(good['clusters'])==1


def test_duplicate_input_ids_fail_before_api(tmp_path,monkeypatch):
    con,sent,_=setup(monkeypatch,tmp_path,[])
    with pytest.raises(ValueError,match='Duplicate input'):con([candidates()[0]]*2)
    assert sent==[]


def test_inspector_recovers_exact_old_failure_and_ignores_later_success(tmp_path):
    from maintenance.inspect_consolidation import inspect
    original=candidates();key=sha256_json(original)
    cache=CallCache(tmp_path/'batches/batch_000001/case_000000/calls.jsonl')
    cache.call('consolidate_input',{},lambda:original)
    bad=valid();bad['clusters'].pop()
    audit=tmp_path/'diagnostics/consolidate_validation.jsonl';audit.parent.mkdir()
    legacy={'input_sha256':key,'accepted':False,'attempt':3,'raw':json.dumps(bad),
            'validation_error':'Consolidate must partition all candidates exactly once'}
    audit.write_text(json.dumps(legacy)+'\n')
    before=cache.path.read_bytes()
    summary=inspect(tmp_path)
    assert summary['unresolved_inputs']==1
    assert summary['failures'][0]['membership_report']['missing_ids']==['B']
    assert cache.path.read_bytes()==before
    with audit.open('a') as stream:stream.write(json.dumps({**legacy,'accepted':True})+'\n')
    assert inspect(tmp_path)['unresolved_inputs']==0
