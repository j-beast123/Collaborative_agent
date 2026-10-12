import json
import os
from src.common.bank_monitor import BankCounters,activity
import monitor


def call(kind,key,value,identity=None):
    return {'kind':kind,'key':key,'identity':identity or {},'value':value}


def test_incremental_bank_counts_deduplicate_and_wait_for_complete_lines(tmp_path):
    p=tmp_path/'calls.jsonl';counter=BankCounters()
    generation=call('offline_chunk_generation','c',{'decisions':{'a':{'action':'CREATE'},'b':{'action':'REUSE_BANK'},'d':{'action':'PEER'}}},{'chunk_index':0})
    gate=call('posthoc_gate','g',{'passed':False})
    p.write_text(json.dumps(generation)+'\n'+json.dumps(gate)[:20])
    r=counter.read(tmp_path)
    assert r['chunks_generated']==1 and r['proposals']=={'CREATE':1,'REUSE_BANK':1,'PEER':1}
    assert r['gate_done']==0 and r['kept'] is None and not r['done']
    with p.open('a') as stream:
        stream.write(json.dumps(gate)[20:]+'\n'+json.dumps(generation)+'\n'+json.dumps(call('rerun','r',{'outcome':1}))+'\n')
    r=counter.read(tmp_path)
    assert r['chunks_generated']==1 and r['gate_rejected']==1 and r['reruns']==1
    assert counter.read(tmp_path)==r
    p.write_text('')
    assert counter.read(tmp_path)['gate_done']==0


def test_targets_and_final_keep_are_only_reported_from_case_records(tmp_path):
    (tmp_path/'bank_rollout.json').write_text(json.dumps({'feedback':[{'valid':True}]*49+[{'valid':False}]}))
    counter=BankCounters();r=counter.read(tmp_path)
    assert r['selected']==48 and r['chunks_total']==6 and r['kept'] is None
    (tmp_path/'bank_online.json').write_text(json.dumps({'selection':[{'retained':True},{'retained':False}],
        'cumulative_bank_input':{'entries':[{}]},'cumulative_bank_output':{'entries':[{},{}]},
        'offline_chunk_bank':{'selected_ids':['a','b']}}))
    r=counter.read(tmp_path)
    assert r['kept']==1 and r['removed']==1 and r['bank_added']==1 and r['done']
    assert r['selected']==2 and r['chunks_total']==1


def test_bank_activity_uses_coordinator_cache_not_remote_actor_role():
    row={'active':{'bank_online.trajectory_match_generate_verify':1,'cache.posthoc_gate':1,'role.rubric_scorer':1},
         'active_since':{'bank_online.trajectory_match_generate_verify':10,'cache.posthoc_gate':20}}
    assert activity(row,40)==('Gate 판정',20)
    assert activity({'active':{'role.rubric_scorer':1}},40)==(None,None)


def test_monitor_shows_bank_stage_counts_and_final_commit_read_only(tmp_path):
    batch=tmp_path/'batches/batch_000001';case=batch/'case_000000';case.mkdir(parents=True)
    (case/'bank_rollout.json').write_text(json.dumps({'feedback':[{'valid':True}]}))
    (case/'calls.jsonl').write_text(json.dumps(call('offline_chunk_generation','c',{'decisions':{'a':{'action':'CREATE'}}},{'chunk_index':0}))+'\n')
    live=tmp_path/'live';live.mkdir()
    (live/'gpu0.json').write_text(json.dumps({'pid':os.getpid(),'gpu':0,'case_index':0,'phase':'bank_online',
        'active':{'cache.posthoc_gate':1},'active_since':{'cache.posthoc_gate':100}}))
    before={str(p):p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}
    state=monitor.snapshot(tmp_path,now=130);text=monitor.render(state)
    assert '케이스 1: Gate 판정 | 현재 작업 00:00:30' in text
    assert 'chunk 생성 응답 1/1' in text and 'CREATE 1' in text
    assert '\n    최종 KEEP ' not in text
    assert before=={str(p):p.stat().st_mtime_ns for p in tmp_path.rglob('*') if p.is_file()}


def test_monitor_reports_bank_work_before_rollout_finishes(tmp_path):
    case=tmp_path/'batches/batch_000001/case_000000';case.mkdir(parents=True)
    (case/'stream_progress.json').write_text(json.dumps({'protocol':'ready_trajectory_chunk8_bank_v1',
        'ready_trajectories':19,'chunks_committed':1,'rollout_done':False,
        'stage':'Rubric 생성·Gate·verification','stage_started':100}))
    live=tmp_path/'live';live.mkdir()
    (live/'gpu0.json').write_text(json.dumps({'pid':os.getpid(),'gpu':0,'case_index':0,'phase':'bank_stream'}))
    state=monitor.snapshot(tmp_path,now=130);text=monitor.render(state)
    assert not state['rollout_cases'][0]['rollout_done']
    assert 'Reflection 준비 19 | bank chunk 반영 1 | rollout 진행 중' in text
    assert 'Rubric 생성·Gate·verification | 현재 작업 00:00:30' in text
    value=json.loads((case/'stream_progress.json').read_text());value['rollout_done']=True
    (case/'stream_progress.json').write_text(json.dumps(value))
    state=monitor.snapshot(tmp_path,now=135)
    assert state['files']['bank_rollout']==1
    assert state['rollout_cases'][0]['bank']['chunks_total']==3


def test_parallel_delta_phase_is_not_misattributed_from_shared_gpu_activity(tmp_path):
    case=tmp_path/'batches/batch_000001/case_000000';case.mkdir(parents=True)
    (case/'stream_progress.json').write_text(json.dumps({'protocol':'parallel_chunk_delta_commit_v1',
        'ready_trajectories':8,'current_chunk':1,'chunks_committed':0,'rollout_done':False,
        'stage':'추가 rubric 중복 비교 (Gemini)','stage_started':100}))
    live=tmp_path/'live';live.mkdir()
    (live/'gpu0.json').write_text(json.dumps({'pid':os.getpid(),'gpu':0,'case_index':0,'phase':'bank_stream',
        'active':{'cache.posthoc_gate':1},'active_since':{'cache.posthoc_gate':50}}))
    state=monitor.snapshot(tmp_path,now=130);text=monitor.render(state)
    assert '케이스 1: 추가 rubric 중복 비교 (Gemini) | 현재 작업 00:00:30' in text
    assert '케이스 1: Gate 판정' not in text
